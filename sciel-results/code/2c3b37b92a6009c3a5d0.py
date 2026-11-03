import numpy as np
import time

N = 400
GRID = 4096  # must be >= N; power of 2
TIME_LIMIT = 110.0  # seconds, safety margin under 120

def _spectral_peak_from_P(P):
    # P: complex array length GRID
    mag2 = (P.real**2 + P.imag**2)
    return float(mag2.max())

def construct():
    t0 = time.time()
    # Precompute exponent vectors E[k, j] = exp(-2 pi i k j / GRID) for k in [0,N)
    # Shape (N, GRID) complex
    ks = np.arange(N)[:, None]
    js = np.arange(GRID)[None, :]
    E = np.exp(-2j * np.pi * ks * js / GRID)  # (N, GRID)
    # This is 400*4096 complex = 26MB, fine.
    
    def peak_of(a):
        P = np.fft.fft(a, GRID)
        return _spectral_peak_from_P(P), P
    
    # Objective value for a sequence (for reporting)
    def value_of(a):
        P = np.fft.fft(a, GRID)
        return _spectral_peak_from_P(P)
    
    best_a = None
    best_val = np.inf
    
    def consider(a):
        nonlocal best_a, best_val
        v = value_of(a)
        if v < best_val:
            best_val = v
            best_a = a.copy()
    
    # ---- Initial candidates ----
    rng = np.random.default_rng(12345)
    
    # 1) all ones (baseline)
    a_ones = np.ones(N)
    consider(a_ones)
    
    # 2) random +-1
    a_rand = rng.choice([-1.0, 1.0], size=N)
    consider(a_rand)
    
    # 3) Thue-Morse (for variety)
    def thue_morse(n):
        # returns +-1 Thue-Morse of length n
        bits = 0
        out = np.empty(n)
        # compute via parity of bit count
        idx = np.arange(n)
        parity = np.zeros(n, dtype=int)
        x = idx.copy()
        while x.max() > 0:
            parity ^= (x & 1)
            x >>= 1
        return 1.0 - 2.0 * parity  # parity 0 -> +1, parity 1 -> -1
    a_tm = thue_morse(N)
    consider(a_tm)
    
    # 4) Continuous relaxation via subgradient descent on b in R^N, then round.
    #    Objective: max_j |P_j|^2, P = fft(b, GRID).
    b = a_ones.copy()  # start from ones
    # Actually start from random to avoid the trivial all-ones basin
    b = rng.choice([-1.0, 1.0], size=N)
    lr = 0.05
    for it in range(2000):
        if time.time() - t0 > TIME_LIMIT - 40:  # leave 40s for local search
            break
        P = np.fft.fft(b, GRID)
        mag2 = P.real**2 + P.imag**2
        jstar = int(mag2.argmax())
        # subgradient w.r.t. b_k: 2 Re(conj(P[jstar]) * exp(-2 pi i k jstar/GRID))
        # exp(-2 pi i k jstar/GRID) for k in [0,N) = conj of E[k, jstar]? 
        # E[k,j] = exp(-2 pi i k j/GRID). So E[:, jstar] = exp(-2 pi i k jstar/GRID).
        vec = E[:, jstar]  # length N, = exp(-2 pi i k jstar/GRID)
        # dJ/db_k = 2 Re(conj(P[jstar]) * vec[k])
        grad = 2.0 * np.real(np.conj(P[jstar]) * vec)
        b = b - lr * grad
        # project to [-1,1] to keep bounded and aid rounding
        b = np.clip(b, -1.0, 1.0)
    a_cont = np.sign(b)
    a_cont[a_cont == 0] = 1.0
    consider(a_cont)
    
    # ---- Local search on +-1 sequences ----
    # We'll run local search from multiple starts: best_a, a_cont, a_rand, a_tm
    starts = [best_a, a_cont, a_rand, a_tm]
    
    def local_search(a0, budget):
        a = a0.copy()
        P = np.fft.fft(a, GRID)
        cur = _spectral_peak_from_P(P)
        no_improve_sweeps = 0
        max_no_improve = 6
        while time.time() - t0 < budget:
            improved = False
            order = rng.permutation(N)
            for k in order:
                if time.time() - t0 > budget:
                    break
                ak = a[k]
                # flip: a_k -> -a_k, delta P = -2 a_k E[k,:]
                new_P = P - 2.0 * ak * E[k, :]
                new_mag2 = new_P.real**2 + new_P.imag**2
                new_peak = float(new_mag2.max())
                if new_peak < cur - 1e-9:
                    a[k] = -ak
                    P = new_P
                    cur = new_peak
                    improved = True
            if not improved:
                no_improve_sweeps += 1
                if no_improve_sweeps >= max_no_improve:
                    break
            else:
                no_improve_sweeps = 0
        return a, cur
    
    # Run local search from each start, with shared deadline
    for s in starts:
        if time.time() - t0 > TIME_LIMIT - 5:
            break
        a_ls, v_ls = local_search(s, TIME_LIMIT - 5)
        if v_ls < best_val:
            best_val = v_ls
            best_a = a_ls
    
    # ---- Final: a bit more local search from the global best if time remains ----
    if time.time() - t0 < TIME_LIMIT - 3:
        a_final, v_final = local_search(best_a, TIME_LIMIT - 3)
        if v_final < best_val:
            best_val = v_final
            best_a = a_final
    
    # Return as a sequence of signed step heights (floats +-1)
    # Ensure exactly N elements, +-1
    out = best_a
    out = np.where(out >= 0, 1.0, -1.0)
    return out
