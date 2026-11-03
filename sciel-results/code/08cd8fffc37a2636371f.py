import time
import numpy as np

N = 400
L = 1024  # FFT length >= 2*N-1 = 799
_SEED = 12345

def _autocorr_fast(h):
    """Linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1."""
    F = np.fft.rfft(h, n=L)
    ac = np.fft.irfft(F * np.conj(F), n=L)[:N]
    return ac

def _hard_score_fast(ac):
    """Compute score from autocorrelation array, optimized for ±1 sequences."""
    # For ±1 sequences, ac[0] is always N
    return float(np.max(np.abs(ac[1:])) / N)

def _ac_delta_single_flip(h, i):
    """
    Compute the change in autocorrelation when flipping h[i] -> -h[i].
    Returns delta_ac of shape (N,).
    """
    delta_ac = np.zeros(N)
    hi = h[i]
    
    # Terms involving h[i] in ac[k] = sum_j h[j]h[j+k]
    # 1. j = i: term is h[i]h[i+k]. Flipping h[i] changes it to -h[i]h[i+k]. Delta = -2*h[i]*h[i+k].
    #    Valid for k such that 0 <= i+k < N => 0 <= k < N-i.
    # 2. j+k = i => j = i-k: term is h[i-k]h[i]. Flipping h[i] changes it to -h[i-k]h[i]. Delta = -2*h[i]*h[i-k].
    #    Valid for k such that 0 <= i-k < N => i-N < k <= i.
    
    # Range for h[i+k]: k in [0, N-i-1]
    if i < N:
        k_start = 0
        k_end = N - i
        if k_end > k_start:
            delta_ac[k_start:k_end] += -2.0 * hi * h[i + k_start:i + k_end]
            
    # Range for h[i-k]: k in [max(0, i-N+1), i]
    k_start = max(0, i - N + 1)
    k_end = min(i, N - 1) + 1
    if k_end > k_start:
        delta_ac[k_start:k_end] += -2.0 * hi * h[i - k_start:i - k_end + 1]
        
    return delta_ac

def _local_polish(h, deadline):
    """
    Greedy local search. Repeatedly flips bits if it reduces the peak sidelobe.
    Returns best_h and best_score.
    """
    h = h.copy()
    ac = _autocorr_fast(h)
    best_score = _hard_score_fast(ac)
    best_h = h.copy()
    
    # Precompute deltas for all i? No, h changes.
    # We will iterate.
    improved = True
    while improved:
        if time.time() > deadline - 5.0:
            break
        improved = False
        # Try flipping each bit
        for i in range(N):
            if time.time() > deadline - 5.0:
                return best_h, best_score
            
            delta_ac = _ac_delta_single_flip(h, i)
            ac_new = ac + delta_ac
            new_score = _hard_score_fast(ac_new)
            
            if new_score < best_score - 1e-12:
                # Accept flip
                h[i] *= -1.0
                ac = ac_new
                best_score = new_score
                best_h = h.copy()
                improved = True
                # Restart loop? Or continue?
                # Continuing is faster, but might miss interactions.
                # For speed, we just continue in the current pass.
                
    return best_h, best_score

def _sa_run(h_init, rng, deadline, T_start=0.01, T_min=1e-5, decay=0.999):
    """
    Simulated Annealing to escape local minima.
    """
    h = h_init.copy()
    ac = _autocorr_fast(h)
    cur_score = _hard_score_fast(ac)
    best_score = cur_score
    best_h = h.copy()
    
    T = T_start
    steps = 0
    
    while time.time() < deadline - 10.0:
        # Choose a random bit to flip
        i = int(rng.integers(0, N))
        
        delta_ac = _ac_delta_single_flip(h, i)
        ac_new = ac + delta_ac
        new_score = _hard_score_fast(ac_new)
        
        dS = new_score - cur_score
        
        if dS < 0:
            # Improvement
            accept = True
        else:
            # Accept with probability exp(-dS/T)
            if T > 1e-10:
                if rng.random() < np.exp(-dS / T):
                    accept = True
                else:
                    accept = False
            else:
                accept = False
        
        if accept:
            h[i] *= -1.0
            ac = ac_new
            cur_score = new_score
            
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        
        T = max(T_min, T * decay)
        steps += 1
        
        # Occasionally restart T if stuck?
        # Simple decay is usually fine for this scale.
        
    return best_h, best_score

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1."""
    tm = np.array([bin(i).count('1') % 2 for i in range(n)])
    return 1.0 - 2.0 * tm

def construct():
    try:
        t0 = time.time()
        # Safety margin: return by t0 + 145
        deadline = t0 + 145.0
        rng = np.random.default_rng(_SEED)
        
        # 1. Generate Seeds
        seeds = []
        
        # Thue-Morse
        seeds.append(_thue_morse(N))
        
        # Random
        seeds.append(rng.choice(np.array([-1.0, 1.0]), size=N))
        
        # Chirp (Quadratic Phase)
        idx = np.arange(N)
        seeds.append(np.sign(np.cos(2 * np.pi * idx**2 / (2 * N))))
        seeds.append(np.sign(np.cos(2 * np.pi * idx**2 / (N + 1))))
        
        # Sine
        seeds.append(np.sign(np.sin(2 * np.pi * idx / N)))
        
        # 2. Local Polish each seed
        best_h = None
        best_score = 1e9
        
        # Budget for local polish of seeds: ~10s
        lp_deadline = t0 + 10.0
        
        for s in seeds:
            if time.time() > lp_deadline:
                break
            h_polished, score = _local_polish(s, lp_deadline)
            if score < best_score:
                best_score = score
                best_h = h_polished.copy()
        
        # 3. SA from best local optimum
        # Budget for SA: ~120s
        sa_deadline = t0 + 135.0
        
        if best_h is not None:
            h_sa, score_sa = _sa_run(best_h, rng, sa_deadline)
            if score_sa < best_score:
                best_score = score_sa
                best_h = h_sa.copy()
        
        # 4. Final Local Polish
        # Budget for final polish: ~5s
        final_deadline = deadline
        
        if best_h is not None:
            h_final, score_final = _local_polish(best_h, final_deadline)
            if score_final < best_score:
                best_score = score_final
                best_h = h_final.copy()
        
        if best_h is None:
            return np.ones(N)
            
        return best_h

    except Exception:
        return np.ones(N)
