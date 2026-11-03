import numpy as np
import time

N = 400

def _autocorr(h):
    # Linear autocorrelation using FFT
    n = 2 * N
    F = np.fft.rfft(h, n=n)
    ac = np.fft.irfft(F * np.conj(F), n=n)[:N]
    return ac

def _hard_score(h):
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)

def _smooth_score(h, beta):
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    # Use squared values to emphasize peaks more strongly in the surrogate
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    if m == 0:
        return 0.0
    z = beta * (y - m)
    return m + float(np.log(np.sum(np.exp(z)))) / beta

def construct():
    try:
        t0 = time.time()
        # 150s limit, use 140s for safety
        deadline = t0 + 140.0
        rng = np.random.default_rng(20240517)
        
        # 1. Diverse Initializations
        idx = np.arange(N)
        inits = [
            rng.choice(np.array([-1.0, 1.0]), size=N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.sign(np.cos(2 * np.pi * idx / N)), # Simple periodic
            np.sign(np.cos(2 * np.pi * idx**2 / N)), # Chirp
            np.sign(np.sin(2 * np.pi * idx / N)),
        ]
        
        best_h = None
        best_hard = 1e9
        for h in inits:
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()
        
        h = best_h.copy()
        cur_hard = best_hard
        cur_smooth = _smooth_score(h, 50.0)
        
        # 2. Simulated Annealing
        T = 0.05
        T_min = 1e-4
        decay = 0.999
        beta = 50.0
        
        # Move types: 0=bit, 1=block_flip, 2=block_swap
        move_probs = [0.2, 0.5, 0.3]
        
        while time.time() < deadline - 5.0:
            r = rng.random()
            if r < move_probs[0]:
                # Single bit flip
                i = int(rng.integers(0, N))
                old = h[i]
                h[i] *= -1.0
            elif r < move_probs[0] + move_probs[1]:
                # Block flip
                size = int(rng.choice([1, 2, 4, 8, 16, 32, 64]))
                if size >= N:
                    size = N // 2
                start = int(rng.integers(0, N - size + 1))
                sl = slice(start, start + size)
                old = h[sl].copy()
                h[sl] *= -1.0
            else:
                # Block swap
                size = int(rng.choice([1, 2, 4, 8, 16, 32]))
                if size >= N // 2:
                    size = N // 4
                # Pick two non-overlapping blocks
                s1 = int(rng.integers(0, N - 2 * size + 1))
                s2 = int(rng.integers(s1 + size, N - size + 1))
                sl1 = slice(s1, s1 + size)
                sl2 = slice(s2, s2 + size)
                old1 = h[sl1].copy()
                old2 = h[sl2].copy()
                h[sl1], h[sl2] = old2, old1
            
            new_smooth = _smooth_score(h, beta)
            new_hard = _hard_score(h)
            
            dS = new_smooth - cur_smooth
            if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
                cur_hard = new_hard
                cur_smooth = new_smooth
                if new_hard < best_hard:
                    best_hard = new_hard
                    best_h = h.copy()
            else:
                # Revert
                if r < move_probs[0]:
                    h[i] = old
                elif r < move_probs[0] + move_probs[1]:
                    h[sl] = old
                else:
                    h[sl1] = old1
                    h[sl2] = old2
            
            T = max(T_min, T * decay)
            
        # 3. Greedy Local Polish
        h = best_h.copy()
        # Repeat until no improvement or time out
        for _pass in range(20):
            if time.time() > deadline - 1.0:
                break
            improved = False
            
            # Single bit flips
            for i in range(N):
                if time.time() > deadline - 1.0:
                    break
                old = h[i]
                h[i] *= -1.0
                new_hard = _hard_score(h)
                if new_hard < best_hard - 1e-12:
                    best_hard = new_hard
                    best_h = h.copy()
                    improved = True
                else:
                    h[i] = old
            
            # Small block flips
            for size in [2, 4, 8]:
                for start in range(0, N - size + 1):
                    if time.time() > deadline - 1.0:
                        break
                    sl = slice(start, start + size)
                    old = h[sl].copy()
                    h[sl] *= -1.0
                    new_hard = _hard_score(h)
                    if new_hard < best_hard - 1e-12:
                        best_hard = new_hard
                        best_h = h.copy()
                        improved = True
                    else:
                        h[sl] = old
            
            if not improved:
                break
        
        return best_h
    except Exception:
        return np.ones(N)
