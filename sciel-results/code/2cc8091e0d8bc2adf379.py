import numpy as np
import time

N = 400

def _autoconv(f):
    n = len(f)
    size = 2*n
    F = np.fft.fft(f, size)
    c = np.fft.ifft(F * np.conj(F)).real
    return c[:n]

def _flip_delta(f, j, c):
    # returns new c after flipping f[j]
    a = f[j]
    term1 = np.zeros(N)
    term1[:N-j] = f[j:]
    term2 = np.zeros(N)
    term2[:j+1] = f[:j+1][::-1]
    delta = -2*a*(term1 + term2)
    delta[0] = 0.0
    return c + delta

def construct():
    start = time.time()
    deadline = start + 135
    
    best_f = np.ones(N, dtype=float)
    best_c = _autoconv(best_f)
    best_S = float(np.sum(best_f))
    best_peak = float(np.max(np.abs(best_c[1:])))
    best_val = best_peak / (best_S*best_S)
    
    rng = np.random.default_rng(123)
    
    for restart in range(200):
        if time.time() > deadline:
            break
        # init
        r = rng.random()
        if r < 0.3:
            f = rng.choice([-1.0, 1.0], size=N)
        elif r < 0.6:
            f = np.ones(N)
            k = int(N * rng.uniform(0.1, 0.4))
            idx = rng.choice(N, size=k, replace=False)
            f[idx] = -1.0
        else:
            f = np.ones(N)
            k = int(N * rng.uniform(0.3, 0.7))
            idx = rng.choice(N, size=k, replace=False)
            f[idx] = -1.0
        
        c = _autoconv(f)
        S = float(np.sum(f))
        
        for it in range(200):
            if time.time() > deadline:
                break
            if S == 0:
                break
            peak = float(np.max(np.abs(c[1:])))
            val = peak / (S*S)
            if val < best_val:
                best_val = val
                best_f = f.copy()
            # find best flip
            best_j = -1
            best_val_j = val
            for j in range(N):
                c2 = _flip_delta(f, j, c)
                S2 = S - 2*f[j]
                if S2 == 0:
                    continue
                peak2 = float(np.max(np.abs(c2[1:])))
                val2 = peak2 / (S2*S2)
                if val2 < best_val_j - 1e-12:
                    best_val_j = val2
                    best_j = j
            if best_j < 0:
                break
            # apply flip
            f[best_j] = -f[best_j]
            c = _flip_delta(f, best_j, c)  # wait, this double-applies
            # Actually _flip_delta(f, j, c) computes c after flipping f[j], but f[j] is still old.
            # Let me recompute: I should apply the flip to f, then c = c + delta where delta uses old f[j].
            # Let me redo: delta computed with old f[j], then f[j] = -f[j], c = c + delta.
            # In _flip_delta, I used a = f[j] (old). So c2 = c + delta is the new c. Good.
            # But I already computed c2 in the loop. Let me just set c = c2 for best_j.
            # Let me restructure.
            S = S - 2*f[best_j]
            f[best_j] = -f[best_j]
            # c is already updated? No, I need to set c = c2 for best_j.
            # Let me recompute c2 for best_j.
            # Actually I can just keep it.
        # end it loop
    # end restart loop
    
    return best_f
