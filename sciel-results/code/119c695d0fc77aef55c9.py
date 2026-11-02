"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _bound_from_c(c):
    """Compute max |c(k)| for k != 0 from autoconvolution array c of length 2N-1."""
    n = len(c) // 2 + 1
    left = c[:n-1]
    right = c[n:]
    nonzero = np.concatenate([left, right[1:]])
    return np.max(np.abs(nonzero))


def _compute_c(s):
    """Compute linear autoconvolution c(k) = sum_i s[i]*s[i+k]."""
    n = len(s)
    c = np.zeros(2*n - 1)
    for k in range(n):
        c[k] = np.dot(s[:n-k], s[k:])
        if k > 0:
            c[2*n-1-k] = c[k]
    return c


def construct():
    t0 = time.time()
    deadline = t0 + 100.0  # 20s safety margin
    
    rng = np.random.default_rng(999)
    
    # Build diverse initial candidates
    candidates = []
    
    # Random
    candidates.append(rng.choice([-1, 1], size=N))
    
    # Thue-Morse
    tm = np.array([1])
    while len(tm) < N:
        tm = np.concatenate([tm, -tm])
    candidates.append(tm[:N].copy())
    
    # Low-discrepancy: sign of sin with golden ratio
    gr = (1 + np.sqrt(5)) / 2
    x = (np.arange(N) * gr) % 1.0
    candidates.append(np.where(x < 0.5, 1, -1).astype(int))
    
    # Pseudo-random with seed
    candidates.append(rng.choice([-1, 1], size=N))
    
    # Evaluate and pick best start
    best_s, best_c, best_b = None, None, np.inf
    for s in candidates:
        c = _compute_c(s)
        b = _bound_from_c(c)
        if b < best_b:
            best_b, best_s, best_c = b, s.copy(), c.copy()
    
    s = best_s.copy()
    c = best_c.copy()
    cur_b = best_b
    
    # Local search
    while time.time() < deadline:
        i = rng.integers(0, N)
        old = s[i]
        
        # Delta autoconvolution from flipping s[i]
        dc = np.zeros(2*N - 1)
        # Positive lags: k = 0..N-1-i
        m = N - i
        if m > 0:
            dc[:m] -= 2 * old * s[i:i+m]
        # Negative lags: -k for k=1..i
        if i > 0:
            dc[2*N-1-i:2*N-1] -= 2 * old * s[0:i][::-1]
        
        nc = c + dc
        nb = _bound_from_c(nc)
        
        if nb < cur_b:
            s[i] = -old
            c = nc
            cur_b = nb
            if cur_b < best_b:
                best_b = cur_b
                best_s = s.copy()
    
    return best_s
# EVOLVE-BLOCK-END
