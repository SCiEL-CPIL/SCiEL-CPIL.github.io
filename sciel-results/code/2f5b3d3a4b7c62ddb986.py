import numpy as np
import time

N = 400

def _score(x):
    """max_{k != 0} |autocorr[k]| / (sum x)^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    n = len(x)
    f = np.fft.fft(x, n=2*n)
    ac = np.fft.ifft(f * np.conj(f)).real[:n]
    # Exclude k=0
    peak = np.max(np.abs(np.concatenate([ac[1:], ac[-(n-1):][::-1]]))) if n > 1 else 0
    # Actually, ac is the linear autocorrelation, ac[k] for k=0..n-1
    # We want max over k=1..n-1 of |ac[k]|
    if n > 1:
        peak = np.max(np.abs(ac[1:]))
    else:
        peak = 0
    return peak / (s * s)

def construct():
    start = time.time()
    deadline = start + 110
    
    best_x = np.ones(N)
    best_val = _score(best_x)
    
    np.random.seed(42)
    
    # Random restarts with {-1, 1}
    for i in range(30000):
        if time.time() > deadline:
            break
        x = np.random.choice([-1.0, 1.0], size=N)
        val = _score(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # Local search
    for i in range(100000):
        if time.time() > deadline:
            break
        x = best_x.copy()
        idx = np.random.randint(N)
        x[idx] *= -1.0
        val = _score(x)
        if val < best_val:
            best_val = val
            best_x = x
    
    return best_x
