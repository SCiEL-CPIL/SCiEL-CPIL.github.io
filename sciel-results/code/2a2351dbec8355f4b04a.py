import numpy as np
import time

N = 400
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

def _thue_morse(n):
    """Generate the first n terms of the Thue-Morse sequence (0/1)."""
    result = np.zeros(n, dtype=np.int8)
    for i in range(n):
        result[i] = bin(i).count('1') % 2
    return result

def _autoconv(f):
    """Compute autoconvolution of f."""
    n = len(f)
    conv = np.zeros(2 * n - 1)
    for k in range(2 * n - 1):
        lo = max(0, k - n + 1)
        hi = min(n, k + 1)
        conv[k] = np.dot(f[lo:hi], f[k - hi + 1:k - lo + 1][::-1])
    return conv

def _score(f):
    """Compute the score: max(|autoconv|) / (sum(f))^2"""
    s = np.sum(f)
    if abs(s) < 1e-10:
        return np.inf
    conv = _autoconv(f)
    return np.max(np.abs(conv)) / (s * s)

def _local_search_binary(f, max_iter=5000):
    """Local search flipping individual bits to improve score."""
    f = f.copy()
    current_score = _score(f)
    n = len(f)
    
    for _ in range(max_iter):
        if time.time() > DEADLINE:
            break
        best_flip = -1
        best_score = current_score
        # Random sampling of positions to flip
        candidates = np.random.choice(n, size=min(50, n), replace=False)
        for i in candidates:
            f[i] = 1 - f[i]
            new_score = _score(f)
            f[i] = 1 - f[i]
            if new_score < best_score:
                best_score = new_score
                best_flip = i
        if best_flip >= 0:
            f[best_flip] = 1 - f[best_flip]
            current_score = best_score
        else:
            break
    
    return f, current_score

def construct():
    # Start with Thue-Morse sequence mapped to {-1, 1}
    # Thue-Morse has low autocorrelation peaks
    tm = _thue_morse(N)
    f = 1 - 2 * tm  # Map 0->1, 1->-1
    
    # Ensure sum is non-zero; if sum is 0, flip one element
    if abs(np.sum(f)) < 1e-10:
        f[0] = -f[0]
    
    # Initial score
    current_score = _score(f)
    best_f = f.copy()
    best_score = current_score
    
    # Local search to refine
    start_time = time.time()
    while time.time() < DEADLINE:
        f, score = _local_search_binary(best_f, max_iter=1000)
        if score < best_score:
            best_score = score
            best_f = f.copy()
        else:
            break
    
    return best_f
