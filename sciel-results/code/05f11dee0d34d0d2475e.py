"""Optimized program for second autocorrelation inequality task."""
import numpy as np
import time
from itertools import combinations

N = 50
DEADLINE = 120.0  # seconds

def _sidon_greedy(n, max_size=8):
    """Greedy Sidon set construction in [0, n-1]."""
    used_sums = set()
    S = []
    for x in range(n):
        ok = True
        for y in S:
            if x + y in used_sums:
                ok = False
                break
        if ok:
            for y in S:
                used_sums.add(x + y)
            used_sums.add(2 * x)
            S.append(x)
            if len(S) >= max_size:
                break
    return S

def _autoconvolve(h):
    """Compute autoconvolution of step heights.
    For step function on equal intervals, the autoconvolution at offset k
    is sum_i h_i * h_{i+k} for valid indices."""
    n = len(h)
    result = np.zeros(2 * n - 1)
    for k in range(-(n - 1), n):
        s = 0.0
        if k >= 0:
            for i in range(n - k):
                s += h[i] * h[i + k]
        else:
            for i in range(-k, n):
                s += h[i] * h[i - k]
        result[k + n - 1] = s
    return result

def _score(h):
    """Compute the ratio: (sum h^2)^2 / (sum h * max autoconv)."""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    ac = _autoconvolve(h)
    ac_max = np.max(ac)
    if ac_max < 1e-15:
        return 0.0
    return (l2_sq ** 2) / (l1 * ac_max)

def _optimize_heights(support, h_init, max_iter=200, lr=0.01):
    """Optimize heights on given support using projected gradient ascent."""
    n = N
    h = np.zeros(n)
    h[support] = h_init
    
    best_score = _score(h)
    best_h = h.copy()
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        
        # Compute gradient of score w.r.t. h
        # score = (sum h^2)^2 / (sum h * max_ac)
        # Use finite differences for gradient
        eps = 1e-6
        grad = np.zeros(n)
        for i in support:
            h_plus = h.copy()
            h_plus[i] += eps
            h_minus = h.copy()
            h_minus[i] -= eps
            if h_minus[i] < 0:
                h_minus[i] = 0
            s_plus = _score(h_plus)
            s_minus = _score(h_minus)
            grad[i] = (s_plus - s_minus) / (2 * eps)
        
        # Gradient ascent step
        h_new = h + lr * grad
        h_new = np.maximum(h_new, 0)  # Project to nonnegative
        
        # Normalize to keep scale reasonable
        if np.sum(h_new) > 0:
            h_new = h_new / np.sum(h_new) * np.sum(h)
        
        s_new = _score(h_new)
        if s_new > best_score:
            best_score = s_new
            best_h = h_new.copy()
        h = h_new
    
    return best_h, best_score

def construct():
    start_time = time.time()
    
    # Try different Sidon set sizes and patterns
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Try various support sizes
    for max_size in [3, 4, 5, 6, 7, 8]:
        if time.time() > DEADLINE - 10:
            break
        
        support = _sidon_greedy(N, max_size)
        if len(support) < 2:
            continue
        
        # Initialize with equal heights on support
        h_init = np.ones(len(support))
        h, score = _optimize_heights(support, h_init, max_iter=100, lr=0.05)
        
        if score > best_score:
            best_score = score
            best_h = h.copy()
    
    # Also try some heuristic patterns
    patterns = [
        # Alternating
        [i for i in range(0, N, 2)][:10],
        # Every 3rd
        [i for i in range(0, N, 3)][:10],
        # Every 4th
        [i for i in range(0, N, 4)][:10],
        # First few
        list(range(5)),
        # Spread out
        [i * N // 8 for i in range(8)],
    ]
    
    for support in patterns:
        if time.time() > DEADLINE - 10:
            break
        if len(support) < 2:
            continue
        # Ensure support is within bounds
        support = [s for s in support if 0 <= s < N]
        if len(support) < 2:
            continue
        
        h_init = np.ones(len(support))
        h, score = _optimize_heights(support, h_init, max_iter=80, lr=0.05)
        
        if score > best_score:
            best_score = score
            best_h = h.copy()
    
    # Return as list of floats
    return [float(x) for x in best_h]

