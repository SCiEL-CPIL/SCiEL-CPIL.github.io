import numpy as np
import time

# ---- Task constants (interface) ----
N = 50
DELTA = 0.5 / N
_XC = -0.25 + (np.arange(N) + 0.5) * DELTA

_DEADLINE = 145.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _ratio(h):
    """Raw objective: L2^2 / (L1 * max_autoconv)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return -np.inf
    l1 = np.sum(h)
    if l1 <= 0.0:
        return 0.0
    l2_sq = np.dot(h, h)
    if l2_sq <= 0.0:
        return 0.0
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 0.0:
        return 0.0
    return l2_sq / (l1 * m)

def _is_sidon_set(S):
    """Check if S is a Sidon set (all pairwise sums i+j for i<=j are unique)."""
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_sidon(candidates):
    """Greedy: add elements in the given order if no conflict."""
    S = []
    sums = set()
    for c in candidates:
        conflict = False
        if c + c in sums:
            conflict = True
        else:
            for s in S:
                if c + s in sums:
                    conflict = True
                    break
            if not conflict:
                S.append(c)
                sums.add(c + c)
                for s in S:
                    sums.add(c + s)
    return S

def _find_max_sidon_set(N, max_restarts=5000):
    """Find the largest Sidon set in [0, N) using deterministic and random greedy."""
    best_set = []
    
    # Deterministic orderings
    for ordering in [list(range(N)), list(range(N-1, -1, -1))]:
        S = _greedy_sidon(ordering)
        if len(S) > len(best_set):
            best_set = S[:]
    
    # Random orderings with multiple seeds
    for seed in range(max_restarts):
        if _time_left() < 5.0:
            break
        rng = np.random.RandomState(seed)
        candidates = list(range(N))
        rng.shuffle(candidates)
        S = _greedy_sidon(candidates)
        if len(S) > len(best_set):
            best_set = S[:]
            if len(best_set) >= 8:  # Upper bound heuristic
                break
    
    # Verify
    if best_set and not _is_sidon_set(best_set):
        # Fallback to singleton
        return [0]
    
    return best_set

def construct():
    global _START
    _START = time.time()
    
    S = _find_max_sidon_set(N)
    
    if not S:
        h = np.zeros(N)
        h[0] = 1.0
        return list(h)
        
    k = len(S)
    h = np.zeros(N)
    for idx in S:
        h[idx] = 1.0 / k
        
    return list(h)
