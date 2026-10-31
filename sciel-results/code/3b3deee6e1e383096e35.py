import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 5.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

def _score(U):
    """
    Calculate score using FFT.
    U: list of non-negative integers.
    """
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    
    # Determine FFT size
    L = 1
    while L < 2 * M + 1:
        L *= 2
        
    # Create indicator vector
    A = np.zeros(L, dtype=np.float64)
    for u in U:
        A[u] = 1.0
        
    F_A = np.fft.fft(A)
    
    # Sumset
    F_S = F_A * F_A
    S = np.fft.ifft(F_S).real
    sum_count = np.sum(S[:2*M+1] > 0.5)
    
    # Diffset
    # Autocorrelation via convolution with reversed A
    A_rev = A[::-1]
    F_A_rev = np.fft.fft(A_rev)
    F_D = F_A * F_A_rev
    D = np.fft.ifft(F_D).real
    
    # Peak is at L-1. Range of interest: L-1-M to L-1+M
    # Since L >= 2M+1, this range is within [0, L-1] without wrapping
    # for the central part.
    start_idx = L - 1 - M
    end_idx = L - 1 + M + 1
    diff_count = np.sum(D[start_idx:end_idx] > 0.5)
    
    if sum_count == 0 or diff_count == 0:
        return 0.0
        
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
        
    ratio = diff_count / sum_count
    if ratio <= 1.0:
        return 1.0 # Log is negative or zero, score <= 1. But we want to maximize.
                   # Actually, if ratio < 1, log is negative, score < 1.
                   # We should return the actual value.
    
    return 1.0 + math.log(ratio, 2) / denom

def _gen_golomb_optimal(n):
    """Return known optimal Golomb rulers for small n."""
    optimal = {
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 25],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 44],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 55],
        11: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 67],
        12: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 80],
        13: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 72, 85],
        14: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 72, 85, 102],
        15: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 72, 85, 102, 121],
    }
    if n in optimal:
        return list(optimal[n])
    return None

def _local_search(U, time_budget=10.0):
    """
    Local search: try moving each element by small integers.
    """
    U = sorted(list(U))
    if 0 not in U:
        U.insert(0, 0)
    best_U = list(U)
    best_s = _score(best_U)
    start = time.time()
    rng = random.Random(42)
    n = len(best_U)
    
    while time.time() - start < time_budget and _remaining() > 2.0:
        improved = False
        indices = list(range(1, n))
        rng.shuffle(indices)
        
        for i in indices:
            if _remaining() < 2.0:
                break
            orig = best_U[i]
            # Try small shifts
            for delta in [-3, -2, -1, 1, 2, 3, -5, 5, -10, 10, -20, 20]:
                new_val = orig + delta
                if new_val < 0:
                    continue
                if new_val in best_U:
                    continue
                new_U = list(best_U)
                new_U[i] = new_val
                new_U = sorted(new_U)
                
                s = _score(new_U)
                if s > best_s + 1e-10:
                    best_s = s
                    best_U = new_U
                    improved = True
                    break
        if not improved:
            # Random perturbation if stuck
            if rng.random() < 0.1:
                idx = rng.randint(1, n-1)
                new_val = max(1, best_U[idx] + rng.randint(-20, 20))
                if new_val not in best_U:
                    new_U = list(best_U)
                    new_U[idx] = new_val
                    new_U = sorted(new_U)
                    s = _score(new_U)
                    if s > best_s:
                        best_s = s
                        best_U = new_U

    return best_U, best_s

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)

    candidates = []
    
    # 1. Known optimal Golomb rulers
    for n in range(3, 16):
        g = _gen_golomb_optimal(n)
        if g:
            candidates.append(g)
            
    # 2. Add some random sets
    for _ in range(20):
        n = random.randint(3, 15)
        M = random.randint(10, 200)
        U = [0] + random.sample(range(1, M+1), n-1)
        candidates.append(sorted(U))

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique.append(list(c))

    # Score all initial candidates
    scored = [(_score(s), s) for s in unique]
    scored.sort(key=lambda x: -x[0])

    if not scored:
        return [0, 1]

    best_U = scored[0][1]
    best_s = scored[0][0]

    # Local search on top candidates
    top_candidates = [s for _, s in scored[:10]]
    
    per_set_time = 5.0
    
    for s in top_candidates:
        if _remaining() < 5.0:
            break
        result, rs = _local_search(s, time_budget=per_set_time)
        if rs > best_s:
            best_s = rs
            best_U = result

    # Final polish
    if _remaining() > 10.0:
        polished, ps = _local_search(best_U, time_budget=min(15.0, _remaining() - 2.0))
        if ps > best_s:
            best_s = ps
            best_U = polished

    # Ensure validity
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
        
    return best_U
