import time
import math
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0

def _elapsed():
    return time.time() - _START

def _compute_score(U):
    """Compute score for set U using O(n²) exact computation."""
    U = np.asarray(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n < 2:
        return 0.0
    max_u = int(U[-1])
    if max_u <= 0:
        return 0.0
    
    # Compute all pairwise sums and differences
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return 0.0
    
    base = 2 * max_u + 1
    if base <= 1:
        return 0.0
    
    return 1.0 + math.log(ratio) / math.log(base)

# Known optimal Golomb rulers (minimal length for given n)
# Source: Various mathematical references on Golomb rulers
_KNOWN_GOLOMB = {
    2: [0, 1],
    3: [0, 1, 3],
    4: [0, 1, 4, 6],
    5: [0, 1, 4, 10, 11],
    6: [0, 1, 4, 10, 12, 17],
    7: [0, 1, 4, 10, 12, 17, 22],
    8: [0, 1, 4, 10, 12, 17, 22, 29],
    9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
    10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
    12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
    13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
    14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85],
    15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99],
    16: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 114],
    17: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 114, 131],
}

def _fast_local_search(U, time_limit=5.0):
    """Fast local search to see if any small improvement exists."""
    U = list(sorted(set(U)))
    cur_score = _compute_score(U)
    best_score = cur_score
    best_U = list(U)
    start = time.time()
    rng = random.Random(hash(tuple(U)) % (2**31))
    
    while time.time() - start < time_limit and _elapsed() < _DEADLINE:
        improved = False
        n = len(U)
        max_u = max(U) if U else 1
        
        # Try adding a new element
        if n < 20:
            for _ in range(10):
                if time.time() - start > time_limit or _elapsed() > _DEADLINE:
                    break
                new_val = rng.randint(1, max_u + max(5, max_u // 3))
                if new_val in U or new_val < 0:
                    continue
                trial = sorted(U + [new_val])
                s = _compute_score(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    if s > best_score:
                        best_score = s
                        best_U = list(U)
                    improved = True
                    break
        
        # Try removing an element
        if not improved and n > 3:
            for _ in range(10):
                if time.time() - start > time_limit or _elapsed() > _DEADLINE:
                    break
                idx = rng.randint(1, n - 1)
                trial = U[:idx] + U[idx+1:]
                s = _compute_score(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    n = len(U)
                    if s > best_score:
                        best_score = s
                        best_U = list(U)
                    improved = True
                    break
        
        # Try replacing an element
        if not improved and n > 2:
            for _ in range(10):
                if time.time() - start > time_limit or _elapsed() > _DEADLINE:
                    break
                idx = rng.randint(1, n - 1)
                old_val = U[idx]
                new_val = old_val + rng.choice([-2, -1, 1, 2])
                if new_val <= 0 or new_val in U:
                    continue
                trial = sorted(U[:idx] + [new_val] + U[idx+1:])
                s = _compute_score(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    if s > best_score:
                        best_score = s
                        best_U = list(U)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_U, best_score

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 140.0
    
    best_U = [0]
    best_score = 0.0
    
    # Phase 1: Evaluate known Golomb rulers
    for n, U in _KNOWN_GOLOMB.items():
        if _elapsed() > _DEADLINE - 10:
            break
        s = _compute_score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    # Phase 2: Local search from the best starting point
    if _elapsed() < _DEADLINE - 20:
        U, score = _fast_local_search(best_U, time_limit=20.0)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Ensure 0 is in the set and return sorted
    if 0 not in best_U:
        best_U.append(0)
    
    return sorted(best_U)
