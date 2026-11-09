import itertools
import math
import time
import numpy as np

def _score_fast(U):
    """Fast score computation using numpy for vectorized set operations."""
    U = np.array(sorted(set(U)))
    if 0 not in U:
        return 0.0
    mx = int(U[-1])
    if mx == 0:
        return 0.0
    
    # Compute all sums and differences using broadcasting
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    n_plus = len(np.unique(sums))
    n_minus = len(np.unique(diffs))
    
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def _score(U):
    """Original score function for compatibility."""
    U = list(set(U))
    if 0 not in U:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    plus = set()
    minus = set()
    for a in U:
        for b in U:
            plus.add(a + b)
            minus.add(a - b)
    n_plus = len(plus)
    n_minus = len(minus)
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Phase 1: Exhaustive search for small sizes with expanded range
    max_val_small = 40
    candidates_small = list(range(1, max_val_small + 1))
    
    for size in range(2, 7):
        for combo in itertools.combinations(candidates_small, size - 1):
            if time.time() > deadline:
                return best_set
            U = [0] + list(combo)
            s = _score_fast(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Phase 2: Random search for larger sizes if time remains
    if time.time() < deadline - 5.0:
        rng = np.random.RandomState(42)
        max_val_large = 200
        max_size = 12
        
        # Try various size ranges
        for size in range(6, max_size + 1):
            if time.time() > deadline:
                break
            
            num_trials = 500
            for trial in range(num_trials):
                if time.time() > deadline:
                    break
                
                # Random subset of size-1 elements from 1..max_val_large
                combo = rng.choice(range(1, max_val_large + 1), size - 1, replace=False)
                U = [0] + sorted(combo.tolist())
                s = _score_fast(U)
                if s > best_score:
                    best_score = s
                    best_set = U
    
    return best_set
