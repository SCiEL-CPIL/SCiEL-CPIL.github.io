import time
import math
from itertools import combinations

def _score(U):
    """Compute the score for a given set U (as a sorted list)."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0  # degenerate case
    
    # Compute U+U
    sum_set = set()
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
    
    # Compute U-U
    diff_set = set()
    for a in U_set:
        for b in U_set:
            diff_set.add(a - b)
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1:
        return 1.0
    
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
    
    return 1.0 + log_ratio / log_denom

def _search_small_sets(deadline):
    """Search over small sets of nonnegative integers containing 0."""
    best_score = 0.0
    best_set = [0]
    
    max_val = 20
    max_size = 6
    
    candidates = list(range(1, max_val + 1))
    
    for size in range(1, max_size + 1):
        if time.time() > deadline:
            break
        for combo in combinations(candidates, size):
            if time.time() > deadline:
                break
            U = [0] + list(combo)
            U.sort()
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    # First, get a good baseline
    best_set, best_score = _search_small_sets(deadline)
    
    # Try geometric progressions
    # Base B, size k+1
    for B in [2, 3, 4, 5]:
        for k in range(1, 10):
            if time.time() > deadline:
                break
            U = [B**i for i in range(k+1)]
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
                
    # Try local search around best found so far
    # If best is small, we can try adding elements
    # But _search_small_sets already did some of this.
    
    return best_set
