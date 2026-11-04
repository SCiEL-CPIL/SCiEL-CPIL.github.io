"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_set = [0, 1, 3]
    best_score = _score(best_set)
    
    # Try powers of 3 constructions
    for n in range(3, 30):
        if time.time() > deadline - 10:
            break
        U = [0] + [3**i for i in range(n)]
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Try mixed constructions: 0 plus a geometric progression
    for base in [2, 3, 4, 5]:
        for n in range(3, 25):
            if time.time() > deadline - 5:
                break
            U = [0] + [base**i for i in range(n)]
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Try adding 0 to arithmetic progressions with large gap
    for n in range(5, 50):
        if time.time() > deadline - 5:
            break
        for gap in [10, 100, 1000]:
            U = [0] + [i * gap for i in range(1, n)]
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    return best_set

def _score(U):
    """Compute the score for a given set U."""
    if not U or 0 not in U:
        return 0
    U = sorted(set(U))
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return 0
    
    # Compute U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    sum_size = len(sumset)
    diff_size = len(diffset)
    
    if sum_size == 0:
        return 0
    
    ratio = diff_size / sum_size
    if ratio <= 0:
        return 0
    
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return 0
    
    return 1 + math.log(ratio) / log_denom

# EVOLVE-BLOCK-END
