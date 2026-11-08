import time
from itertools import combinations, product

def compute_score(U):
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0
    max_val = max(U_set)
    sumset = set()
    diffset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
            diffset.add(a - b)
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    import math
    denom = math.log(2 * max_val + 1)
    if denom == 0:
        return 1.0
    return 1 + math.log(ratio) / denom

def construct():
    deadline = time.time() + 140  # safety margin
    
    # Strategy 1: Powers of 2
    # U = {0, 1, 2, 4, 8, ...}
    best_score = 0
    best_set = [0, 1, 3]
    
    # Test powers of 2
    for k in range(1, 50):
        if time.time() > deadline:
            break
        U = [0] + [2**i for i in range(k)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Test powers of 3
    for k in range(1, 30):
        if time.time() > deadline:
            break
        U = [0] + [3**i for i in range(k)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Test mixed: {0} union {a^i} for various a
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for k in range(1, 40):
            if time.time() > deadline:
                break
            U = [0] + [base**i for i in range(k)]
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Try structured sets from literature: {0, 1, 3, 9, 27, ...} already covered
    # Try {0, 1, 2, 6, 18, 54, ...} (mix of 2 and 3)
    
    # Try sets like {0, 1, 3, 10, 30, ...}
    for k in range(1, 30):
        if time.time() > deadline:
            break
        # Geometric with ratio ~3 but with some structure
        U = [0]
        val = 1
        for i in range(k):
            U.append(val)
            val *= 3
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Try to find good small sets by exhaustive search
    if time.time() < deadline - 30:
        # Exhaustive search for small sets
        # Try all subsets of {0,1,2,...,15} of size 3-8 containing 0
        candidates = list(range(0, 16))
        for size in range(3, 9):
            if time.time() > deadline - 10:
                break
            for subset in combinations(candidates, size):
                if 0 not in subset:
                    continue
                U = list(subset)
                s = compute_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U
    
    return best_set
