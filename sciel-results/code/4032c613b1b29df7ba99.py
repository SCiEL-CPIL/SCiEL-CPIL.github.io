import time
import math

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Theoretical best found: {0, 1, 3, 9}
    best_U = [0, 1, 3, 9]
    best_score = compute_score(best_U)
    
    def try_set(U):
        nonlocal best_U, best_score
        if time.time() > deadline:
            return False
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
        return time.time() <= deadline
    
    # Try a few other candidates just in case
    candidates = [
        [0, 1, 3, 9, 27],
        [0, 1, 3, 9, 10],
        [0, 1, 3, 9, 12],
        [0, 1, 3, 9, 18],
        [0, 1, 3, 9, 20],
        [0, 1, 3, 9, 30],
        [0, 1, 3, 9, 100],
        [0, 1, 3, 9, 1000],
        [0, 1, 3, 9, 3**4],
        [0, 1, 3, 9, 3**5],
        [0, 1, 3, 9, 3**6],
        [0, 1, 3, 9, 3**7],
    ]
    
    for U in candidates:
        if not try_set(U):
            break
    
    # Also try small variations of the base set
    base = [0, 1, 3, 9]
    for add_val in range(10, 50):
        if not try_set(base + [add_val]):
            break
    
    # Exhaustive search over 4-element sets {0, a, b, c} with 0 < a < b < c <= 40
    for a in range(1, 20):
        if time.time() > deadline:
            break
        for b in range(a + 1, 30):
            if time.time() > deadline:
                break
            for c in range(b + 1, 41):
                if not try_set([0, a, b, c]):
                    break
    
    # Exhaustive search over 5-element sets {0, a, b, c, d} with 0 < a < b < c < d <= 25
    for a in range(1, 8):
        if time.time() > deadline:
            break
        for b in range(a + 1, 12):
            if time.time() > deadline:
                break
            for c in range(b + 1, 18):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 26):
                    if not try_set([0, a, b, c, d]):
                        break
    
    # Try structured families: {0, a, a+b, a+b+c} (additive chains)
    for a in range(1, 15):
        if time.time() > deadline:
            break
        for b in range(1, 20):
            if time.time() > deadline:
                break
            for c in range(1, 25):
                if not try_set([0, a, a + b, a + b + c]):
                    break
    
    # Try {0, a, b, a*b} type multiplicative structures
    for a in range(1, 12):
        if time.time() > deadline:
            break
        for b in range(a + 1, 20):
            if time.time() > deadline:
                break
            prod = a * b
            if prod > 200:
                continue
            if not try_set([0, a, b, prod]):
                break
    
    # Try 3-element sets for comparison
    for a in range(1, 30):
        if time.time() > deadline:
            break
        for b in range(a + 1, 60):
            if not try_set([0, a, b]):
                break
    
    return best_U
