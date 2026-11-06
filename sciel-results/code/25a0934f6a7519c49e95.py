import time
from itertools import combinations

def _check_valid(S):
    """Check if S is reciprocal-subsum-free."""
    S_set = set(S)
    for x in S:
        others = [y for y in S if y != x]
        target = 1.0 / x
        # Check subsets of others of size 1 to min(len(others), 5)
        # For large x, even size 1 is too big, so we can limit
        max_subset_size = min(len(others), 6)
        for r in range(1, max_subset_size + 1):
            for subset in combinations(others, r):
                s = sum(1.0 / y for y in subset)
                if abs(s - target) < 1e-9:
                    return False
                # Early termination: if sum already exceeds target, skip larger subsets
                # But combinations are sorted, so we can't easily break
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    best = list(range(13, 25))  # fallback: the seed
    if _check_valid(best):
        pass  # seed is valid
    
    # Try to find larger valid sets
    # Strategy 1: Greedy from large to small
    all_elements = list(range(1, 25))
    
    # Try multiple greedy orderings
    orderings = [
        list(range(24, 0, -1)),  # large to small
        list(range(1, 25)),      # small to large
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        current = []
        for x in ordering:
            if time.time() > deadline:
                break
            candidate = current + [x]
            if _check_valid(candidate):
                current = candidate
        if len(current) > len(best):
            best = current
    
    # Try to improve: start from best, try adding elements not in best
    if time.time() < deadline - 10:
        improved = list(best)
        for x in range(1, 25):
            if time.time() > deadline:
                break
            if x not in improved:
                candidate = improved + [x]
                if _check_valid(candidate):
                    improved = candidate
        if len(improved) > len(best):
            best = improved
    
    # Try removing one element and adding two
    if time.time() < deadline - 15:
        for i in range(len(best)):
            if time.time() > deadline:
                break
            for j in range(1, 25):
                if time.time() > deadline:
                    break
                if j in best:
                    continue
                for k in range(j+1, 25):
                    if time.time() > deadline:
                        break
                    if k in best:
                        continue
                    candidate = [x for idx, x in enumerate(best) if idx != i] + [j, k]
                    if _check_valid(candidate):
                        if len(candidate) > len(best):
                            best = candidate
                        break
                break
    
    return best
