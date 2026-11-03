import math
from itertools import combinations

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    # Search over base-B binary expansion subsets
    # For base B and n digits, generate all 2^n binary numbers,
    # then try subsets of various sizes
    
    for B in range(3, 10):
        for n in range(2, 6):
            # Generate all base-B binary numbers (digits 0 or 1)
            all_vals = []
            for mask in range(1, 1 << n):  # exclude 0, we'll add it separately
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                all_vals.append(val)
            
            # Try subsets of size 2 to 5 (plus 0)
            for size in range(2, min(6, len(all_vals) + 1)):
                for combo in combinations(all_vals, size):
                    U = [0] + list(combo)
                    U = list(set(U))
                    if len(U) != size + 1:
                        continue
                    s = calc_score(U)
                    if s > best_score:
                        best_score = s
                        best = U
    
    return best
