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
    n = len(U)
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
    
    # Search over base-B binary expansion full sets
    # U consists of all numbers with digits 0 or 1 in base B, up to n digits.
    
    # First: try full binary-base sets with wider range
    for B in range(2, 30):
        for n in range(2, 12):
            # Generate all base-B binary numbers
            all_vals = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                all_vals.append(val)
            
            U = all_vals
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best = U
    
    # Second: try subsets of promising base-B sets
    # Focus on bases that gave good scores, and try removing elements
    for B in range(3, 20):
        for n in range(3, 8):
            all_vals = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                all_vals.append(val)
            
            # Try subsets of size 4-10 from the full set
            if len(all_vals) > 20:
                continue
            for size in range(3, min(8, len(all_vals) + 1)):
                for combo in combinations(all_vals, size):
                    U_sub = [0] + list(combo)
                    U_sub = list(set(U_sub))
                    if len(U_sub) != size + 1:
                        continue
                    s_sub = calc_score(U_sub)
                    if s_sub > best_score:
                        best_score = s_sub
                        best = U_sub
    
    # Third: try custom digit patterns - not just 0/1 but specific subsets
    # For each base, try all subsets of {0,1,...,k} as allowed digits
    for B in range(2, 15):
        for n in range(2, 6):
            # Try different digit sets
            digit_options = [
                [0, 1],
                [0, 1, 2],
                [0, 2],
                [0, 1, 3],
                [0, 2, 4],
                [0, 1, 2, 3],
            ]
            for digits in digit_options:
                # Generate all numbers using only these digits
                all_vals = [0]
                for mask in range(1, 1 << n):
                    val = 0
                    valid = True
                    for j in range(n):
                        if mask & (1 << j):
                            # Use the j-th digit from our set
                            if j < len(digits):
                                val += digits[j] * (B ** j)
                            else:
                                valid = False
                                break
                    if valid:
                        all_vals.append(val)
                
                U = list(set(all_vals))
                if len(U) < 2:
                    continue
                s = calc_score(U)
                if s > best_score:
                    best_score = s
                    best = U
    
    # Fourth: try geometric progressions with different ratios
    for ratio in range(2, 20):
        for n in range(3, 10):
            U = [ratio ** i for i in range(n)]
            U = list(set(U))
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best = U
    
    # Fifth: try sparse sets - pick elements from large base expansions
    for B in range(5, 15):
        for n in range(4, 7):
            # Generate candidates
            candidates = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                candidates.append(val)
            
            # Try subsets of size 5-12
            if len(candidates) > 16:
                continue
            for size in range(4, min(10, len(candidates) + 1)):
                for combo in combinations(candidates, size):
                    U_sub = [0] + list(combo)
                    U_sub = list(set(U_sub))
                    if len(U_sub) != size + 1:
                        continue
                    s_sub = calc_score(U_sub)
                    if s_sub > best_score:
                        best_score = s_sub
                        best = U_sub
    
    return best
