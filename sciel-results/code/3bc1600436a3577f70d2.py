import math
import random
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
    
    # Search over base-B binary expansion full sets and subsets
    # U consists of all numbers with digits 0 or 1 in base B, up to n digits.
    
    for B in range(2, 21):
        for n in range(2, 11):
            # Generate all base-B binary numbers
            all_vals = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                all_vals.append(val)
            
            U_full = all_vals
            # Ensure 0 is present (it is, mask=0)
            s_full = calc_score(U_full)
            if s_full > best_score:
                best_score = s_full
                best = U_full
            
            # If the full set is small, do exhaustive search
            # If large, do greedy/heuristic search for sparse subsets
            max_subset_size = 12
            
            if len(all_vals) <= max_subset_size:
                # Exhaustive search for small sets
                for size in range(2, len(all_vals) + 1):
                    for combo in combinations(all_vals, size):
                        U_sub = [0] + list(combo)
                        U_sub = list(set(U_sub))
                        if len(U_sub) != size + 1:
                            continue
                        s_sub = calc_score(U_sub)
                        if s_sub > best_score:
                            best_score = s_sub
                            best = U_sub
            else:
                # Heuristic search for larger sets
                # Try greedy addition
                U_greedy = [0]
                current_score = calc_score(U_greedy)
                
                # To make it deterministic, we can try a few random restarts
                # or just one greedy pass from 0
                # Since we want to beat 1.0281, and sparse sets are good,
                # we try to build a set of size ~5-10
                
                # We'll do a simple local search:
                # Start with random subset of size 5
                # Try adding/removing elements to improve score
                
                for trial in range(5):
                    # Random start
                    U_curr = [0]
                    # Pick random elements from all_vals
                    candidates = [x for x in all_vals if x != 0]
                    random.shuffle(candidates)
                    for x in candidates[:4]:
                        U_curr.append(x)
                    U_curr = list(set(U_curr))
                    
                    improved = True
                    while improved:
                        improved = False
                        # Try adding
                        for x in candidates:
                            if x not in U_curr:
                                U_new = U_curr + [x]
                                s_new = calc_score(U_new)
                                if s_new > current_score:
                                    current_score = s_new
                                    U_curr = U_new
                                    improved = True
                        # Try removing (if size > 2)
                        if len(U_curr) > 2:
                            for x in U_curr:
                                if x == 0: continue
                                U_new = [y for y in U_curr if y != x]
                                s_new = calc_score(U_new)
                                if s_new > current_score:
                                    current_score = s_new
                                    U_curr = U_new
                                    improved = True
                                    
                    if current_score > best_score:
                        best_score = current_score
                        best = U_curr

    return best
