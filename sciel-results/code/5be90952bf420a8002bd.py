import math
from itertools import combinations

def score_set(U):
    U = sorted(U)
    if not U or U[0] != 0:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute |U+U|
    sum_set = set()
    for i in range(len(U)):
        for j in range(i, len(U)):
            sum_set.add(U[i] + U[j])
    
    # Compute |U-U|
    diff_set = set()
    for i in range(len(U)):
        for j in range(len(U)):
            diff_set.add(U[i] - U[j])
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0
        
    ratio = size_diff / size_sum
    
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def construct():
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    # Search for optimal D and B for k=1
    # U = D
    # Constraint: B > 2 * max(D) is not strictly required for k=1,
    # but it ensures that the "base expansion" logic holds for larger k if we wanted to use it.
    # However, for k=1, we just need to find a set D that maximizes the score.
    # The score depends on |D+D|, |D-D| and max(D).
    
    # We can just search all subsets of {0, ..., MAX_VAL}
    # But we can be smarter.
    # We know max(D) should not be too large.
    # Let's search max(D) up to 50.
    
    MAX_VAL = 50
    best_local_score = 0
    best_local_set = []
    
    for max_d in range(1, MAX_VAL + 1):
        # D must contain max_d? Not necessarily, but if max_d is not in D, we can shrink max.
        # So we can assume max(D) = max_d.
        # We pick other elements from 1 to max_d-1.
        # To limit combinations, we can restrict size of D.
        # Score improves with larger |D-D|/|D+D|.
        # Usually small D is better for small max.
        
        # Try sizes 2, 3, 4, 5
        for size in range(2, 6):
            # If size is large, combinations explode.
            # C(49, 4) is 211k. C(49, 5) is 1.9M.
            # 1.9M * 50 (max_d) = 95M. Too slow.
            
            # We need a heuristic.
            # Let's just test specific patterns or smaller max_d for larger sizes.
            
            if max_d > 20 and size > 3:
                continue
                
            # Generate combinations of size-1 elements from 1 to max_d-1
            # One element must be max_d? No, we fixed max_d as the max.
            # So we pick size-1 elements from 1 to max_d-1, and add max_d?
            # No, we pick size elements from 0 to max_d, one is 0, one is max_d?
            # If 0 is in D, and max_d is in D.
            # Pick size-2 elements from 1 to max_d-1.
            
            if size == 2:
                # D = {0, max_d}
                D = [0, max_d]
                s = score_set(D)
                if s > best_local_score:
                    best_local_score = s
                    best_local_set = D[:]
            else:
                # Pick size-2 from 1 to max_d-1
                # Limit search space
                if max_d > 15 and size > 4:
                    continue
                    
                for combo in combinations(range(1, max_d), size - 2):
                    D = [0] + list(combo) + [max_d]
                    D.sort()
                    s = score_set(D)
                    if s > best_local_score:
                        best_local_score = s
                        best_local_set = D[:]
                        
    if best_local_score > best_score:
        best_score = best_local_score
        best_set = best_local_set
        
    return best_set
