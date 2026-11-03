import itertools
import math

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return 0
    U_plus = set(a+b for a in U for b in U)
    U_minus = set(a-b for a in U for b in U)
    max_u = max(U)
    if max_u == 0:
        return 1.0
    ratio = len(U_minus) / len(U_plus)
    return 1 + math.log(ratio) / math.log(2 * max_u + 1)

def construct():
    best_set = [0, 1]
    best_score = score_set(best_set)
    
    # Search small sets
    max_val = 20
    for size in range(2, 6):
        # Generate combinations
        # To save time, only consider sets with 0 and 1
        # and increasing order
        for comb in itertools.combinations(range(2, max_val+1), size-2):
            U = [0, 1] + list(comb)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
                
    return best_set
