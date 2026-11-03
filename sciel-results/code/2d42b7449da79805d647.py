import math

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
    best_U = [0]
    best_score = 0.0
    
    # Search parameters
    bases = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 50]
    max_depth = 8
    max_size = 25
    
    for B in bases:
        # Generate candidates: all numbers sum b_j B^j with b_j in {0,1}
        cand_list = []
        for mask in range(1, 1 << max_depth):
            val = 0
            for j in range(max_depth):
                if mask & (1 << j):
                    val += B ** j
            cand_list.append(val)
        
        # Greedy Construction
        current_U = [0]
        sums = {0}
        diffs = {0}
        max_u = 0
        
        for _ in range(max_size):
            best_add = None
            best_add_score = 1 + math.log(len(diffs) / len(sums)) / math.log(2 * max_u + 1) if max_u > 0 else 0
            best_new_sums = None
            best_new_diffs = None
            best_new_max_u = max_u
            
            for c in cand_list:
                if c in current_U:
                    continue
                
                new_max_u = max(max_u, c)
                
                new_sums = set(sums)
                new_diffs = set(diffs)
                
                for u in current_U:
                    new_sums.add(c + u)
                    new_diffs.add(c - u)
                    new_diffs.add(u - c)
                
                num = len(new_diffs) / len(new_sums)
                den = math.log(2 * new_max_u + 1)
                if den == 0:
                    s = 0
                else:
                    s = 1 + math.log(num) / den
                
                if s > best_add_score:
                    best_add_score = s
                    best_add = c
                    best_new_sums = new_sums
                    best_new_diffs = new_diffs
                    best_new_max_u = new_max_u
            
            if best_add is None:
                break
            
            current_U.append(best_add)
            sums = best_new_sums
            diffs = best_new_diffs
            max_u = best_new_max_u
            
            if best_add_score > best_score:
                best_score = best_add_score
                best_U = list(current_U)
                
    return best_U
