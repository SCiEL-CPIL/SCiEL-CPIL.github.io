import itertools
import math

def construct():
    best_score = 0
    best_set = [0, 1, 3, 7]
    
    def calc_score(u):
        if not u or len(u) < 2:
            return 0
        max_u = u[-1]
        if max_u == 0:
            return 0
            
        sum_mask = 0
        diff_mask = 0
        
        # Use integers as bitsets
        for i in u:
            for j in u:
                sum_mask |= (1 << (i + j))
                diff_mask |= (1 << (i - j + max_u))
                
        num_sums = bin(sum_mask).count('1')
        num_diffs = bin(diff_mask).count('1')
        
        if num_sums == 0:
            return 0
            
        ratio = num_diffs / num_sums
        if ratio <= 1:
            return 0
            
        denom = math.log2(2 * max_u + 1)
        if denom == 0:
            return 0
            
        return 1 + math.log2(ratio) / denom

    # Search small M to find high ratio sets with low penalty
    # M up to 20 is safe for Python loops with bitmask
    for M in range(1, 21):
        # Iterate over all subsets of {1, ..., M}
        # Represent subset as bitmask of length M
        # 0 means not in set, 1 means in set
        # We must include 0 in U, so we just pick from 1..M
        for mask in range(1, 1 << M):
            # Count bits to estimate size
            # If size is too large, sums become dense, ratio drops
            # But let's just check all
            u = [0]
            for i in range(M):
                if mask & (1 << i):
                    u.append(i + 1)
                    
            s = calc_score(u)
            if s > best_score:
                best_score = s
                best_set = u

    return best_set
