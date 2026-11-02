import itertools
import math

def construct():
    # Bitmask approach for fast sum/diff calculation
    # Max element M. Max sum 2M. Max diff M.
    # We search for sizes 5 to 8, elements 0..30
    
    best_score = 0
    best_set = [0, 1, 3, 7, 15]
    
    # To allow larger sets, we limit M
    # n=8, M=20 -> C(20,7) = 77520. Very fast.
    # n=9, M=20 -> C(20,8) = 125970.
    # n=10, M=20 -> C(20,9) = 167960.
    # Let's try n up to 10, M up to 30.
    # C(30, 9) is 14M. Too slow in Python.
    # C(20, 9) is 167k. Fast.
    # C(25, 9) is 2M. Maybe too slow.
    
    # Strategy:
    # 1. Search small M (0..20) for larger n (up to 12)
    # 2. Search larger M (0..40) for smaller n (up to 8)
    
    def calc_score(u):
        if not u:
            return 0
        max_u = u[-1]
        # Bitmask for sums and diffs
        # Size needs to be 2*max_u + 1 for sums (0..2max_u)
        # Size needs to be 2*max_u + 1 for diffs (-max_u..max_u) -> shift by max_u
        
        sum_mask = 0
        diff_mask = 0
        
        # Use integers as bitsets
        # i+j
        for i in u:
            for j in u:
                sum_mask |= (1 << (i + j))
                
        # i-j + max_u
        for i in u:
            for j in u:
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

    # Search 1: Small M, larger n
    for M in [10, 15, 20, 25]:
        for n in range(4, 11):
            # Combinations of n-1 from 1..M
            # C(M, n-1)
            if n-1 > M:
                continue
            # Skip if too many
            # Approx C(25, 10) = 3.2M. Might be slow.
            # Let's limit C(M, n-1) < 100,000 for safety in Python
            # C(20, 10) = 184k.
            # C(25, 9) = 2M.
            
            # We will iterate and break if it takes too long? No, just limit.
            # Let's just do M=20, n up to 10.
            
            if M > 20 and n > 8:
                continue
            if M > 25 and n > 7:
                continue

            for comb in itertools.combinations(range(1, M + 1), n - 1):
                u = [0] + list(comb)
                u.sort()
                s = calc_score(u)
                if s > best_score:
                    best_score = s
                    best_set = u

    return best_set
