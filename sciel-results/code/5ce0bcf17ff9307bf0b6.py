import math

def construct():
    best_score = 0
    best_set = [0, 1, 3, 7, 15]
    
    def calc_score(u):
        if not u:
            return 0
        max_u = u[-1]
        sum_mask = 0
        diff_mask = 0
        
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

    # Parametric search: U = {0, d, 2d, ..., (k-2)d, M}
    # k is size.
    # We limit M to keep bitmasks small and fast.
    # Max M ~ 1000? 2000 bits is fast.
    
    for k in range(3, 15):
        for d in range(1, 50):
            # M must be larger than (k-2)*d to be distinct and "large gap"
            # M range: (k-1)*d to (k-1)*d + 100
            base_last = (k - 2) * d
            for M in range(base_last + 1, base_last + 101):
                u = [0]
                for i in range(1, k - 1):
                    u.append(i * d)
                u.append(M)
                u.sort()
                
                # Check uniqueness
                if len(set(u)) != len(u):
                    continue
                    
                s = calc_score(u)
                if s > best_score:
                    best_score = s
                    best_set = u

    return best_set
