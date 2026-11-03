import math

def construct():
    best_score = 0
    best_set = [0]
    
    def calc_score(u):
        if not u:
            return 0
        max_u = max(u)
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

    # Greedy removal search
    # Try various starting sizes M
    for M in range(5, 101):
        # Start with full set 0..M
        u = list(range(M + 1))
        current_score = calc_score(u)
        
        # Greedy loop
        improved = True
        while improved:
            improved = False
            best_remove = None
            best_new_score = current_score
            best_new_u = u
            
            # Try removing each element except 0
            for x in u:
                if x == 0:
                    continue
                new_u = [y for y in u if y != x]
                new_score = calc_score(new_u)
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_remove = x
                    best_new_u = new_u
            
            if best_remove is not None:
                u = best_new_u
                current_score = best_new_score
                improved = True
        
        if current_score > best_score:
            best_score = current_score
            best_set = u
            
    return best_set
