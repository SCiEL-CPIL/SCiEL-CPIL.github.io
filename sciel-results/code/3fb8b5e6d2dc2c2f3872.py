import time
import math

def construct():
    # Known best solution
    best_solution = [8, 31, 248]
    best_max = 248
    
    target_num = 5
    target_den = 31
    
    # Search for 3-term solutions with max < best_max
    # 1/a + 1/b + 1/c = 5/31
    # a < b < c < best_max
    # a >= ceil(31/5) = 7
    
    for a in range(7, best_max):
        # 1/b + 1/c = 5/31 - 1/a = (5a - 31) / (31a)
        N = 5 * a - 31
        D = 31 * a
        if N <= 0:
            continue
        
        # 1/b + 1/c = N/D
        # (Nb - D)(Nc - D) = D^2
        # X = Nb - D, Y = Nc - D
        # X < Y (since b < c)
        # X * Y = D^2
        # X must be a divisor of D^2
        # X < D (since b < c implies X < D? 
        #   b < c => 1/b > 1/c => N/D - 1/c > N/D - 1/b => 1/b > ... 
        #   Actually X = Nb - D. If b < D/N, X < 0. But b > D/N for 1/b < N/D.
        #   We need 1/b < N/D => b > D/N.
        #   Also 1/b + 1/c = N/D => 1/b < N/D => b > D/N.
        #   And 1/c < N/D => c > D/N.
        #   b < c => X < Y.
        #   Also X + Y = N(b+c).
        #   XY = D^2.
        #   Since b < c, X < Y, so X < D.
        
        D_sq = D * D
        # Iterate divisors X of D^2 such that X < D
        # D is at most 31 * 247 = 7657. D^2 ~ 5.8 * 10^7.
        # Finding divisors is fast enough for small D.
        
        # Optimization: 
        # b must be > D/N
        # b must be < c < best_max
        # So b < best_max
        # Also c = (D+Y)/N < best_max => D+Y < N*best_max => Y < N*best_max - D
        
        # Let's just iterate b from max(a+1, ceil(D/N)+1) to best_max-1
        # and check if c is integer.
        
        b_min = max(a + 1, (D + N) // N + 1) # b > D/N
        # Also c > b, so 1/c < 1/b.
        # 1/c = N/D - 1/b = (Nb - D) / (Db)
        # c = Db / (Nb - D)
        # We need Nb - D to divide Db.
        # And c > b => Db / (Nb - D) > b => D > Nb - D => 2D > Nb => b < 2D/N.
        
        b_max = min(best_max - 1, (2 * D) // N)
        
        if b_min > b_max:
            continue
            
        for b in range(b_min, b_max + 1):
            num_c = D * b
            den_c = N * b - D
            if den_c <= 0:
                continue
            if num_c % den_c == 0:
                c = num_c // den_c
                if c > b and c < best_max:
                    # Found a better solution
                    best_solution = [a, b, c]
                    best_max = c
                    # Can break early if we find very small max?
                    # But we want the absolute min.
                    # Since we iterate a, b, c, we might find smaller c later.
                    # However, smaller c implies smaller max.
                    # We are searching for max < current best_max.
                    # So any found solution improves best_max.
                    # We can update best_max and continue to see if we can go lower.
                    pass
    
    # Search for 4-term solutions with max < best_max
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # a < b < c < d < best_max
    
    # This is O(N^3) or O(N^2) with checks.
    # N is small (<= 247).
    
    # We can optimize by iterating a, b and solving for c, d.
    # 1/c + 1/d = 5/31 - 1/a - 1/b
    
    # To keep it fast, we only do this if best_max is still relatively large.
    # But 247^3 is ~ 15 million. In Python, this might take ~1-5 seconds.
    # Acceptable within 60s budget.
    
    # However, we can prune heavily.
    # 1/a < 5/31 => a >= 7.
    # 1/a + 1/b < 5/31 => b > ...
    
    # Let's implement a simple DFS for 4 terms with limit best_max.
    
    def dfs(rem_num, rem_den, min_d, current, terms_left, limit):
        if time.time() > 55.0: # Safety
            return None
        
        if rem_num == 0:
            if terms_left == 0 and current:
                return current[:]
            return None
        
        if terms_left == 0:
            return None
            
        if terms_left == 1:
            if rem_num == 1:
                d = rem_den
                if min_d <= d < limit:
                    current.append(d)
                    res = current[:]
                    current.pop()
                    return res
            return None
            
        # Pruning
        # Max sum of remaining terms: sum(1/(min_d+i))
        # Min sum: terms_left / (limit-1)
        
        # d_start
        d_start = max(min_d, (rem_den + rem_num - 1) // rem_num)
        
        # d_bound
        # 1/d >= rem_num/rem_den / terms_left ? No.
        # 1/d < rem_num/rem_den
        # d > rem_den/rem_num
        
        # Also rem_num/rem_den - 1/d must be formable by terms_left-1 terms.
        # Min sum of next terms: (terms_left-1) / (limit-1)
        # Max sum of next terms: sum_{i=1}^{terms_left-1} 1/(d+i)
        
        # Iterate d
        # Upper bound for d:
        # rem_num/rem_den - 1/d <= sum_{i=1}^{terms_left-1} 1/(d+i)
        # This is hard to invert.
        # But d < limit.
        
        for d in range(d_start, limit):
            if time.time() > 55.0:
                return None
                
            num_rem = rem_num * d - rem_den
            if num_rem <= 0:
                continue
                
            den_rem = rem_den * d
            
            # Check if num_rem/den_rem can be formed by terms_left-1 terms in (d, limit)
            # Min possible sum: (terms_left-1) / (limit-1)
            if num_rem * (limit - 1) < (terms_left - 1) * den_rem:
                continue
                
            # Max possible sum: sum 1/(d+i)
            # Approx (terms_left-1)/d
            if num_rem * d > (terms_left - 1) * den_rem:
                break # d is too large, 1/d too small, rem too large?
                # Wait. If d is large, 1/d is small. rem = target - 1/d is large.
                # If rem is too large to be formed by small terms (which are > d, so small), then fail.
                # Actually, if d is large, next terms are > d, so they are small.
                # Max sum of next terms is small.
                # So if rem > max_sum, fail.
                # rem = num_rem/den_rem.
                # max_sum ~ (terms_left-1)/d.
                # num_rem/den_rem > (terms_left-1)/d
                # num_rem * d > (terms_left-1) * den_rem
                # This condition means rem is TOO BIG.
                # So we should BREAK if d is too large?
                # Yes, as d increases, 1/d decreases, rem increases.
                # So if rem is too big for current d, it will be even bigger for larger d.
                # So break.
                
            current.append(d)
            res = dfs(num_rem, den_rem, d + 1, current, terms_left - 1, limit)
            if res:
                return res
            current.pop()
            
        return None

    # Run 4-term search
    # Target 5/31
    # min_d = 7
    # limit = best_max
    res = dfs(5, 31, 7, [], 4, best_max)
    if res:
        best_solution = res
        best_max = max(res)

    return best_solution

if __name__ == "__main__":
    print(construct())
