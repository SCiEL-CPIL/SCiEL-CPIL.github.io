import time
from fractions import Fraction

def construct():
    # Known minimal Egyptian fraction representations for 1 with denominators >= 5
    # We want to minimize the number of terms (denominators)
    
    # 5 terms: 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140? Let's verify
    # Actually, let's use known results:
    # 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140? 
    # 1/5 + 1/6 + 1/7 + 1/42 + 1/140
    # = 84/420 + 70/420 + 60/420 + 10/420 + 3/420 = 227/420 != 1
    
    # Let me use a known 5-term representation:
    # 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140 is not correct.
    
    # Known minimal: 
    # 4 terms is impossible with distinct denominators >= 5?
    # Let's check: 1/5 + 1/6 + 1/7 + 1/8 = 0.2+0.1667+0.1429+0.125 = 0.6345 < 1
    # So we need at least 5 terms.
    
    # 5 terms: Try to find a known solution
    # 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140? Already checked, not 1.
    
    # Let me compute: 1 - 1/5 - 1/6 - 1/7 = 1 - 0.2 - 0.16667 - 0.14286 = 0.49048
    # 0.49048 = 1/x + 1/y where x < y
    # 1/x < 0.49048 => x > 2.039, so x >= 3, but we need x >= 5
    # 1/x <= 0.49048/2 = 0.24524 => x >= 4.077, so x >= 5
    # Try x=5: 0.49048 - 0.2 = 0.29048, 1/0.29048 = 3.44, not integer
    # Try x=6: 0.49048 - 0.16667 = 0.32381, 1/0.32381 = 3.09, not integer
    # Try x=7: 0.49048 - 0.14286 = 0.34762, 1/0.34762 = 2.88, not integer
    # Try x=8: 0.49048 - 0.125 = 0.36548, 1/0.36548 = 2.74, not integer
    # Try x=9: 0.49048 - 0.11111 = 0.37937, 1/0.37937 = 2.64, not integer
    # Try x=10: 0.49048 - 0.1 = 0.39048, 1/0.39048 = 2.56, not integer
    # Try x=11: 0.49048 - 0.09091 = 0.39957, 1/0.39957 = 2.50, close to 2.5 but not integer
    
    # Let me use exact fractions:
    # 1 - 1/5 - 1/6 - 1/7 = (210 - 42 - 35 - 30)/210 = 103/210
    # 103/210 = 1/x + 1/y = (x+y)/(xy)
    # 103*xy = 210*(x+y)
    # 103*xy - 210*x - 210*y = 0
    # (103x - 210)(103y - 210) = 210^2 + 103^2 = 44100 + 10609 = 54709
    # 54709 is prime? sqrt(54709) ≈ 233.9
    # Check if 54709 has factors... this is getting complex.
    
    # Let me just use a known valid 5-term solution or fall back to search.
    # Actually, let's use a simpler approach: try known small solutions.
    
    # A known 5-term Egyptian fraction for 1:
    # 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140? No.
    
    # Let me try: 1/5 + 1/6 + 1/7 + 1/42 + 1/140
    # LCM(5,6,7,42,140) = 420
    # 84 + 70 + 60 + 10 + 3 = 227, not 420.
    
    # Try: 1/5 + 1/6 + 1/7 + 1/42 + 1/20?
    # 84 + 70 + 60 + 10 + 21 = 245, not 420.
    
    # Let me just use the search but with a very tight deadline and known fallbacks.
    
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Known valid solutions (verified):
    # 5 terms: [5, 6, 7, 42, 140] - need to verify
    # Let's verify: 1/5 + 1/6 + 1/7 + 1/42 + 1/140
    # = 84/420 + 70/420 + 60/420 + 10/420 + 3/420 = 227/420 ≠ 1
    
    # Try another: [5, 6, 7, 42, 210]?
    # 84/420 + 70/420 + 60/420 + 10/420 + 2/420 = 226/420 ≠ 1
    
    # Let me use a different approach. Known result:
    # 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140 is not correct.
    
    # Actually, let me just use the DFS search with better pruning and a short deadline.
    # If it doesn't find quickly, use a known 6-term or 7-term solution.
    
    # Known 6-term: 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + 1/??? 
    # This is getting complicated. Let me just use the search with a 10-second budget.
    
    def dfs_exact(rem_sum, min_d, remaining_terms, current_list, deadline):
        if time.time() > deadline:
            return None
            
        if remaining_terms == 0:
            if rem_sum == 0:
                return current_list
            return None
            
        if remaining_terms == 1:
            if rem_sum > 0:
                d = 1 / rem_sum
                if d >= min_d and d == int(d) and int(d) >= 5 and int(d) <= 10000000:
                    return current_list + [int(d)]
            return None

        # Pruning: max possible sum
        max_possible_sum = Fraction(0, 1)
        for i in range(remaining_terms):
            max_possible_sum += Fraction(1, min_d + i)
            
        if max_possible_sum < rem_sum:
            return None

        inv_rem = 1 / rem_sum
        d_min = max(min_d, int(inv_rem) + (1 if inv_rem != int(inv_rem) else 0))
        
        for d in range(d_min, min(10000001, min_d + 10000)):  # Limit search depth
            if time.time() > deadline:
                return None
                
            if Fraction(1, d) > rem_sum:
                break
                
            if remaining_terms > 1:
                rem_new = rem_sum - Fraction(1, d)
                max_next_sum = Fraction(0, 1)
                for i in range(1, remaining_terms):
                    max_next_sum += Fraction(1, d + i)
                    
                if rem_new > max_next_sum:
                    break
                    
            res = dfs_exact(rem_new, d + 1, remaining_terms - 1, current_list + [d], deadline)
            if res is not None:
                return res
                
        return None

    # Try 5 terms first with a 15-second budget
    deadline_5 = start_time + 15
    res = dfs_exact(Fraction(1, 1), 5, 5, [], deadline_5)
    if res is not None:
        return res
    
    # Try 6 terms with remaining time
    deadline_6 = start_time + 35
    res = dfs_exact(Fraction(1, 1), 5, 6, [], deadline_6)
    if res is not None:
        return res
    
    # Try 7 terms
    deadline_7 = start_time + 48
    res = dfs_exact(Fraction(1, 1), 5, 7, [], deadline_7)
    if res is not None:
        return res
    
    # Fallback: use a known valid solution
    # 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + 1/??? 
    # Let me just return a known valid 7-term solution if search fails.
    
    # Actually, let me verify a simple one:
    # 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + 1/???
    # Sum so far: 227/420, need 193/420
    # 193/420 = 1/x + 1/y + 1/z? This is getting too complex.
    
    # Let me just use the subset DP fallback which is guaranteed to work.
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
            
    raise RuntimeError("no solution found")
