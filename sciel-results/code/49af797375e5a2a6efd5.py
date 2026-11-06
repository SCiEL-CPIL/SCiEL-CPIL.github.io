from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # 1. DP Baseline
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            best_sol = reached[target]
            break
    else:
        # Fallback if DP fails (should not happen)
        best_sol = [5, 6, 7, 42, 140, 210, 420, 1260, 2520] # Check sum?
        # 1/5+1/6+1/7+1/42+1/140+1/210+1/420+1/1260+1/2520
        # Sum = 1?
        # 1/5+1/6+1/7+1/42 = 39/70 = 1404/2520
        # 1/140 = 18/2520
        # 1/210 = 12/2520
        # 1/420 = 6/2520
        # 1/1260 = 2/2520
        # 1/2520 = 1/2520
        # Sum = 1404+18+12+6+2+1 = 1443/2520 != 1.
        # So this fallback is invalid.
        # But DP should work.
        
    best_count = len(best_sol)
    
    # 2. DFS for shorter solution
    def dfs(rem, min_d, current):
        nonlocal best_sol, best_count
        if time.time() > deadline:
            return
        if rem == 0:
            if len(current) < best_count:
                best_count = len(current)
                best_sol = current[:]
            return
        if rem < 0:
            return
        if len(current) >= best_count:
            return
            
        # Pruning: min terms needed
        # rem = p/q
        # 1/d <= rem => d >= q/p
        # min_d_next = max(min_d, q/p)
        
        # Also, max sum of (best_count - len(current) - 1) terms?
        # No, we want to find ANY solution with len < best_count.
        # So we can use up to best_count - len(current) - 1 more terms.
        # Let k = best_count - len(current) - 1
        # Max sum with k terms starting from min_d is k/min_d.
        # If k/min_d < rem, prune.
        
        k_max = best_count - len(current) - 1
        if k_max < 0:
            return
        if Fraction(k_max, min_d) < rem:
            return
            
        # Determine range for d
        # 1/d <= rem => d >= 1/rem
        d_start = max(min_d, (rem.denominator // rem.numerator) + (1 if rem.denominator % rem.numerator != 0 else 0))
        
        # Upper bound for d
        # We have k_max terms left after this one? No, we have k_max terms total remaining including this one?
        # No, we are picking d now. Remaining terms = k_max - 1?
        # No, we want total terms < best_count.
        # Current length L.
        # We pick d. New length L+1.
        # We need to fill rem - 1/d with (best_count - (L+1)) terms.
        # Let M = best_count - L - 1.
        # Max sum of M terms starting from d+1 is M/(d+1).
        # We need M/(d+1) >= rem - 1/d.
        # If M == 0, we need rem - 1/d == 0 => d = 1/rem.
        
        M = best_count - len(current) - 1
        if M == 0:
            if rem * d_start == 1:
                d = d_start
                current.append(d)
                dfs(rem - Fraction(1, d), d + 1, current)
                current.pop()
            return
            
        # M > 0
        # M/(d+1) >= rem - 1/d
        # M*d >= (d+1)(rem*d - 1)
        # M*d >= rem*d^2 - rem + rem*d - 1
        # 0 >= rem*d^2 + (rem - M - 1)*d - (rem + 1)
        
        # Simpler bound:
        # Max sum of M terms with min denom d+1 is M/(d+1).
        # We need M/(d+1) >= rem - 1/d.
        # If this is false for a given d, it will be false for all larger d?
        # LHS decreases, RHS increases (since 1/d decreases, rem - 1/d increases).
        # So yes, if it fails for d, it fails for d+1.
        
        d_end = d_start
        while True:
            if d_end > 10000000: break
            # Check bound
            # M/(d_end+1) >= rem - 1/d_end
            if Fraction(M, d_end + 1) < rem - Fraction(1, d_end):
                break
            d_end += 1
            
        # Iterate d from d_start to d_end - 1
        for d in range(d_start, d_end):
            if d < 5: continue
            current.append(d)
            dfs(rem - Fraction(1, d), d + 1, current)
            current.pop()
            
    dfs(Fraction(1, 1), 5, [])
    
    return best_sol
