import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe

    # Known 8-term solution
    best_solution = [5, 6, 7, 42, 140, 420, 840, 2520]
    best_len = len(best_solution)
    
    max_den = 10000000
    min_den = 5

    # DFS with integer arithmetic
    # State: N, D (remaining sum = N/D), last_d, path
    # We want N/D = 1
    N, D = 1, 1
    
    def search(N, D, last_d, path):
        nonlocal best_solution, best_len
        if time.time() > deadline:
            return
        
        if N == 0:
            if len(path) < best_len:
                best_solution = path[:]
                best_len = len(path)
            return
            
        if len(path) >= best_len:
            return
            
        # Pruning:
        # 1. Lower bound for next d: d > last_d and 1/d <= N/D => d >= D/N
        # 2. Upper bound for next d: 
        #    We have k = best_len - len(path) terms left to place (including this one? No, this is the current choice).
        #    Actually, we are choosing the next term.
        #    Let k be the number of terms remaining to be chosen, including the current one?
        #    No, we are at a node where we have chosen `len(path)` terms.
        #    We need to choose `best_len - len(path)` more terms? No, we can choose fewer.
        #    But we prune if `len(path) >= best_len`. So we are looking for strictly better.
        #    Max terms we can add is `best_len - 1 - len(path)`? No, we can add up to `best_len - len(path)` terms.
        #    If we add `k` terms, the sum is at most k / d_min? No.
        #    The sum of the remaining terms is N/D.
        #    The smallest possible sum of `k` distinct denominators > last_d is approx k / (last_d + k).
        #    The largest possible sum is approx k / (last_d + 1).
        #    So N/D <= k / (last_d + 1) is a necessary condition?
        #    Actually, N/D = 1/d1 + ... + 1/dk.
        #    Since d1 > last_d, 1/d1 < 1/(last_d+1).
        #    So N/D < k / (last_d+1).
        #    This gives a lower bound on k: k > N/D * (last_d+1).
        #    If `best_len - len(path)` is the max number of terms we can use, then:
        #    Let max_terms = best_len - len(path)
        #    If N/D > max_terms / (last_d + 1), it's impossible to reach N/D with max_terms terms.
        #    Wait, we want to MINIMIZE count. So we are searching for count < best_len.
        #    So the max number of terms we can use is best_len - 1.
        #    Let max_terms_allowed = best_len - 1 - len(path)
        #    If N/D > max_terms_allowed / (last_d + 1), we can't do it.
        
        max_terms_allowed = best_len - 1 - len(path)
        if max_terms_allowed <= 0:
            return
            
        # Check if N/D can be represented by max_terms_allowed terms > last_d
        # Max sum with max_terms_allowed terms > last_d is approx max_terms_allowed / (last_d + 1)
        # If N/D > max_terms_allowed / (last_d + 1), fail.
        # N * (last_d + 1) > D * max_terms_allowed
        if N * (last_d + 1) > D * max_terms_allowed:
            return
            
        # Min d
        min_d = max(last_d + 1, D // N + 1 if N > 0 else max_den + 1)
        if min_d > max_den:
            return
            
        # Max d
        # N/D = 1/d + rem. rem > 0.
        # Also, we need to be able to form the rest.
        # A simpler upper bound: d must be such that 1/d < N/D.
        # d > D/N.
        # Also, if we pick d, the remaining sum is N/D - 1/d = (Nd - D) / (Dd).
        # This remaining sum must be representable by (max_terms_allowed - 1) terms.
        
        # Iterate d in a limited range around the greedy choice
        # Greedy choice: smallest d such that 1/d <= N/D.
        d_greedy = max(min_d, D // N + 1)
        
        # Try d from d_greedy to d_greedy + 20
        # Also try a few smaller d if they are valid?
        # Smaller d means larger 1/d, so N/D - 1/d is smaller.
        # We want to leave a sum that is easy to decompose.
        
        for d in range(d_greedy, d_greedy + 20):
            if time.time() > deadline:
                return
            if d > max_den:
                break
            
            # Check if 1/d <= N/D
            if d * N < D:
                break # 1/d > N/D, so this and larger d are invalid? No, larger d means smaller 1/d.
                # Wait. 1/d <= N/D => d >= D/N.
                # So if d * N < D, then 1/d > N/D. This is INVALID.
                # Since d is increasing, 1/d is decreasing.
                # So if d * N < D, we must INCREASE d.
                # My loop starts at d_greedy which is >= D/N.
                # So d * N >= D should hold.
                pass
            
            # Calculate new N, D
            # New Sum = N/D - 1/d = (Nd - D) / (Dd)
            new_N = N * d - D
            new_D = D * d
            
            if new_N == 0:
                path.append(d)
                search(0, 1, d, path) # Optimization: if N becomes 0, we are done
                path.pop()
            elif new_N > 0:
                # Reduce fraction?
                # g = gcd(new_N, new_D)
                # new_N //= g
                # new_D //= g
                # Skipping reduction might make numbers large, but helps speed.
                # However, large numbers might cause overflow or slow comparisons.
                # Let's reduce.
                g = gcd(new_N, new_D)
                if g > 1:
                    new_N //= g
                    new_D //= g
                
                path.append(d)
                search(new_N, new_D, d, path)
                path.pop()
            else:
                # new_N < 0 means 1/d > N/D, which shouldn't happen if d >= D/N
                continue

    search(N, D, 4, [])
    
    return best_solution
