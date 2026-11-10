import math
import time

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    # Prime factorization of 12!
    # 2: 10, 3: 5, 5: 2, 7: 1, 11: 1
    primes = [2, 3, 5, 7, 11]
    total_exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors of N
    # N = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = []
    for a in range(11):
        for b in range(6):
            for c in range(3):
                for d in range(2):
                    for e in range(2):
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        if val > 12:
                            divisors.append((val, [a, b, c, d, e]))
                            
    # Sort by value
    divisors.sort(key=lambda x: x[0])
    
    # We want to find the minimal M such that there exists a subset of divisors <= M
    # whose exponent vectors sum to total_exponents.
    # The subset size must be <= 12.
    
    # Lower bound for M
    # Geometric mean of 12 factors is ~47.
    # Let's start from 47.
    
    # To speed up, we can pre-filter or just iterate M.
    # Since divisors are sorted, we can find the index up to M.
    
    # Memoization key: (e2, e3, e5, e7, e11, last_idx, count)
    # We want to know if a solution exists from this state.
    # We also need to reconstruct the path, so we can return the list of factors.
    
    # Since we iterate M from low to high, the first success is optimal.
    
    # To avoid deep recursion overhead and repeated work, we can use a set for visited states
    # in the DFS for a specific M. But since M changes, the set of available candidates changes.
    # However, if a state (e, last_idx, count) is reachable with a set of candidates S_M,
    # it is also reachable with S_M' where S_M' is a superset? No, we are constrained by last_idx.
    # Actually, if we sort candidates, `last_idx` refers to the index in the sorted list.
    # If M increases, we add more candidates at the end (higher indices).
    # So `last_idx` remains valid.
    # If a state is reachable with candidates up to index `i`, it is reachable with candidates up to `j > i`?
    # Yes, because the previous choices are still valid.
    # So we can potentially reuse memoization across M values?
    # No, because the set of *available* candidates for the *next* step changes.
    # But if we define the state as `(e2, e3, e5, e7, e11, last_idx, count)`,
    # and the transition is "pick a candidate at index > last_idx",
    # then if we know that from state `S` with candidates up to `K` we can reach the goal,
    # does it help for candidates up to `K+1`?
    # Not directly, because the new candidate at `K+1` might be the one needed.
    # But if we just want to know if *any* solution exists, and we are increasing M,
    # we can just run DFS for each M.
    # The number of divisors is ~700.
    # M will likely be found quickly (e.g., M=50-60).
    # The number of candidates for M=50 is small.
    
    # Let's implement a DFS that takes the current exponents, the index of the last chosen factor,
    # and the current path.
    # We iterate through candidates from `last_idx + 1` to `max_idx` (where candidates[max_idx] <= M).
    
    # Optimization: Pruning
    # If sum of max possible exponents from remaining candidates < remaining exponents, fail.
    # But calculating this is expensive.
    # Simple pruning: if count > 12, fail.
    # if e2 > 10 or e3 > 5 etc, fail (should be handled by transition).
    
    # Let's just run it.
    
    best_factors = None
    best_M = N
    
    # We can binary search M?
    # If a solution exists for M, it exists for M+1.
    # So we can binary search for the smallest M.
    # Range: 47 to max(divisor).
    # Max divisor is N.
    # But N is huge.
    # However, the solution likely has M < 100.
    # Let's just linear scan from 47.
    
    # To be safe, let's cap the search time.
    
    # Pre-compute indices for each M? No, just use bisect.
    import bisect
    
    values = [d[0] for d in divisors]
    
    def dfs(e_rem, last_idx, count, path, max_idx):
        if time.time() - start_time > 55:
            return None
            
        if count > 12:
            return None
            
        if e_rem[0] == 0 and e_rem[1] == 0 and e_rem[2] == 0 and e_rem[3] == 0 and e_rem[4] == 0:
            return list(path)
            
        # Pruning: if we have no more candidates or not enough slots
        if last_idx + 1 > max_idx:
            return None
            
        # Check if it's even possible to reach 0 exponents with remaining slots
        # Max exponent contribution per slot is bounded by the largest candidate's exponents?
        # Not exactly, but we can check if sum of remaining exponents is 0.
        
        # Try each candidate from last_idx + 1 to max_idx
        for i in range(last_idx + 1, max_idx + 1):
            val, exps = divisors[i]
            # Check if exps <= e_rem
            if exps[0] <= e_rem[0] and exps[1] <= e_rem[1] and exps[2] <= e_rem[2] and exps[3] <= e_rem[3] and exps[4] <= e_rem[4]:
                new_e_rem = [e_rem[j] - exps[j] for j in range(5)]
                path.append(val)
                res = dfs(new_e_rem, i, count + 1, path, max_idx)
                if res is not None:
                    return res
                path.pop()
                
        return None

    for M in range(47, 200): # Upper bound guess, can increase if needed
        if time.time() - start_time > 55:
            break
            
        # Find max index for M
        max_idx = bisect.bisect_right(values, M) - 1
        
        if max_idx < 0:
            continue
            
        # Initial state
        initial_e_rem = total_exponents[:]
        path = []
        
        res = dfs(initial_e_rem, -1, 0, path, max_idx)
        
        if res is not None:
            return res
            
    # If not found in 47..199, expand search or return fallback
    # Fallback: return [N] is valid? No, N is not > 12? N is 479001600 > 12.
    # But we want to minimize max factor.
    # If we didn't find a small M, we should try larger M or return the single factor N.
    # But N is a single factor. Is it valid?
    # "1..12 DISTINCT integers". Yes, 1 integer is allowed.
    # But we want to minimize the largest.
    # If no combination of <= 12 factors with max < N exists, then N is the answer.
    # But surely a combination exists with max < N.
    # Let's just return [N] if we run out of time or range, but try to find something better.
    
    # Let's try a few more M values if time permits
    for M in range(200, 1000):
        if time.time() - start_time > 58:
            break
        max_idx = bisect.bisect_right(values, M) - 1
        if max_idx < 0: continue
        initial_e_rem = total_exponents[:]
        path = []
        res = dfs(initial_e_rem, -1, 0, path, max_idx)
        if res is not None:
            return res
            
    return [N]
