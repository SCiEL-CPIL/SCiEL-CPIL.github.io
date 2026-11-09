import math
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 90s limit
    
    target = math.factorial(12)  # 479001600
    
    # Prime factorization of 12!: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors > 12 using prime factorization
    divisors_set = set()
    current_divs = [1]
    for i, (p, exp) in enumerate(zip(primes, exponents)):
        new_divs = []
        for d in current_divs:
            val = d
            for e in range(exp + 1):
                if val > 12:
                    divisors_set.add(val)
                new_divs.append(val)
                if e < exp:
                    val *= p
        current_divs = new_divs
    for d in current_divs:
        if d > 12:
            divisors_set.add(d)
    
    candidates = sorted(divisors_set)
    
    if not candidates:
        return [target]
    
    # Map each candidate to its exponent vector for fast subtraction
    def factorize(n):
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
    
    cand_exp = {c: factorize(c) for c in candidates}
    target_exp = tuple(exponents)
    
    def rem_exp_to_val(rem_exp):
        val = 1
        for i, p in enumerate(primes):
            val *= p ** rem_exp[i]
        return val
    
    # Direct optimized search: try to find a decomposition with small max factor
    # Strategy: try increasing max bounds, but use smart ordering and pruning
    
    best_sol = None
    best_max = float('inf')
    
    # Try a range of max factor bounds
    # Theoretical lower bound: if we use k factors, each ~ (12!)^(1/k)
    # For k=7: ~24.5, k=8: ~20.5, k=9: ~17.8, k=10: ~15.9, k=11: ~14.7, k=12: ~13.7
    
    # Try max bounds from low to high, stop when we find a solution
    # But also try to find the best one within time budget
    
    # First, try to find any solution with a reasonable max
    # Then try to improve
    
    # Use a priority-based search: prefer smaller factors first to keep max low
    
    fail_cache = set()
    solution_found = []
    
    def dfs(rem_exp, start_idx, depth, current_factors):
        nonlocal best_sol, best_max, solution_found
        if time.time() > deadline:
            return False
        if depth > 12:
            return False
        if all(e == 0 for e in rem_exp):
            if current_factors:
                sol = list(current_factors)
                max_f = max(sol)
                if max_f < best_max:
                    best_max = max_f
                    best_sol = sol
                    solution_found.append(sol)
            return True
        
        rem_val = rem_exp_to_val(rem_exp)
        
        # Quick check: if rem_val itself is a valid candidate at or after start_idx
        if rem_val > 12 and rem_val in cand_exp:
            idx = bisect.bisect_left(candidates, rem_val)
            if idx < len(candidates) and candidates[idx] == rem_val and idx >= start_idx:
                new_max = max(current_factors + [rem_val]) if current_factors else rem_val
                if new_max < best_max:
                    best_max = new_max
                    best_sol = list(current_factors) + [rem_val]
                    solution_found.append(best_sol)
                return True
        
        if (rem_exp, start_idx) in fail_cache:
            return False
        
        # Pruning: if rem_val < smallest candidate at start_idx, fail
        if start_idx < len(candidates):
            min_cand = candidates[start_idx]
            if rem_val < min_cand:
                fail_cache.add((rem_exp, start_idx))
                return False
        
        # Pruning: if remaining product can't be split into (12 - depth) distinct factors all >= next candidate
        # The minimum product of (12-depth) distinct factors starting from start_idx
        remaining_slots = 12 - depth
        if remaining_slots > 1 and start_idx + remaining_slots - 1 < len(candidates):
            min_possible = 1
            for j in range(start_idx, start_idx + remaining_slots - 1):
                min_possible *= candidates[j]
            if rem_val < min_possible:
                fail_cache.add((rem_exp, start_idx))
                return False
        
        # Try candidates in order, but limit the range for efficiency
        # We want to try smaller factors first to minimize the max
        for i in range(start_idx, len(candidates)):
            f = candidates[i]
            if f > rem_val:
                break
            
            # If we already have a solution, skip factors that would exceed best_max
            if best_sol is not None and f > best_max and depth >= 1:
                # Only allow this if it's the last factor
                if depth < 11:
                    continue
            
            f_exp = cand_exp[f]
            valid = True
            for k in range(len(primes)):
                if f_exp[k] > rem_exp[k]:
                    valid = False
                    break
            if not valid:
                continue
            
            new_rem_exp = tuple(rem_exp[k] - f_exp[k] for k in range(len(primes)))
            current_factors.append(f)
            if dfs(new_rem_exp, i + 1, depth + 1, current_factors):
                current_factors.pop()
                return True
            current_factors.pop()
        
        fail_cache.add((rem_exp, start_idx))
        return False
    
    # Run the search
    dfs(target_exp, 0, 0, [])
    
    if best_sol is not None:
        # Verify
        prod = 1
        for f in best_sol:
            prod *= f
        if prod == target and len(set(best_sol)) == len(best_sol) and all(f > 12 for f in best_sol) and 1 <= len(best_sol) <= 12:
            return best_sol
    
    return [target]
