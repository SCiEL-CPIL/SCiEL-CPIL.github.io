import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin
    
    N = factorial(12)
    
    # Generate all divisors of N
    divisors = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
        i += 1
        
    # Filter divisors > 12 and sort ascending
    valid_divisors = sorted([d for d in divisors if d > 12])
    
    best_max = N + 1
    best_solution = None
    
    # Greedy heuristic to find an initial valid solution and set a tight bound
    # We try to balance the factors to keep the max low.
    def greedy_balanced(k_target):
        rem = N
        factors = []
        last = 12
        slots_left = k_target
        while rem > 1 and slots_left > 0:
            if slots_left == 1:
                if rem > last and rem > 12:
                    factors.append(rem)
                    return factors
                else:
                    return None
            
            # Target value for next factor
            # We want the remaining product to be distributable.
            # Ideal next factor is roughly (rem) ** (1/slots_left)
            target = rem ** (1.0 / slots_left)
            
            best_d = None
            min_diff = float('inf')
            
            # Search for a divisor close to target
            # We must ensure we can still form distinct factors > d
            # So d should not be too large, but also not too small.
            
            for d in valid_divisors:
                if d <= last:
                    continue
                if d > rem:
                    break
                if rem % d != 0:
                    continue
                
                # Check if d is "good"
                # If d is too small, the remaining factors might need to be large.
                # If d is too large, we might not have enough distinct factors left.
                
                diff = abs(d - target)
                if diff < min_diff:
                    # Heuristic check: can we fit slots_left-1 distinct factors > d in rem//d?
                    # This is expensive, so we just pick the closest one for now.
                    min_diff = diff
                    best_d = d
            
            if best_d is None:
                # Fallback: pick smallest valid divisor
                for d in valid_divisors:
                    if d > last and d <= rem and rem % d == 0:
                        best_d = d
                        break
                if best_d is None:
                    return None
            
            factors.append(best_d)
            rem //= best_d
            last = best_d
            slots_left -= 1
            
        return factors if factors else None

    # Try different target counts to find a good initial solution
    for k in range(12, 1, -1):
        sol = greedy_balanced(k)
        if sol:
            current_max = max(sol)
            if current_max < best_max:
                best_max = current_max
                best_solution = sol

    # If no solution found yet, use the trivial one
    if not best_solution:
        best_solution = [N]
        best_max = N

    # Precompute max product of m distinct integers in (x, limit)
    # This is used for pruning.
    # max_prod(m, last, limit) = product of the m largest integers < limit and > last
    # We can compute this on the fly or cache it.
    # Given the small range of factors (likely < 1000), on-the-fly is fine.
    
    def max_possible_product(m, last, limit):
        # Product of m distinct integers from {last+1, ..., limit-1}
        # To maximize product, pick the largest m integers.
        # These are limit-1, limit-2, ..., limit-m
        # But they must be > last.
        # So the set is {max(last+1, limit-m), ..., limit-1}
        # If limit-m <= last, we can't pick m distinct integers > last and < limit.
        # Specifically, we need limit - 1 - last >= m - 1 => limit - last >= m
        if limit - last < m:
            return 0
        
        prod = 1
        for i in range(m):
            val = limit - 1 - i
            if val <= last:
                return 0
            prod *= val
        return prod

    # DFS to minimize the largest factor
    def dfs(remaining, min_idx, path, slots_left):
        nonlocal best_max, best_solution
        
        if time.time() > deadline:
            return
            
        if remaining == 1:
            if path:
                current_max = path[-1]
                if current_max < best_max:
                    best_max = current_max
                    best_solution = list(path)
            return
            
        if slots_left == 0:
            return
            
        # Pruning:
        # 1. If remaining is 1, handled above.
        # 2. If slots_left == 1, remaining must be a valid factor.
        if slots_left == 1:
            if remaining > path[-1] if path else remaining > 12:
                if remaining < best_max:
                    path.append(remaining)
                    dfs(1, len(valid_divisors), path, 0)
                    path.pop()
            return

        # 3. Max possible product prune
        # We need to factor `remaining` into `slots_left` distinct factors, all > last_factor and < best_max.
        # The maximum product of `slots_left` such factors is max_possible_product(slots_left, last_factor, best_max).
        # If remaining > that product, it's impossible.
        last_factor = path[-1] if path else 12
        max_prod = max_possible_product(slots_left, last_factor, best_max)
        if remaining > max_prod:
            return
            
        # Iterate through possible next factors
        for i in range(min_idx, len(valid_divisors)):
            d = valid_divisors[i]
            
            # Pruning: if d >= best_max, no further d can improve best_max
            if d >= best_max:
                break
                
            # Pruning: if d > remaining, no further d can divide remaining
            if d > remaining:
                break
                
            if remaining % d != 0:
                continue
                
            path.append(d)
            dfs(remaining // d, i + 1, path, slots_left - 1)
            path.pop()
            
            # If we found the theoretical minimum, we can stop
            # The smallest possible max factor is bounded by the geometric mean.
            # If best_max is very small, we can stop.
            # For 12! into 12 factors, min max is around 25.
            # If best_max <= 25, it's likely optimal.
            if best_max <= 25:
                return

    dfs(N, 0, [], 12)
    
    if best_solution:
        return best_solution
    else:
        # Fallback: 12! itself
        return [N]
