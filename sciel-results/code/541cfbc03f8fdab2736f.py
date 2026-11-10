"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import factorial

    target = factorial(12)  # 479001600

    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    primes = [2, 3, 5, 7, 11]
    target_exp = [10, 5, 2, 1, 1]
    target_state = tuple(target_exp)

    # Generate all divisors of 12!
    def gen_divisors(exp_bounds, idx=0, current=1):
        if idx == len(exp_bounds):
            return [current]
        result = []
        for e in range(exp_bounds[idx] + 1):
            result.extend(gen_divisors(exp_bounds, idx + 1, current * primes[idx] ** e))
        return result

    all_divisors = gen_divisors(target_exp)
    # Filter: > 12
    candidates = [d for d in all_divisors if d > 12]
    
    # Sort candidates in ascending order
    candidates.sort()

    # Compute exponent vectors for candidates
    def exp_vector(n):
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)

    cand_vecs = [exp_vector(c) for c in candidates]

    # DP to find minimum max factor
    # dp[state] = min max factor to reach state
    # parent[state] = (prev_state, factor_value)
    
    zero_state = (0, 0, 0, 0, 0)
    dp = {zero_state: 0}
    parent = {zero_state: None}
    
    final_state = None
    
    for i, f in enumerate(candidates):
        cv = cand_vecs[i]
        new_dp = dp.copy()
        new_parent = parent.copy()
        
        # Iterate over current states
        for state, current_max in dp.items():
            # Calculate new state
            new_state = tuple(state[j] + cv[j] for j in range(5))
            
            # Check bounds
            if all(new_state[j] <= target_exp[j] for j in range(5)):
                # The new max factor is max(current_max, f)
                # Since we process f in increasing order, and current_max comes from 
                # factors <= f (actually < f, since f is current), 
                # max(current_max, f) is always f.
                new_max = f
                
                # Check if this is better than existing
                if new_max < new_dp.get(new_state, float('inf')):
                    new_dp[new_state] = new_max
                    new_parent[new_state] = (state, f)
                    
                    if new_state == target_state:
                        # We found a solution with max factor f
                        # Since f is increasing, this is the minimal max factor
                        dp = new_dp
                        parent = new_parent
                        final_state = new_state
                        break
        
        if final_state is not None:
            break
        
        dp = new_dp
        parent = new_parent

    if final_state is None:
        return [target]

    # Reconstruct path
    result = []
    state = final_state
    while state != zero_state:
        prev_state, f = parent[state]
        result.append(f)
        state = prev_state
    
    result.reverse()
    
    # Verify
    prod = 1
    for r in result:
        prod *= r
    if prod != target:
        return [target]
    if len(result) != len(set(result)):
        return [target]
    if any(r <= 12 for r in result):
        return [target]
    if len(result) > 12:
        return [target]

    return result
# EVOLVE-BLOCK-END
