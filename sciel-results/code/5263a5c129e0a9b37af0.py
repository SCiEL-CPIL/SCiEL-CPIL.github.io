import time
from math import factorial

def construct():
    start = time.time()
    deadline = start + 85

    N = factorial(12)  # 479001600
    # Prime factorization: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    total_exp = (10, 5, 2, 1, 1)
    primes = (2, 3, 5, 7, 11)

    # Generate all divisors of N with their exponent vectors
    def gen_divisors():
        divs = []
        def rec(idx, current_val, exps):
            if idx == len(primes):
                if current_val > 12:
                    divs.append((current_val, tuple(exps)))
                return
            p = primes[idx]
            max_e = total_exp[idx]
            for e in range(max_e + 1):
                exps[idx] = e
                rec(idx + 1, current_val * (p ** e), exps)
            exps[idx] = 0
        rec(0, 1, [0] * 5)
        divs.sort(key=lambda x: x[0])
        return divs

    divs = gen_divisors()
    # divs is list of (value, exp_vector)
    values = [d[0] for d in divs]
    exps = [d[1] for d in divs]

    # Binary search on M
    def can_factor(M):
        if time.time() > deadline:
            return None
        
        # Filter divs to those with value <= M
        # Find the index
        lo, hi = 0, len(values)
        while lo < hi:
            mid = (lo + hi) // 2
            if values[mid] <= M:
                lo = mid + 1
            else:
                hi = mid
        local_vals = values[:lo]
        local_exps = exps[:lo]
        
        if not local_vals:
            return None
        
        # DFS on exponent vectors
        # State: (idx, rem_exp, count, chosen)
        # rem_exp is a tuple of 5 ints
        
        result = None
        
        def dfs(idx, rem_exp, count, chosen):
            nonlocal result
            if time.time() > deadline:
                return False
            
            if rem_exp == (0, 0, 0, 0, 0):
                result = list(chosen)
                return True
            
            if count >= 12:
                return False
            
            if idx >= len(local_vals):
                return False
            
            # Prune: if any component of rem_exp is negative, fail (shouldn't happen)
            for i in range(5):
                if rem_exp[i] < 0:
                    return False
            
            # Prune: the minimum value we can form with remaining factors
            # If rem_exp corresponds to a value < local_vals[idx], fail
            # Compute the value from rem_exp
            rem_val = 1
            for i in range(5):
                rem_val *= primes[i] ** rem_exp[i]
            
            if rem_val < local_vals[idx]:
                return False
            
            # Prune: if rem_val > 1, we need at least one more factor
            # The maximum value we can form with (12-count-1) more factors
            # is bounded by the product of the largest (12-count-1) local_vals from idx onward
            # But that's expensive. Skip for now.
            
            for i in range(idx, len(local_vals)):
                v = local_vals[i]
                e = local_exps[i]
                
                # Check if we can subtract e from rem_exp
                can_sub = True
                for j in range(5):
                    if rem_exp[j] < e[j]:
                        can_sub = False
                        break
                if not can_sub:
                    continue
                
                # New rem_exp
                new_rem = tuple(rem_exp[j] - e[j] for j in range(5))
                
                chosen.append(v)
                if dfs(i + 1, new_rem, count + 1, chosen):
                    chosen.pop()
                    return True
                chosen.pop()
            
            return False
        
        if dfs(0, total_exp, 0, []):
            return result
        return None

    # Binary search on M
    lo, hi = 13, N
    best = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        res = can_factor(mid)
        if res is not None:
            best = res
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best is None:
        return [N]
    
    # Verify
    if len(best) > 12 or len(best) < 1:
        return [N]
    if any(f <= 12 for f in best):
        return [N]
    if len(set(best)) != len(best):
        return [N]
    prod = 1
    for f in best:
        prod *= f
    if prod != N:
        return [N]
    
    return best
