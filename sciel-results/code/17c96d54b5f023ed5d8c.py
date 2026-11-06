import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    N = 479001600  # 12!
    
    best_max_factor = N
    best_factors = [N]
    
    # Precompute divisors of N that are > 12?
    # Or just iterate?
    # Iterating f from min_factor to sqrt(remaining) is efficient.
    
    def dfs(remaining, min_factor, current_factors, depth):
        nonlocal best_max_factor, best_factors
        
        if time.time() > deadline:
            return

        if remaining == 1:
            if current_factors:
                max_f = max(current_factors)
                if max_f < best_max_factor:
                    best_max_factor = max_f
                    best_factors = current_factors[:]
            return
            
        if depth >= 12:
            return

        # The next factor f must be >= min_factor
        # And f must divide remaining
        # And remaining // f must be >= min_factor (if we stop here, remaining//f must be 1? No, if we pick f, remaining becomes remaining//f. If remaining//f == 1, we are done. If remaining//f > 1, we need more factors.)
        
        # Optimization: 
        # If we pick f, the new remaining is R' = remaining // f.
        # If R' == 1, we add f and finish.
        # If R' > 1, we must factor R' into numbers > f.
        # The smallest possible next factor is f+1.
        # So R' must be at least f+1? 
        # Actually, R' is the product of the remaining factors.
        # The smallest product of one factor > f is f+1.
        # So if R' < f+1 and R' != 1, it's impossible.
        
        # Also, we can limit the upper bound of f.
        # If we pick f, the remaining product is R'.
        # We want to minimize the max factor.
        # This doesn't directly bound f from above in a simple way, but f <= sqrt(remaining) is a good heuristic if we expect at least 2 more factors?
        # If we only pick one more factor, f can be up to remaining/13 (since next > 12, but next > f).
        # So f * (f+1) <= remaining?
        # If we pick f, the smallest possible continuation is f+1.
        # So f * (f+1) <= remaining is a necessary condition for being able to have at least one more factor?
        # Wait, if R' == 1, we don't need more factors.
        # So if remaining // f == 1, it's valid.
        # If remaining // f > 1, we need to factor it. The smallest factor is f+1.
        # So we need remaining // f >= f + 1.
        # => remaining >= f * (f + 1).
        
        # So the upper bound for f is roughly sqrt(remaining).
        # Let's iterate f from min_factor to int(math.isqrt(remaining)) + 1
        
        # However, if remaining is prime and > min_factor, we might miss it if we stop at sqrt.
        # If f > sqrt(remaining), then remaining // f < sqrt(remaining) < f.
        # So the next factor would be smaller than f, which violates the distinct/increasing order.
        # UNLESS remaining // f == 1.
        # So we can iterate f up to int(math.isqrt(remaining)).
        # Plus, we must check if remaining itself is a valid factor (i.e., if we stop here).
        # But if we stop here, remaining must be 1? No, if we pick f=remaining, then remaining//f = 1.
        # But f must be >= min_factor.
        # So if remaining >= min_factor, we can pick f=remaining.
        # This corresponds to the case where this is the last factor.
        
        # Strategy:
        # 1. Check if remaining can be the last factor.
        #    If remaining >= min_factor:
        #       new_factors = current_factors + [remaining]
        #       max_f = max(new_factors)
        #       if max_f < best_max_factor: update.
        
        # 2. Try to split remaining into f * R' where f >= min_factor and f < remaining.
        #    f must divide remaining.
        #    R' = remaining // f.
        #    R' must be factorable into numbers > f.
        #    This implies R' >= f + 1 (if R' > 1).
        #    So f * (f + 1) <= remaining.
        
        # Let's implement this.
        
        # Case 1: remaining is the last factor
        if remaining >= min_factor:
            new_max = max(current_factors + [remaining])
            if new_max < best_max_factor:
                best_max_factor = new_max
                best_factors = current_factors + [remaining]

        # Case 2: Split remaining
        # f ranges from min_factor to int(sqrt(remaining))
        # Also f must be a divisor.
        
        # We can iterate through divisors of remaining that are >= min_factor.
        # But generating divisors each time is slow.
        # Iterating integers and checking divisibility is okay for small N, but N is large.
        # However, the depth is small.
        
        # Let's iterate f from min_factor to int(math.isqrt(remaining))
        # And check if remaining % f == 0.
        
        # Optimization: 
        # The condition R' >= f + 1 means remaining // f >= f + 1
        # => remaining >= f * (f + 1)
        # So f_max = floor((-1 + sqrt(1 + 4*remaining)) / 2)
        # This is approx sqrt(remaining).
        
        limit = int(math.isqrt(remaining))
        # Ensure limit is not less than min_factor
        if limit < min_factor:
            return
            
        for f in range(min_factor, limit + 1):
            if time.time() > deadline:
                return
                
            if remaining % f != 0:
                continue
                
            R_prime = remaining // f
            
            # Check if R_prime can be formed by factors > f
            # If R_prime == 1, it's handled by Case 1? No, Case 1 handles remaining being the factor.
            # Here we are splitting. If R_prime == 1, then f == remaining.
            # But f <= sqrt(remaining), so f < remaining (unless remaining is 1, but we handled that).
            # So R_prime > 1.
            
            # We need to factor R_prime using factors > f.
            # The smallest possible factor is f + 1.
            # So we require R_prime >= f + 1.
            if R_prime < f + 1:
                continue
                
            dfs(R_prime, f + 1, current_factors + [f], depth + 1)

    dfs(N, 13, [], 0)
    
    return best_factors
