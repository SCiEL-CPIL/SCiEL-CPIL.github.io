import math
import time

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin before 90s kill
    
    N = math.factorial(12)  # 479001600
    
    # Precompute divisors of N
    # N = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = [1]
    primes_exp = [(2, 10), (3, 5), (5, 2), (7, 1), (11, 1)]
    for p, e in primes_exp:
        new_divs = []
        for d in divisors:
            for i in range(e + 1):
                new_divs.append(d * (p ** i))
        divisors = new_divs
    
    # Filter: > 12 and sort
    valid_divisors = sorted([d for d in divisors if d > 12])
    
    # Map remaining product to its valid divisors > 12
    # This allows fast lookup in the DFS
    # Key: remaining product
    # Value: list of divisors d such that d > 12 and d divides remaining
    # We can just compute this on the fly or precompute for all divisors of N.
    # Since there are only 792 divisors, precomputation is trivial.
    
    divisors_of_N = set(divisors)
    divs_map = {}
    for rem in divisors_of_N:
        divs_map[rem] = [d for d in valid_divisors if rem % d == 0]
        # Sort to ensure we pick in increasing order
        divs_map[rem].sort()
        
    # Memoization for is_feasible
    # State: (remaining, last_factor)
    # We also need to track depth to enforce <= 12 factors.
    # But since we are checking feasibility for a specific max_val,
    # we can just check if a solution exists.
    
    def can_decompose(rem, last, depth, max_val):
        if time.time() - start_time > deadline:
            return False
        
        if rem == 1:
            return True
        
        if depth >= 12:
            return False
            
        # Get candidates
        candidates = divs_map.get(rem, [])
        
        for d in candidates:
            if d <= last:
                continue
            if d > max_val:
                break
            
            if can_decompose(rem // d, d, depth + 1, max_val):
                return True
                
        return False

    # Binary search for the minimum max_val
    low = 13
    high = N
    best_max = N
    best_factors = [N]
    
    while low <= high:
        if time.time() - start_time > deadline:
            break
            
        mid = (low + high) // 2
        
        # Check if decomposition is possible with max factor <= mid
        if can_decompose(N, 12, 0, mid):
            # Try to find the actual factors to return
            # We need a reconstruction pass
            # We can modify can_decompose to return the path, or do a second DFS
            # Given the speed, a second DFS is fine.
            
            path = []
            def find_path(rem, last, depth, max_val):
                if rem == 1:
                    return True
                if depth >= 12:
                    return False
                candidates = divs_map.get(rem, [])
                for d in candidates:
                    if d <= last:
                        continue
                    if d > max_val:
                        break
                    path.append(d)
                    if find_path(rem // d, d, depth + 1, max_val):
                        return True
                    path.pop()
                return False
            
            if find_path(N, 12, 0, mid):
                best_max = mid
                best_factors = list(path)
                high = mid - 1
            else:
                # Should not happen if can_decompose is correct, but safety
                low = mid + 1
        else:
            low = mid + 1
            
    # If no solution found (should not happen), fallback
    if not best_factors:
        best_factors = [N]
        
    return best_factors
