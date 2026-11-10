import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    N = 479001600  # 12!
    
    def get_divisors(n):
        """Generate all divisors of n."""
        divs = [1]
        d = 2
        while d * d <= n:
            if n % d == 0:
                new_divs = []
                for dv in divs:
                    new_divs.append(dv * d)
                    e = d
                    while n % e == 0:
                        e *= d
                        new_divs.append(dv * e)
                divs.extend(new_divs)
            d += 1
        return sorted(divs)
    
    # Get all divisors of N
    all_divs = get_divisors(N)
    # Filter: must be > 12
    valid_divs = [d for d in all_divs if d > 12]
    
    # Create a set for O(1) lookup
    valid_divs_set = set(valid_divs)
    
    # For each valid divisor, compute its "signature" as a tuple of prime exponents
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    
    def prime_signature(n):
        sig = []
        for p in primes:
            e = 0
            while n % p == 0:
                e += 1
                n //= p
            sig.append(e)
        return tuple(sig)
    
    target_sig = (10, 5, 2, 1, 1)
    
    # Precompute signatures for all valid divisors
    sig_map = {}
    for d in valid_divs:
        sig_map[d] = prime_signature(d)
    
    # Strategy: Binary search on the maximum factor M
    # For a given M, check if we can decompose N into 1-12 distinct integers in (12, M]
    
    def can_decompose(N, max_val, min_factors=1, max_factors=12):
        """
        Check if N can be decomposed into distinct integers in (12, max_val].
        Uses DFS with pruning.
        """
        if time.time() - start_time > DEADLINE:
            return None
        
        # Get candidate divisors in (12, max_val]
        candidates = [d for d in valid_divs if d <= max_val]
        if not candidates:
            return None
        
        # Sort candidates
        candidates.sort()
        
        # Memoization: (remaining_N, min_factor, count_used) -> (success, factors)
        # But this might be too large. Let's use a different approach.
        
        # DFS with backtracking
        # We pick factors in increasing order to avoid permutations
        used = set()
        path = []
        
        def dfs(remaining, min_factor, count):
            if time.time() - start_time > DEADLINE:
                return None
            
            if remaining == 1:
                if min_factors <= count <= max_factors:
                    return list(path)
                return None
            
            if count >= max_factors:
                return None
            
            # Pruning: if remaining is prime and > max_val, impossible
            # (but remaining might be composite with small factors)
            
            # Find candidates that divide remaining and are >= min_factor
            for i in range(len(candidates)):
                c = candidates[i]
                if c < min_factor:
                    continue
                if c > max_val:
                    break
                if remaining % c != 0:
                    continue
                if c in used:
                    continue
                
                # Pruning: the smallest possible product of (max_factors - count) distinct 
                # integers >= c+1 must be <= remaining // c
                # Actually, we need at least (max_factors - count) more factors, each >= c+1
                # But we also need at least 1 more factor (since remaining > 1)
                # Let's check: can we form remaining//c with (max_factors - count) distinct ints > c?
                # Minimum product of k distinct ints > c is (c+1)*(c+2)*...*(c+k)
                # But this is a lower bound on what we need to form.
                # Actually, we need remaining//c to be decomposable into 
                # (max_factors - count) distinct ints > c, each <= max_val.
                # A necessary condition: remaining//c >= (c+1) if we need at least 1 more factor
                
                new_remaining = remaining // c
                new_count = count + 1
                
                # If new_remaining == 1, we're done (checked above)
                # If new_remaining > 1, we need at least 1 more factor
                # The smallest next factor is c+1
                if new_remaining > 1:
                    if new_count >= max_factors:
                        continue
                    # Check if new_remaining can be formed by factors > c
                    # Simple check: new_remaining must be >= c+1 (if we need 1 more)
                    # Or more generally, check feasibility
                    pass
                
                used.add(c)
                path.append(c)
                result = dfs(new_remaining, c + 1, new_count)
                path.pop()
                used.discard(c)
                
                if result is not None:
                    return result
            
            return None
        
        return dfs(N, 13, 0)
    
    # Binary search on max factor
    # Lower bound: 13 (smallest integer > 12)
    # Upper bound: N (single factor)
    lo = 13
    hi = N
    best_solution = None
    
    while lo <= hi:
        if time.time() - start_time > DEADLINE:
            break
        
        mid = (lo + hi) // 2
        solution = can_decompose(N, mid)
        
        if solution is not None:
            best_solution = solution
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_solution is None:
        # Fallback: single factor N
        best_solution = [N]
    
    return best_solution
