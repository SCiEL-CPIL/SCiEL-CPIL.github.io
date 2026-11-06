"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, isqrt

def construct():
    target = factorial(12)  # 479001600
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Precompute divisors of target
    # We'll use DFS: pick distinct factors > 12 in increasing order
    # Product must equal target exactly
    
    best = [target]  # single factor = 12! itself, max = 479001600
    best_max = target
    
    # Get all divisors of target
    def get_divisors(n):
        divs = []
        for i in range(1, isqrt(n) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return sorted(divs)
    
    all_divs = get_divisors(target)
    # Filter divisors > 12
    valid_divs = [d for d in all_divs if d > 12]
    
    # DFS: build list of factors in increasing order
    # At each step, choose next factor > last_factor, factor must divide remaining
    # Prune: if current_max >= best_max, stop
    # Prune: remaining must be expressible as product of distinct divisors > last_factor, all <= best_max
    
    def dfs(remaining, last_factor, current_factors, current_max):
        nonlocal best, best_max
        
        if time.time() > deadline:
            return
        
        if remaining == 1:
            if len(current_factors) >= 1 and current_max < best_max:
                best_max = current_max
                best = list(current_factors)
            return
        
        # If remaining is prime and > 12, it must be a single factor
        # Check if remaining is prime
        if is_prime(remaining) and remaining > 12:
            if remaining > last_factor and remaining <= best_max and len(current_factors) + 1 <= 12:
                if remaining > last_factor:
                    new_max = max(current_max, remaining)
                    if new_max < best_max:
                        best_max = new_max
                        best = current_factors + [remaining]
            return
        
        # Try each possible next factor
        # Factor must be > last_factor, divide remaining, and <= best_max
        # Also, remaining / factor must be expressible with distinct factors > factor
        for d in valid_divs:
            if d <= last_factor:
                continue
            if d > best_max:
                break
            if remaining % d != 0:
                continue
            if len(current_factors) + 1 > 12:
                break
            
            new_remaining = remaining // d
            new_max = max(current_max, d)
            
            # Prune: if new_max >= best_max, skip
            if new_max >= best_max:
                continue
            
            # Prune: new_remaining must be >= 1
            # If new_remaining == 1, we've found a complete factorization
            if new_remaining == 1:
                if len(current_factors) + 1 >= 1:
                    if new_max < best_max:
                        best_max = new_max
                        best = current_factors + [d]
                continue
            
            # Additional pruning: new_remaining must have a divisor > d
            # (since all factors must be distinct and > d)
            # Actually, new_remaining could be 1 (handled above) or must be
            # product of distinct factors all > d
            # Simple prune: new_remaining > d (since smallest factor > d is at least d+1)
            # But that's not quite right - new_remaining could be prime > d
            # Better: check if new_remaining has any divisor > d that also divides it
            # For now, just recurse
            
            dfs(new_remaining, d, current_factors + [d], new_max)
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    dfs(target, 12, [], 0)
    
    return best

# EVOLVE-BLOCK-END
