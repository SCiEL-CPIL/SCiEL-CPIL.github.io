"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _largest_prime_factor(n):
    """Return the largest prime factor of n, or 1 if n==1."""
    if n <= 1:
        return 1
    result = 1
    # trial by 2
    while n % 2 == 0:
        result = 2
        n //= 2
    # trial by odd numbers
    f = 3
    while f * f <= n:
        while n % f == 0:
            result = f
            n //= f
        f += 2
    if n > 1:
        result = n
    return result

def _is_smooth(n, prime_bound):
    """Check if all prime factors of n are <= prime_bound. Returns True if smooth."""
    if n <= 1:
        return True
    # trial division up to sqrt(n), but only need to check primes up to prime_bound
    # Actually, we just need to verify no prime factor exceeds prime_bound
    f = 2
    while f * f <= n:
        while n % f == 0:
            if f > prime_bound:
                return False
            n //= f
        f += 1
    if n > 1 and n > prime_bound:
        return False
    return True

def _largest_prime_factor_fast(n, prime_bound):
    """
    Find largest prime factor of n. If any prime factor exceeds prime_bound,
    return a value > prime_bound (to signal not smooth enough).
    Otherwise return the actual largest prime factor.
    """
    if n <= 1:
        return 1
    result = 1
    f = 2
    while f * f <= n:
        while n % f == 0:
            if f > prime_bound:
                return f  # signal: exceeds bound
            result = f
            n //= f
        f += 1
    if n > 1:
        if n > prime_bound:
            return n
        result = max(result, n)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 70  # 70s safety margin out of 90s
    
    target = 123457
    
    best_a = 1
    best_b = target - 1
    best_score = _largest_prime_factor(best_b)  # P(1)=1, so max(1, P(123456))
    
    # Strategy: try increasing smoothness bounds M
    # For each M, generate all M-smooth numbers <= target
    # For each such a, check if b = target - a is also M-smooth
    # If yes, we found a pair with max(P(a),P(b)) <= M
    
    # Precompute primes up to some limit for generating smooth numbers
    def generate_smooth_numbers(bound, limit):
        """Generate all numbers <= limit with all prime factors <= bound."""
        # Use recursive generation
        primes = []
        n = 2
        while n <= bound:
            is_p = True
            for p in primes:
                if p * p > n:
                    break
                if n % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(n)
            n += 1
        
        smooth = set()
        smooth.add(1)
        stack = [1]
        idx = 0
        # BFS/DFS to generate smooth numbers
        while idx < len(stack):
            val = stack[idx]
            idx += 1
            for p in primes:
                new_val = val * p
                if new_val > limit:
                    break
                if new_val not in smooth:
                    smooth.add(new_val)
                    stack.append(new_val)
        return sorted(smooth)
    
    # Try increasing M values
    # Start with small M and increase
    M_values = list(range(2, 200)) + list(range(200, 1000, 5)) + list(range(1000, 5000, 20)) + list(range(5000, 20000, 100)) + list(range(20000, 100000, 500))
    
    # Also include some specific values
    M_values = sorted(set(M_values))
    
    for M in M_values:
        if time.time() > deadline:
            break
        
        # Generate M-smooth numbers up to target
        smooth_nums = generate_smooth_numbers(M, target)
        
        # For each smooth a, check if b = target - a is also M-smooth
        smooth_set = set(smooth_nums)
        
        found = False
        for a in smooth_nums:
            if time.time() > deadline:
                break
            b = target - a
            if b < 1:
                continue
            if b in smooth_set:
                # Both are M-smooth, so max(P(a), P(b)) <= M
                found = True
                best_a = a
                best_b = b
                best_score = M
                break
        
        if found:
            # We found a pair with score <= M. Since we're increasing M,
            # this is the minimum M that works.
            # But let's verify the actual score
            pa = _largest_prime_factor(best_a)
            pb = _largest_prime_factor(best_b)
            actual_score = max(pa, pb)
            # The actual score might be less than M
            # We can try to do better with a more targeted search
            break
    
    # Try a direct search for even better pairs
    # Check if we can do better than current best_score
    if best_score > 1:
        # Try to find pairs where both have smaller largest prime factor
        # Direct search: for a in range, check P(a) and P(b)
        # But this is expensive. Instead, try a few more targeted M values
        # below the current best if we didn't find the true minimum
        
        # Actually, let's do a more careful search around the found M
        # Try M values between 2 and best_score
        if best_score > 2:
            lower_M_values = list(range(2, best_score))
            lower_M_values = sorted(set(lower_M_values))
            
            for M in lower_M_values:
                if time.time() > deadline:
                    break
                
                smooth_nums = generate_smooth_numbers(M, target)
                smooth_set = set(smooth_nums)
                
                for a in smooth_nums:
                    if time.time() > deadline:
                        break
                    b = target - a
                    if b < 1:
                        continue
                    if b in smooth_set:
                        pa = _largest_prime_factor(a)
                        pb = _largest_prime_factor(b)
                        actual = max(pa, pb)
                        if actual < best_score:
                            best_score = actual
                            best_a = a
                            best_b = b
                        break
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
