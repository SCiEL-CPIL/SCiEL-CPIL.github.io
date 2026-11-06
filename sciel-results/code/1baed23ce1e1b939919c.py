import time
import heapq

def construct():
    deadline = time.time() + 85.0
    TARGET = 123457
    
    def largest_prime_factor(n):
        """Compute the largest prime factor of n."""
        if n <= 1:
            return 1
        lp = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                lp = d
                n //= d
            d += 1
        if n > 1:
            lp = max(lp, n)
        return lp
    
    def generate_smooth_numbers(primes, limit, max_count=200000):
        """Generate all numbers <= limit whose prime factors are all in primes."""
        if not primes:
            return {1}
        
        result = {1}
        for p in primes:
            new_vals = set()
            for x in result:
                y = x
                while y <= limit:
                    new_vals.add(y)
                    if y > limit // p:
                        break
                    y *= p
            result = new_vals
            if len(result) > max_count:
                break
        return result
    
    # Try increasing sets of primes
    # Start with just {2}, then {2,3}, {2,3,5}, etc.
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    
    best_result = [1, TARGET - 1]
    best_P = largest_prime_factor(TARGET - 1)
    
    for num_primes in range(1, len(all_primes) + 1):
        if time.time() > deadline - 15:
            break
        primes = all_primes[:num_primes]
        max_P = primes[-1]
        
        # Generate all smooth numbers
        smooth = generate_smooth_numbers(primes, TARGET)
        
        # Check if any two smooth numbers sum to TARGET
        # Use a set for O(1) lookup
        for a in smooth:
            if time.time() > deadline - 10:
                break
            b = TARGET - a
            if b < 1:
                continue
            if b in smooth:
                # Found a valid pair with max prime factor = max_P
                return [a, b]
    
    # Fallback: search for better pairs using trial division
    for a in range(1, TARGET // 2 + 1):
        if time.time() > deadline - 5:
            break
        b = TARGET - a
        pa = largest_prime_factor(a)
        pb = largest_prime_factor(b)
        maxp = max(pa, pb)
        if maxp < best_P:
            best_P = maxp
            best_result = [a, b]
            if best_P <= 2:
                return best_result
    
    return best_result
