import time
import heapq

def construct():
    deadline = time.time() + 85.0
    TARGET = 123457
    
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
    
    def generate_smooth_numbers(primes, limit):
        """Generate all numbers <= limit whose prime factors are all in primes."""
        if not primes:
            return set()
        result = {1}
        # Use iterative approach: for each prime, multiply existing smooth numbers
        # Actually, let's use a different approach: BFS/DFS
        # Start with 1, then for each prime p, multiply all current numbers by p^k
        smooth = set([1])
        for p in primes:
            new_smooth = set()
            for x in smooth:
                y = x
                while y <= limit:
                    new_smooth.add(y)
                    y *= p
            smooth = new_smooth
        return {x for x in smooth if x <= limit}
    
    # Try increasing sets of primes
    # Start with just {2}, then {2,3}, {2,3,5}, etc.
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    
    best_result = None
    best_P = float('inf')
    
    for num_primes in range(1, len(all_primes) + 1):
        if time.time() > deadline - 5:
            break
        primes = all_primes[:num_primes]
        max_P = primes[-1]
        
        # Generate all smooth numbers
        smooth = generate_smooth_numbers(primes, TARGET)
        
        # Check if any two smooth numbers sum to TARGET
        found = False
        for a in smooth:
            if time.time() > deadline - 3:
                break
            b = TARGET - a
            if b < 1:
                continue
            if b in smooth:
                # Found a valid pair
                return [a, b]
        found = True  # We went through all
    
    # Fallback: return something reasonable
    if best_result:
        return best_result
    
    # Last resort: use trial division to find a good pair
    # Try to find a where both a and TARGET-a have small largest prime factor
    # For now, return [1, TARGET-1] as fallback
    return [1, TARGET - 1]
