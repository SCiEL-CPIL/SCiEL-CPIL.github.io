import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = math.factorial(12)
    
    # Generate all divisors of N
    divisors = []
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    
    # Filter divisors > 12
    valid_divisors = [d for d in divisors if d > 12]
    # Sort them? 
    # If we sort ascending, we try small factors first.
    # This might lead to many factors, keeping max low?
    # Or we can sort descending to pick large chunks?
    # Let's try sorting ascending.
    valid_divisors.sort()
    
    # Precompute primality for numbers up to N? 
    # N is ~4.8e8. Sieve is too big.
    # Simple trial division for primality check in DFS.
    # Or just check if remainder is in divisors and has no other divisors?
    # Actually, if remainder is prime, it must be in valid_divisors (if > 12).
    # If remainder is prime and > 12, it's in valid_divisors.
    # If remainder is prime and <= 12, it's not in valid_divisors.
    
    # To check if a number is prime quickly:
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    # We can optimize is_prime by checking against small primes first.
    # But trial division up to sqrt(4.8e8) ~ 22000 is fast enough for a few calls.
    # However, DFS might call it many times.
    # Better: If remainder is not in valid_divisors, it's composite or <= 12.
    # If remainder > 12 and not in valid_divisors, it's composite.
    # If remainder > 12 and in valid_divisors, it *could* be prime.
    # We need to distinguish prime from composite to apply the "must take as last factor" rule.
    
    best_max = N
    best_factors = [N]
    
    def dfs(remainder, used, current_max, count):
        nonlocal best_max, best_factors
        
        if time.time() > deadline:
            return False
            
        if remainder == 1:
            if current_max < best_max:
                best_max = current_max
                best_factors = list(used)
            return True
            
        if count >= 12:
            return False
            
        if current_max >= best_max:
            return False
            
        # Check if remainder is prime
        # Optimization: If remainder is in valid_divisors, it might be prime.
        # If remainder is NOT in valid_divisors, it's composite (or <= 12, but we handle > 12).
        # Actually, if remainder > 12 and remainder not in valid_divisors, it's composite.
        # If remainder > 12 and remainder in valid_divisors, check if prime.
        
        if remainder > 12 and remainder in valid_divisors_set:
            if is_prime(remainder):
                # Must take remainder as the last factor
                if remainder in used:
                    return False
                new_max = max(current_max, remainder)
                if new_max < best_max:
                    used.add(remainder)
                    dfs(1, used, new_max, count + 1)
                    used.remove(remainder)
                return True # Only one path for prime remainder
                
        # Try all valid divisors of remainder
        # We need divisors of `remainder` that are in `valid_divisors`
        # and not in `used`
        
        # To make this fast, we can iterate through valid_divisors and check if remainder % d == 0
        # But valid_divisors has ~700 elements.
        # 700 checks per node is fine.
        
        # Heuristic: Sort candidate divisors?
        # We want to find a solution with small max.
        # Picking d close to sqrt(remainder) balances the factors.
        # Let's collect candidates
        candidates = []
        for d in valid_divisors:
            if d in used:
                continue
            if remainder % d == 0:
                # Additional pruning: 
                # The remaining product is remainder // d
                # If remainder // d is 1, it's handled by the base case or prime logic?
                # If remainder // d == 1, then d == remainder.
                # If d == remainder, we are taking the whole remainder as one factor.
                # This is valid if d > 12 and not used.
                candidates.append(d)
                
        if not candidates:
            return False
            
        # Sort candidates to explore promising paths first
        # To minimize max, we might want to pick d such that the remaining factors are small.
        # This usually means d should not be too small (leaves huge remainder) 
        # and not too large (might be the max, but if it's small, good).
        # Actually, if we pick a small d, the max is determined by the later factors.
        # If we pick a large d, it might be the max.
        # Let's try sorting by d? 
        # Or by the resulting max? 
        # Let's just iterate. The pruning `current_max < best_max` will kick in.
        
        for d in candidates:
            new_max = max(current_max, d)
            if new_max >= best_max:
                continue
                
            used.add(d)
            dfs(remainder // d, used, new_max, count + 1)
            used.remove(d)
            
            # If we found a solution with max 13 (theoretical min), we can stop
            if best_max == 13:
                return True
                
    valid_divisors_set = set(valid_divisors)
    
    # Start DFS
    # We can start with an empty set
    dfs(N, set(), 0, 0)
    
    return best_factors
