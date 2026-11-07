import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 256
    candidates = list(range(2, N + 1))
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    # Start with all primes
    S = set(primes)
    
    # Try to add composites in descending order
    # Larger numbers are less likely to divide products of smaller numbers
    composites.sort(reverse=True)
    
    for x in composites:
        if time.time() > deadline:
            break
            
        can_add = True
        n_S = len(S)
        S_list = list(S)
        
        # Check all pairs (a, b) in S
        for i in range(n_S):
            a = S_list[i]
            for j in range(i + 1, n_S):
                b = S_list[j]
                
                # Check 1: x | a * b
                # Since a, b, x must be distinct for the rule to apply.
                # Here x is not in S yet, so x is distinct from a and b.
                if (a * b) % x == 0:
                    can_add = False
                    break
                    
                # Check 2: a | x * b
                # a, b, x must be distinct.
                # x is not in S, so x != a and x != b.
                if (x * b) % a == 0:
                    can_add = False
                    break
                    
                # Check 3: b | x * a
                # b, a, x must be distinct.
                if (x * a) % b == 0:
                    can_add = False
                    break
                    
            if not can_add:
                break
                
        if can_add:
            S.add(x)
            
    # Convert to list of ints and sort
    result = [int(x) for x in S]
    return sorted(result)
