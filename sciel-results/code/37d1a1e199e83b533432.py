import math

def construct():
        N = 123457
        
        # Generate primes up to N using simple sieve (fast enough)
        is_prime = [True] * (N + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(N)) + 1):
            if is_prime[i]:
                for j in range(i*i, N + 1, i):
                    is_prime[j] = False
        primes = [i for i in range(2, N + 1) if is_prime[i]]
        
        # We want to find the minimum B such that there exist a, b 
        # with a+b=N and P(a)<=B, P(b)<=B.
        # We iterate B through the primes.
        # Why primes? Because P(x) is always a prime (or 1).
        # So the max value will be a prime.
        
        for idx, B in enumerate(primes):
            # Get primes <= B
            primes_le_B = primes[:idx+1]
            
            # Generate all B-smooth numbers <= N
            smooths = {1}
            # We can generate them by multiplying existing smooths by primes in primes_le_B
            # But we need to be careful not to exceed N.
            # A standard way is to use a priority queue or just recursive generation.
            # Given N is small, a simple recursive generation with memoization or set expansion works.
            
            # Let's use a list and expand it.
            # Start with [1]
            # For each prime p in primes_le_B:
            #   For each existing number x in smooths:
            #       while x * p <= N:
            #           add x * p to smooths
            #           x *= p
            #           # But this misses combinations like 2*3, 2*3*3, etc. if we do it sequentially incorrectly.
            #           
            # Better: 
            # Initialize S = {1}
            # For each p in primes_le_B:
            #   NewS = S
            #   for x in S:
            #       y = x
            #       while y * p <= N:
            #           y *= p
            #           NewS.add(y)
            #   S = NewS
            # This generates all numbers formed by primes in primes_le_B.
            
            S = {1}
            for p in primes_le_B:
                new_S = set(S)
                for x in S:
                    y = x
                    while y * p <= N:
                        y *= p
                        new_S.add(y)
                S = new_S
            
            # Now check if any a in S has N-a in S
            found = False
            best_pair = None
            for a in S:
                if a > N: continue
                b = N - a
                if b < 1: continue
                if b in S:
                    # We found a pair. 
                    # The cost is max(P(a), P(b)). Since a,b are B-smooth, cost <= B.
                    # Is it exactly B? Not necessarily, but B is the upper bound we are testing.
                    # We want to minimize the actual max(P(a), P(b)).
                    # Since we iterate B in increasing order, the first B for which a pair exists
                    # might not yield the minimal max(P(a), P(b)) if the pair has factors < B.
                    # However, if a pair exists with factors < B, it would have been found in a previous iteration.
                    # So the first B that yields a pair ensures that the minimal possible max factor is <= B.
                    # But wait. If we find a pair at B=10, but the actual max factor is 7, 
                    # it would have been found at B=7.
                    # So the first B that works gives the correct minimal upper bound.
                    # We should return the pair.
                    return [a, b]
                    
            # If no pair found, continue to next B
            # Note: S grows. We can optimize by checking if len(S) is too large, but N is small.
            
        # Fallback: should not reach here
        return [1, N-1]
