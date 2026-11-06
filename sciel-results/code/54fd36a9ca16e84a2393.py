import heapq
import math

def construct():
    N = 123457
    
    # Sieve of Eratosthenes to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
                
    # Iterate through primes as candidate max prime factor K
    # We only need to check K up to N
    # But we expect small K
    
    for K in primes:
        # Generate all K-smooth numbers <= N
        # K-smooth means all prime factors are in primes[:idx+1]
        # We need to find the index of K in primes
        # Since we are iterating, we can maintain a list of active primes
        
        # Actually, let's just use the list of primes <= K
        # We can find the index by binary search or just iterate
        # Since we are iterating K in order, we can keep a list
        pass

    # Let's restructure to keep active primes
    active_primes = []
    for K in primes:
        active_primes.append(K)
        
        # Generate K-smooth numbers
        # Use heap
        h = [1]
        heapq.heapify(h)
        generated = set()
        smooths = []
        
        while h:
            x = heapq.heappop(h)
            if x > N:
                break
            if x in generated:
                continue
            generated.add(x)
            smooths.append(x)
            
            for p in active_primes:
                nx = x * p
                if nx <= N:
                    heapq.heappush(h, nx)
                else:
                    # Since active_primes is sorted, if x*p > N, larger p will also be > N
                    # But wait, x is fixed. p increases.
                    # So we can break?
                    # Yes, if x*p > N, then x*q > N for all q > p.
                    break
        
        # Check if there exists a in smooths such that N-a in smooths
        # Convert smooths to set for O(1) lookup
        smooth_set = set(smooths)
        
        found = False
        for a in smooths:
            b = N - a
            if b in smooth_set:
                # Ensure a, b are positive
                # a >= 1, b >= 1 is guaranteed since smooths contains 1 and max is N
                # If a=N, b=0 which is not in smooths (smooths are >= 1)
                # So b >= 1 is ensured by b in smooth_set
                return [a, b]
                
    # Fallback: should not reach here if we iterate up to N
    # But if K=N, smooths includes all numbers <= N
    # So a=1, b=N-1 will be found if N-1 is smooth (which it is, since all primes <= N)
    # Wait, if K=N, active_primes includes all primes <= N.
    # So all numbers <= N are K-smooth.
    # So a=1, b=N-1 will be found.
    return [1, N - 1]
