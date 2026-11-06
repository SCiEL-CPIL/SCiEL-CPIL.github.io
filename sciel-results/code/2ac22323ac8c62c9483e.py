import math

def construct():
    N = 123457
    
    # Sieve of Eratosthenes to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Binary search on the index of the prime bound
    # We want the smallest prime B such that a pair exists
    low = 0
    high = len(primes) - 1
    best_pair = None
    
    def find_pair_with_smooth(B_idx):
        B = primes[B_idx]
        # Generate all B-smooth numbers up to N
        smooth = set()
        
        def dfs(idx, current):
            if current > N:
                return
            smooth.add(current)
            # Try multiplying by primes from idx to B_idx
            for i in range(idx, B_idx + 1):
                p = primes[i]
                next_val = current * p
                if next_val > N:
                    break
                dfs(i, next_val)
                
        dfs(0, 1)
        
        # Check for pair
        for a in smooth:
            b = N - a
            if b in smooth:
                return (a, b)
        return None

    while low <= high:
        mid = (low + high) // 2
        pair = find_pair_with_smooth(mid)
        if pair:
            best_pair = pair
            high = mid - 1
        else:
            low = mid + 1
            
    if best_pair is None:
        # Fallback, though shouldn't happen as N itself is smooth with B=N
        # Actually if B is the largest prime, all numbers are B-smooth? No, only those with factors <= B.
        # If B is the largest prime <= N, then any number <= N has prime factors <= N, so they are all B-smooth?
        # Yes, if B >= largest prime factor of any number <= N. The largest prime factor of any number <= N is <= N.
        # So if we pick B = largest prime <= N, then all numbers 1..N are B-smooth.
        # So a pair always exists (e.g., 1 and N-1).
        best_pair = find_pair_with_smooth(high)
        
    return [best_pair[0], best_pair[1]]
