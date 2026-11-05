import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N_MAX = 1000000
    N_MIN = 10000
    LIMIT = N_MAX + 1  # We need to check n+1, so up to 1000001
    
    # Get primes up to a certain limit
    # We will increase Y, so we need primes up to, say, 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    
    # We iterate Y from 2 upwards
    # For each Y, we get the subset of primes <= Y
    # We generate Y-smooth numbers
    # We check for consecutive pairs in [N_MIN, N_MAX]
    
    # To optimize, we can cache the smooth numbers for previous Y?
    # No, as Y increases, the set grows.
    # We can maintain a set of all smooth numbers found so far?
    # If a number is Y-smooth, it is also (Y+1)-smooth.
    # So we can incrementally add numbers?
    # It's easier to just regenerate or use a list of primes and filter.
    
    # Let's try regenerating for each Y.
    # The number of Y-smooth numbers is small for small Y.
    
    current_primes = []
    for Y in range(2, 1001):
        if time.time() > deadline:
            break
            
        # Update current_primes if Y is prime
        # Actually, we just need primes <= Y
        # We can precompute prefix lists of primes
        # Or just filter the global primes list
        
        p_list = [p for p in primes if p <= Y]
        
        # Generate Y-smooth numbers up to LIMIT
        # Using a set
        smooths = {1}
        frontier = {1}
        
        while frontier:
            next_frontier = set()
            for s in frontier:
                for p in p_list:
                    ns = s * p
                    if ns > LIMIT:
                        continue
                    if ns not in smooths:
                        smooths.add(ns)
                        next_frontier.add(ns)
            frontier = next_frontier
            
        # Filter smooths to range [N_MIN, N_MAX] for n
        # We need n and n+1 to be in smooths
        # So we check if n is in smooths and n+1 is in smooths
        # And N_MIN <= n <= N_MAX
        
        # To be efficient, we can iterate through sorted smooths in range
        # Or just check each n in smooths if n >= N_MIN
        # But checking membership of n+1 is O(1)
        
        # Let's collect candidates
        candidates = []
        for n in smooths:
            if N_MIN <= n <= N_MAX:
                if (n + 1) in smooths:
                    candidates.append(n)
                    
        if candidates:
            # We found pairs with max LPF <= Y
            # Since we are iterating Y upwards, the first Y that yields candidates
            # contains the optimal solution?
            # Yes, because any pair with max LPF < Y would have been found in previous iterations.
            # So the min max LPF is <= Y.
            # Actually, we want to minimize the LPF.
            # The pair found here has LPF <= Y.
            # Is it possible that a pair with LPF < Y was missed? No, because if LPF < Y, it is Y-smooth.
            # So the set of pairs found at Y includes all pairs with LPF <= Y.
            # The "best" pair is the one with the smallest max LPF.
            # All pairs found at the *first* Y that has any candidates have max LPF <= Y.
            # But do they have the *minimum* max LPF?
            # Suppose optimal LPF is 13.
            # At Y=10, no candidates.
            # At Y=11, no candidates.
            # At Y=12, no candidates.
            # At Y=13, candidates found.
            # These candidates have LPF <= 13.
            # Could one of them have LPF 11?
            # If it had LPF 11, it would be 11-smooth, and would have been found at Y=11.
            # Since it wasn't found at Y=11, its LPF must be > 11.
            # So all candidates found at the *first* successful Y have LPF exactly Y?
            # Not exactly Y, but > Y-1. So LPF is Y (if Y is prime) or the largest prime factor is the new prime added?
            # Actually, if Y is not prime, the set of smooth numbers doesn't change from Y-1.
            # So we should only check when Y is prime?
            # If Y is not prime, Y-smooth numbers == (Y-1)-smooth numbers.
            # So we can skip non-prime Y.
            
            # Let's return the first candidate found?
            # Or the one with the smallest n?
            # The task asks to minimize the LPF.
            # All candidates in this batch have LPF <= Y.
            # And since no candidates were found for Y' < Y, all candidates have LPF > Y'.
            # So the min LPF among these candidates is the global min.
            # We can just return the first one, or the one with smallest n?
            # The score is based on the LPF value, not n.
            # Any n in this batch yields the same score (the min LPF).
            # So returning any is fine.
            
            return candidates[0]
            
    # Fallback if no solution found in loop (should not happen)
    # Run the original sieve
    lpf = [0] * (LIMIT + 1)
    for i in range(2, LIMIT + 1):
        if lpf[i] == 0:
            for j in range(i, LIMIT + 1, i):
                lpf[j] = i
                
    best_n = N_MIN
    best_m = float('inf')
    for n in range(N_MIN, N_MAX + 1):
        m = max(lpf[n], lpf[n+1])
        if m < best_m:
            best_m = m
            best_n = n
            
    return best_n
