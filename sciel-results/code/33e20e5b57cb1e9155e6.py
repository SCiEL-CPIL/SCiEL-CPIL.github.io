def construct():
    N = 123457
    
    # Sieve primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N+1) if is_prime[i]]
    
    # S contains all p-smooth numbers for the current prime p
    # Start with 1 (which is 1-smooth, or 0-prime smooth)
    S = {1}
    
    # We iterate through primes in increasing order.
    # After processing prime p, S contains all p-smooth numbers <= N.
    # We check if a pair exists.
    
    for p in primes:
        # Generate new smooth numbers formed by multiplying existing ones by p
        new_numbers = []
        # We must iterate over a snapshot of S or a list, because S is being updated
        # But wait, we can just iterate over the current S.
        # If we add to S while iterating, it might skip items or error.
        # So we collect new items first.
        for x in S:
            y = x * p
            if y <= N:
                if y not in S:
                    new_numbers.append(y)
            else:
                # Since S is not sorted, we can't break. 
                # But if x is large, x*p is large.
                # We can't assume S is sorted.
                pass
        
        # Add new numbers to S
        for y in new_numbers:
            S.add(y)
            
        # Check for a pair a, b in S such that a + b = N
        # We can iterate through S and check if N-a is in S.
        # To avoid duplicate work or to find the pair quickly, 
        # we can just check all a in S.
        found = False
        for a in S:
            b = N - a
            if b >= 1 and b in S:
                # We found a pair. 
                # Since we are processing primes in increasing order, 
                # the current p is the smallest prime such that a solution exists.
                # Thus max(P(a), P(b)) <= p. 
                # Actually, could max be smaller? 
                # If a solution existed with max prime < p, it would have been found in a previous iteration.
                # So this is optimal.
                return [a, b]
                
    # Fallback (should not happen as 1 and N-1 are always a solution if N-1 is smooth? 
    # No, N-1 might not be smooth for small primes. 
    # Eventually p=N-1 will make N-1 smooth. 
    # But the loop goes up to N.
    return [1, N-1]
