import time
from collections import defaultdict

def construct():
    deadline = time.time() + 85.0
    TARGET = 123457
    
    # Sieve primes up to TARGET
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(TARGET)
    
    # For each prime, generate all multiples (smooth numbers) up to TARGET
    # A number is B-smooth if all its prime factors are <= B
    # We'll generate smooth numbers by BFS/DFS over prime exponents
    
    # Map: number -> largest prime factor
    smooth_map = {}  # number -> P(number)
    
    # Generate all smooth numbers up to TARGET
    # Start with 1, then multiply by primes
    # Use a set to track visited numbers
    
    # More efficient: for each prime p, for each multiple, check if it's p-smooth
    # Actually, let's generate by iterating over primes and building up
    
    # Approach: generate all numbers <= TARGET that are p-smooth for some p
    # We can do this by: for each prime p in order, generate numbers whose prime factors are all in {primes up to p}
    
    # Let's use a different approach:
    # For each number from 2 to TARGET, compute its largest prime factor
    # Then sort by largest prime factor and search for pairs
    
    # Compute largest prime factor for all numbers up to TARGET
    # Using a modified sieve
    lp = [0] * (TARGET + 1)  # lp[n] = largest prime factor of n
    lp[1] = 1
    for i in range(2, TARGET + 1):
        if lp[i] == 0:  # i is prime
            for j in range(i, TARGET + 1, i):
                lp[j] = i  # i is a prime factor of j, and since we go in order, this will be the largest
    
    # Now lp[n] gives the largest prime factor of n for all n
    # We want to find a, b with a+b=TARGET minimizing max(lp[a], lp[b])
    
    # Sort numbers by their largest prime factor
    # For each possible value of max P, check if there's a valid pair
    
    # Better: iterate a from 1 to TARGET-1, compute score = max(lp[a], lp[TARGET-a])
    # Track the minimum
    
    best_a = 1
    best_score = lp[TARGET - 1]  # max(lp[1], lp[TARGET-1]) = lp[TARGET-1]
    
    # We can be smarter: only consider a where lp[a] is small
    # But let's just iterate all and track best
    
    # Actually, let's be more strategic. 
    # Sort candidate a values by lp[a] ascending
    # For small lp[a], check if lp[TARGET-a] is also small
    
    # Create list of (lp_value, number) for all numbers
    candidates = [(lp[i], i) for i in range(1, TARGET)]
    candidates.sort()
    
    # For each candidate a (in order of increasing lp[a]), 
    # check if lp[TARGET-a] < current best
    # Stop when lp[a] >= best_score (since max(lp[a], lp[TARGET-a]) >= lp[a])
    
    for lp_val, a in candidates:
        if lp_val >= best_score:
            break
        b = TARGET - a
        if b < 1:
            break
        b_lp = lp[b]
        score = max(lp_val, b_lp)
        if score < best_score:
            best_score = score
            best_a = a
    
    a = best_a
    b = TARGET - best_a
    
    return [a, b]
