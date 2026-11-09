"""
Strategy: Generate powerful numbers up to 10^8, count all pair sums,
find N with the most representations, and return the top pairs.
"""
import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Generate all powerful numbers up to LIMIT
    # A powerful number has the form prod p_i^{e_i} where each e_i is 0 or >= 2
    LIMIT = 10**8

    def generate_powerful(limit):
        """Generate all powerful numbers up to limit."""
        powerful = set()
        powerful.add(1)
        
        # Primes up to isqrt(limit)
        # For p^2 > limit, p^2 itself is too big, but p^2 * (other factors) could still work
        # Actually, if p^2 > limit, then the only powerful number with prime p is p^2 itself (if <= limit)
        # So we need primes up to sqrt(limit) for the "squared" part
        # But also, p^2 * q^2 * ... could exceed limit quickly
        
        # A powerful number is of the form: prod p_i^{e_i}, e_i >= 2 for primes that appear
        # Let's generate by recursively combining prime powers
        # For each prime p, the exponent e can be 2, 3, 4, ... such that p^e <= limit
        # But we also need to multiply by other prime powers
        
        # Better approach: generate all prime powers p^e where e >= 2 and p^e <= limit
        # Then combine them (taking distinct primes) such that the product <= limit
        
        # First, collect all prime powers p^e (e >= 2, p^e <= limit)
        # For a fixed p, max e is floor(log_p(limit))
        
        # We need primes up to sqrt(limit) = 10000 for p^2 <= limit
        # For p > sqrt(limit), p^2 > limit, so no prime power p^e (e>=2) exists for p > 10000
        
        # Actually, for p > limit^(1/2), p^2 > limit, so the only way p appears is with e >= 2,
        # meaning p^e >= p^2 > limit. So p cannot appear in any powerful number <= limit.
        # Thus we only need primes up to isqrt(limit).
        
        max_prime = isqrt(limit)
        
        # Sieve of Eratosthenes up to max_prime
        is_prime = [True] * (max_prime + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(max_prime) + 1):
            if is_prime[i]:
                for j in range(i * i, max_prime + 1, i):
                    is_prime[j] = False
        primes = [i for i in range(2, max_prime + 1) if is_prime[i]]
        
        # For each prime, generate all p^e for e >= 2, p^e <= limit
        # Then do a DFS/backtracking to combine distinct prime powers
        
        # Precompute prime power options for each prime
        prime_powers = {}
        for p in primes:
            powers = []
            val = p * p
            while val <= limit:
                powers.append(val)
                val *= p
            if powers:
                prime_powers[p] = powers
        
        # Now generate powerful numbers by multiplying combinations of these prime powers
        # Each prime can contribute at most one "block" (p^e for some e >= 2)
        # This is a subset product problem with multiple choices per prime
        
        # DFS over primes, maintaining current product
        powerful_list = [1]
        
        def dfs(idx, current):
            if idx == len(primes):
                return
            p = primes[idx]
            if p not in prime_powers:
                dfs(idx + 1, current)
                return
            
            # Option 1: don't include this prime
            dfs(idx + 1, current)
            
            # Option 2: include p^e for each valid e
            for pe in prime_powers[p]:
                new_val = current * pe
                if new_val > limit:
                    break
                powerful_list.append(new_val)
                dfs(idx + 1, new_val)
        
        dfs(0, 1)
        
        powerful_set = set(powerful_list)
        powerful_set.add(1)  # 1 is powerful
        return sorted(powerful_set)
    
    powerful = generate_powerful(LIMIT)
    powerful_set = set(powerful)
    n_powerful = len(powerful)
    print(f"Generated {n_powerful} powerful numbers up to {LIMIT}")
    
    # Count representations: for each pair (a, b) with a <= b, a+b <= 2*LIMIT, a+b = N
    # We want to find N with the most pairs
    
    # Use a dictionary to count: count[N] = list of pairs
    # But storing all pairs might be memory-intensive. Let's first count, then retrieve.
    
    # Count first
    counts = defaultdict(int)
    for i in range(n_powerful):
        a = powerful[i]
        # b >= a, so b starts from a
        for j in range(i, n_powerful):
            b = powerful[j]
            s = a + b
            if s > 2 * LIMIT:
                break
            counts[s] += 1
    
    if not counts:
        # Fallback: return a trivial result
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find N with maximum count
    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    print(f"Best N = {best_N}, count = {best_count}")
    
    # Retrieve the actual pairs for best_N
    pairs = []
    for i in range(n_powerful):
        a = powerful[i]
        b = best_N - a
        if a <= b and b in powerful_set:
            pairs.append([a, b])
    
    # Deduplicate and sort
    pairs = sorted(set(tuple(p) for p in pairs))
    pairs = [[a, b] for a, b in pairs]
    
    # Check time
    if time.time() > deadline:
        # Still return what we have
        pass
    
    return {'N': best_N, 'pairs': pairs}
