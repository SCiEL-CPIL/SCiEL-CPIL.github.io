import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 256
    
    # Precompute prime factors for each number 1..256
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]: # Simple check, though 0 is empty
            # Actually, let's just compute for all
            pass
            
    # Better way to compute prime factors
    for i in range(2, N + 1):
        x = i
        p = 2
        while p * p <= x:
            if x % p == 0:
                prime_factors[i].add(p)
                while x % p == 0:
                    x //= p
            else:
                p += 1
        if x > 1:
            prime_factors[i].add(x)
            
    # Initial set: 1 and the best power for each prime <= 256
    # First, identify all primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Calculate best standalone value for each prime
    best_standalone = {}
    for p in primes:
        max_val = p
        val = p
        while val * p <= N:
            val *= p
            max_val = val
        best_standalone[p] = max_val
        
    # Initialize current_set with 1 and all best_standalone
    current_set = [1] + list(best_standalone.values())
    
    # Map each prime to the number in current_set that contains it
    # Since current_set is coprime, each prime appears in at most one number
    prime_to_num = {}
    for num in current_set:
        for p in prime_factors[num]:
            prime_to_num[p] = num
            
    improved = True
    while improved:
        improved = False
        # Sort current_set to ensure deterministic behavior?
        # Iterate through all possible replacement numbers n
        for n in range(2, N + 1):
            if time.time() > deadline:
                break
                
            # If n is already in the set, skip
            if n in current_set:
                continue
                
            P_n = prime_factors[n]
            
            # Find conflicts: numbers in current_set that share a prime with n
            conflicts = []
            for num in current_set:
                if num == 1:
                    continue
                if P_n & prime_factors[num]:
                    conflicts.append(num)
            
            # If no conflicts, we can just add n?
            # But we started with all primes covered. Adding n without removing anything
            # would mean n is coprime to all. But n has prime factors.
            # Those primes must be in some number in current_set.
            # So conflicts cannot be empty unless n=1 (which is skipped) or n has no prime factors (impossible for n>1).
            # Wait, if n is coprime to all, its prime factors are not in current_set.
            # But we initialized with all primes. So this case shouldn't happen unless we removed some primes.
            
            if not conflicts:
                # This implies n's prime factors are not in the set.
                # We can add n.
                current_set.append(n)
                for p in P_n:
                    prime_to_num[p] = n
                improved = True
                continue # Restart loop to re-evaluate
            
            # Calculate gain
            sum_conflicts = sum(conflicts)
            gain = n - sum_conflicts
            
            if gain > 0:
                # Remove conflicts
                for c in conflicts:
                    current_set.remove(c)
                    for p in prime_factors[c]:
                        # Remove mapping if c was the owner
                        # Since set is coprime, c is the only owner of its primes
                        if prime_to_num.get(p) == c:
                            del prime_to_num[p]
                
                # Add n
                current_set.append(n)
                for p in P_n:
                    prime_to_num[p] = n
                    
                improved = True
                break # Restart loop
                
        if time.time() > deadline:
            break
            
    # Convert to list of ints
    return [int(x) for x in current_set]
