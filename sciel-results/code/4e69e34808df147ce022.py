import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Generate small primes up to 10^6 for factorization and base primes
    limit_small = 10**6
    sieve = [True] * (limit_small + 1)
    sieve[0:2] = [False, False]
    for i in range(2, int(math.isqrt(limit_small)) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    # Helper: Deterministic Miller-Rabin for n < 2^64
    def is_prime_mr(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Helper: Pollard-Brent factorization
    def pollard_brent(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
        y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), random.randint(1, n - 1)
        g, r, q = 1, 1, 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        return g

    def factor(n):
        factors = []
        stack = [n]
        while stack:
            x = stack.pop()
            if x == 1: continue
            if is_prime_mr(x):
                factors.append(x)
            else:
                d = pollard_brent(x)
                stack.append(d)
                stack.append(x // d)
        return factors

    def is_p_smooth(n, p):
        if n == 1: return True
        temp = n
        for q in small_primes:
            if q > p: break
            if q * q > temp: break
            while temp % q == 0:
                temp //= q
        if temp == 1: return True
        if is_prime_mr(temp): return temp <= p
        factors = factor(temp)
        return all(f <= p for f in factors)

    best_k = 0
    best_p = 3

    # Phase 1: Exhaustive search for small p
    # This is fast and establishes a baseline.
    for p in small_primes:
        if time.time() > deadline:
            break
        base = p * p
        k = 0
        for offset in range(1, 201):
            if time.time() > deadline:
                break
            if is_p_smooth(base + offset, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Constrained Combinatorial Search
    # Instead of picking p and checking, we pick the structure of the factors.
    # For a run of length k, p^2+i = \prod_{j} q_{i,j}^{e_{i,j}}
    # We require q_{i,j} <= p.
    
    # We will search for runs of length 3, 4, 5...
    # For each k, we sample random factorizations of numbers in [p^2, p^2+k]
    # that are p-smooth.
    
    # To do this efficiently, we iterate over small primes q and build up the factors.
    # This is hard to do directly. Instead, we use a heuristic:
    # Pick a random set of small primes Q.
    # Generate random p-smooth numbers N_1, ..., N_k from Q.
    # Check if N_i - N_1 is a perfect square? No, N_i = p^2 + i.
    # So N_1 = p^2 + 1, N_2 = p^2 + 2, etc.
    # N_i - N_j = i - j.
    # So we need p-smooth numbers N_1, ..., N_k such that N_i - N_j = i - j.
    
    # This is a system of equations.
    # N_1 = p^2 + 1
    # N_2 = p^2 + 2
    # ...
    
    # We can generate p-smooth numbers and check if they form an arithmetic progression with difference 1?
    # No, they are p^2+1, p^2+2... so they are consecutive integers.
    # So we are looking for k consecutive integers that are all p-smooth,
    # where p is a prime such that p^2 < N_1.
    
    # So the problem reduces to: Find k consecutive integers N_1, ..., N_k
    # such that all are p-smooth for some prime p > sqrt(N_1).
    
    # How to find such runs?
    # We can sieve for p-smooth numbers.
    # But p is unknown.
    
    # Heuristic:
    # Pick a random prime p.
    # Generate random p-smooth numbers near p^2.
    # Check if they are consecutive.
    
    # This is what the previous program did, but inefficiently.
    
    # Let's try a different angle:
    # Fix k.
    # We want N_1, ..., N_k consecutive and p-smooth.
    # This implies N_i / N_j is a ratio of p-smooth numbers.
    
    # Let's just stick to the sampling but improve the sampling.
    # The previous program sampled p and checked.
    # This program will sample the *factors*.
    
    # For a given k, we can iterate over possible "shapes" of the factorization.
    # But the space is huge.
    
    # Let's try a hybrid:
    # 1. Find the best k for p < 10^6 (done above).
    # 2. For larger p, use a segmented sieve to find p-smooth numbers in [p^2, p^2+200].
    #    But we don't know p.
    
    # Let's try to find p-smooth runs directly.
    # We can generate p-smooth numbers for a fixed p.
    # But we want to maximize k over all p.
    
    # What if we fix the *small* factors?
    # Let Q be the set of primes <= 100.
    # Generate all Q-smooth numbers up to 10^10.
    # Check for consecutive runs.
    # If we find a run N_1, ..., N_k, then p must be > sqrt(N_1).
    # And p must be a prime.
    # And all factors of N_i must be <= p.
    # Since N_i are Q-smooth, their factors are in Q.
    # So we need max(Q) <= p.
    # So if we find a run of Q-smooth numbers, and we pick p = next_prime(max(Q)),
    # then the run is valid.
    
    # This is a valid strategy!
    # We can increase the size of Q to find longer runs.
    
    # Let's implement this.
    
    def find_smooth_runs(Q, limit):
        # Find consecutive Q-smooth numbers up to limit.
        # Return list of (start, length).
        smooth = [False] * (limit + 1)
        smooth[1] = True
        for q in Q:
            for i in range(2, limit + 1, q):
                # This is not efficient for large Q.
                # Better: generate smooth numbers directly.
                pass
        
        # Direct generation of Q-smooth numbers.
        # This is the "Hamming numbers" problem.
        # We can use a heap or a DP approach.
        # For Q up to 100, the number of smooth numbers is manageable.
        
        # Let's use a set to store smooth numbers.
        # This might be too slow for limit=10^10.
        
        # Alternative:
        # We know that for large p, the density of p-smooth numbers is low.
        # So runs are rare.
        
        # Let's go back to sampling p, but use a better pre-filter.
        # The previous program used is_p_smooth_approx.
        # Let's improve it.
        
        pass

    # Let's try a different structural change:
    # Instead of random sampling, use a **genetic algorithm** on the prime p.
    # Fitness: length of the p-smooth run starting at p^2+1.
    # We start with a population of primes.
    # We evaluate their fitness.
    # We select the best, mutate them (add/subtract small offsets), and evaluate.
    
    # This is a heuristic search, but it might find better p's than random sampling.
    
    population = list(small_primes[100:200]) # Start with some medium primes
    for _ in range(1000): # 1000 generations
        if time.time() > deadline:
            break
        
        # Evaluate fitness
        fitness = []
        for p in population:
            base = p * p
            k = 0
            for offset in range(1, 201):
                if time.time() > deadline:
                    break
                if is_p_smooth(base + offset, p):
                    k += 1
                else:
                    break
            fitness.append(k)
            if k > best_k:
                best_k = k
                best_p = p
        
        # Select top 50%
        sorted_pop = sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)
        new_pop = [p for p, f in sorted_pop[:len(population)//2]]
        
        # Mutate
        while len(new_pop) < len(population):
            p = random.choice(new_pop)
            # Mutate: add/subtract a random small number
            offset = random.randint(-100, 100)
            new_p = p + offset
            if new_p > 2 and is_prime_mr(new_p):
                new_pop.append(new_p)
            else:
                new_pop.append(p) # Keep original if mutation fails
        
        population = new_pop

    return {'p': best_p, 'k': best_k}
