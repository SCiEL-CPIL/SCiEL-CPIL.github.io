import time
import random
from collections import Counter

# Precompute prime factors for all numbers up to 20000
MAX_SUM = 20000
prime_factors = [set() for _ in range(MAX_SUM + 1)]
for i in range(2, MAX_SUM + 1):
    if not prime_factors[i]:
        for j in range(i, MAX_SUM + 1, i):
            prime_factors[j].add(i)

def construct():
    deadline = time.time() + 85.0
    rng = random.Random(42)
    n = 12
    
    # We want to minimize the number of distinct primes in the union of factors of all pairwise sums.
    
    def get_score(ks):
        """Calculate the number of distinct prime factors for all pairwise sums."""
        all_primes = set()
        for i in range(n):
            for j in range(i + 1, n):
                s = ks[i] + ks[j]
                if s <= MAX_SUM:
                    all_primes |= prime_factors[s]
                else:
                    # Fallback for safety, though 10000+10000=20000 is max
                    # Factor manually if out of range (should not happen)
                    x = s
                    f = 2
                    while f * f <= x:
                        if x % f == 0:
                            all_primes.add(f)
                            while x % f == 0:
                                x //= f
                        f += 1 if f == 2 else 2
                    if x > 1:
                        all_primes.add(x)
        return len(all_primes)
    
    def get_prime_set(ks):
        all_primes = set()
        for i in range(n):
            for j in range(i + 1, n):
                s = ks[i] + ks[j]
                if s <= MAX_SUM:
                    all_primes |= prime_factors[s]
        return all_primes

    best_ks = None
    best_score = float('inf')
    
    # Strategy 1: Try to find a set where all sums are 7-smooth (primes 2,3,5,7)
    # This would give a score of 4.
    # We can do a targeted search for this.
    # But local search with the general objective will find it if it exists.
    
    # Initial candidates:
    # 1. Small integers
    # 2. Multiples of small smooth numbers
    # 3. Random
    
    initial_strategies = []
    
    # Small integers 1..12
    initial_strategies.append(list(range(1, 13)))
    
    # Multiples of 6 (forces 2,3 into sums)
    initial_strategies.append([6 * i for i in range(1, 13)])
    
    # Multiples of 30 (forces 2,3,5 into sums)
    initial_strategies.append([30 * i for i in range(1, 13)])
    
    # Random small
    for _ in range(5):
        ks = []
        while len(ks) < n:
            v = rng.randint(1, 100)
            if v not in ks:
                ks.append(v)
        initial_strategies.append(ks)
        
    # Random larger
    for _ in range(5):
        ks = []
        while len(ks) < n:
            v = rng.randint(1, 10000)
            if v not in ks:
                ks.append(v)
        initial_strategies.append(ks)

    # Local Search Function
    def local_search(ks, deadline, rng, max_iters=50000):
        current = list(ks)
        current_primes = get_prime_set(current)
        current_score = len(current_primes)
        best_local_ks = list(current)
        best_local_score = current_score
        
        # Simulated Annealing parameters
        temp = 1.0
        cooling = 0.99
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick random index to change
            idx = rng.randint(0, n - 1)
            old_val = current[idx]
            
            # Generate new value
            # Bias towards small values or values close to existing ones
            if rng.random() < 0.5:
                # Small random
                new_val = rng.randint(1, 200)
            else:
                # Perturb existing
                delta = rng.randint(-50, 50)
                new_val = old_val + delta
            
            if new_val < 1 or new_val > 10000:
                continue
            if new_val == old_val:
                continue
            if new_val in current:
                continue
                
            # Calculate delta in primes
            # Remove primes from pairs (old_val, current[j])
            # Add primes from pairs (new_val, current[j])
            
            # To be efficient, we can just recalculate the full set if it's fast enough.
            # n=12, 66 pairs. Precomputed factors make this very fast.
            
            # Temporarily update
            current[idx] = new_val
            new_primes = get_prime_set(current)
            new_score = len(new_primes)
            
            # Decide to accept
            if new_score < current_score:
                accept = True
            elif new_score == current_score:
                accept = True
            else:
                # Accept worse solutions with probability exp(-delta / temp)
                delta = new_score - current_score
                if rng.random() < pow(2.71828, -delta / temp):
                    accept = True
                else:
                    accept = False
            
            if accept:
                current_primes = new_primes
                current_score = new_score
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local_ks = list(current)
            else:
                current[idx] = old_val
                
            temp *= cooling
            
        return best_local_ks, best_local_score

    # Run local search from multiple starts
    for start_ks in initial_strategies:
        if time.time() > deadline:
            break
        # Ensure distinct and valid
        if len(set(start_ks)) != n or any(x < 1 or x > 10000 for x in start_ks):
            continue
            
        res_ks, res_score = local_search(start_ks, deadline, rng, max_iters=20000)
        if res_score < best_score:
            best_score = res_score
            best_ks = res_ks
            
    # If we found a score of 4, we can stop early as it's likely optimal for small sets
    # The theoretical minimum is likely 3 or 4. 
    # Can we get 3? Primes {2,3,5}. Sums must be 5-smooth.
    # If best_score == 4, we might try to improve to 3 with more time, but 4 is very good.
    
    if best_ks is None:
        best_ks = list(range(1, 13))
        
    return best_ks

# EVOLVE-BLOCK-END
