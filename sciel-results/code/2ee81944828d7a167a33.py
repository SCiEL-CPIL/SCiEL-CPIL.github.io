import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55

    # Primes and factorization
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    def factorize(n):
        exponents = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                exponents[i] += 1
                n //= p
        return tuple(exponents)

    # Precompute vectors for 2..40
    vectors = {}
    for n in range(2, 41):
        vectors[n] = factorize(n)
        
    zero = (0,) * num_primes

    def can_add(sums, v):
        """Check if vector v can be added to the set with existing sums."""
        new_sums = set()
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(num_primes))
            if ns in sums:
                return False
            new_sums.add(ns)
        return True

    def add_to_sums(sums, v):
        """Update sums with new vector v. Assumes can_add passed."""
        new_sums = set()
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(num_primes))
            new_sums.add(ns)
        sums.update(new_sums)

    # Try to find a set of size 15
    # Heuristic: Start with primes (size 12), then try to add composites
    # Primes are linearly independent, so they are a good start.
    base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    best_set = []
    best_size = 0

    # Strategy 1: Greedy from base primes, adding composites that don't cause collisions
    # Composites are 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 38, 39, 40
    composites = [n for n in range(2, 41) if n not in base]
    
    def solve_greedy(order):
        current = []
        sums = {zero}
        for n in order:
            v = vectors[n]
            if can_add(sums, v):
                add_to_sums(sums, v)
                current.append(n)
        return current

    # Try different orderings of composites to add to the base
    # Randomized restarts
    max_found = 0
    best_solution = []
    
    # First, ensure the base primes work (they should)
    base_sums = {zero}
    for p in base:
        v = vectors[p]
        if can_add(base_sums, v):
            add_to_sums(base_sums, v)
        else:
            # Should not happen for primes
            pass
            
    current_best = list(base)
    current_best_sums = set(base_sums)
    max_found = len(current_best)
    best_solution = list(current_best)
    
    # Try to add composites
    # We want to add 3 more to reach 15.
    
    # Randomize the order of composites
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        shuffled_comp = composites[:]
        random.shuffle(shuffled_comp)
        
        trial_set = list(base)
        trial_sums = set(base_sums)
        
        for n in shuffled_comp:
            v = vectors[n]
            if can_add(trial_sums, v):
                add_to_sums(trial_sums, v)
                trial_set.append(n)
                
        if len(trial_set) > max_found:
            max_found = len(trial_set)
            best_solution = trial_set[:]
            
        if max_found == 15:
            break

    # If we didn't find 15, try local search or different base?
    # But 15 is the max.
    
    # Final check
    if not best_solution:
        best_solution = base
        
    return [int(x) for x in best_solution]
