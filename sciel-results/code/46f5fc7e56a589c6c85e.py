import time
import random
import numpy as np

def construct():
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    # Use multiple large primes for modular hashing
    PRIMES = [1000000007, 1000000009, 1000000033, 1000000087, 1000000093]
    NUM_PRIMES = len(PRIMES)
    
    def mod_inverse(a, m):
        """Compute modular inverse of a mod m using Fermat's little theorem."""
        return pow(a, m - 2, m)
    
    def try_greedy_modular(order):
        """
        Greedily build a set from 'order' using modular hashing.
        Returns the selected subset.
        """
        current_set = []
        # Use lists of booleans for fast membership testing
        sums_mod = [set() for _ in range(NUM_PRIMES)]
        for p_idx in range(NUM_PRIMES):
            sums_mod[p_idx].add(0)
        
        for x in order:
            # Compute 1/x mod p for each prime
            inv_x = [mod_inverse(x, p) for p in PRIMES]
            
            # Check for collisions: new sums must not intersect old sums
            valid = True
            for p_idx in range(NUM_PRIMES):
                p = PRIMES[p_idx]
                w = inv_x[p_idx]
                old_sums = sums_mod[p_idx]
                
                # Check if any (s + w) % p is already in old_sums
                for s in old_sums:
                    if (s + w) % p in old_sums:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Update sums for each prime
                for p_idx in range(NUM_PRIMES):
                    p = PRIMES[p_idx]
                    w = inv_x[p_idx]
                    old_sums = sums_mod[p_idx]
                    new_sums = set()
                    for s in old_sums:
                        new_sums.add((s + w) % p)
                    sums_mod[p_idx] = old_sums | new_sums
                current_set.append(x)
        
        return current_set
    
    def verify_exact(subset):
        """
        Verify that all reciprocal subset sums are distinct using exact rational arithmetic.
        Returns True if valid, False otherwise.
        """
        if not subset:
            return True
        
        from fractions import Fraction
        subset_sums = {Fraction(0, 1)}
        
        for x in subset:
            new_sums = set()
            for s in subset_sums:
                new_sums.add(s + Fraction(1, x))
            subset_sums = subset_sums | new_sums
        
        # All sums must be distinct
        return len(subset_sums) == (1 << len(subset))
    
    best = []
    
    # 1. Deterministic orderings - try these first as they're fast
    base_orders = [
        list(range(1, 23)),       # ascending
        list(range(22, 0, -1)),   # descending
        # Powers of 2 first
        [1, 2, 4, 8, 16] + [x for x in range(1, 23) if x not in [1, 2, 4, 8, 16]],
        # Primes first
        [2, 3, 5, 7, 11, 13, 17, 19] + [x for x in range(1, 23) if x not in [2, 3, 5, 7, 11, 13, 17, 19]],
        # Mix: alternate small and large
        [1, 22, 2, 21, 3, 20, 4, 19, 5, 18, 6, 17, 7, 16, 8, 15, 9, 14, 10, 13, 11, 12],
    ]
    
    for order in base_orders:
        if time.time() > deadline - 30.0:
            break
        res = try_greedy_modular(order)
        if len(res) > len(best):
            if verify_exact(res):
                best = res
    
    # 2. Randomized greedy - main search loop
    random.seed(42)
    powers_of_2 = [1, 2, 4, 8, 16]
    
    while time.time() < deadline - 15.0:
        order = list(range(1, 23))
        random.shuffle(order)
        
        # Sometimes put powers of 2 first
        if random.random() < 0.3:
            rest = [x for x in order if x not in powers_of_2]
            order = powers_of_2 + rest
        
        res = try_greedy_modular(order)
        if len(res) > len(best):
            if verify_exact(res):
                best = res
    
    # 3. Local search improvement
    if time.time() < deadline - 5.0:
        best_set = set(best)
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for x in list(best):
                if time.time() > deadline - 3.0:
                    break
                
                # Remove x from best
                candidate = [y for y in best if y != x]
                candidate_set = set(candidate)
                
                # Try to add remaining elements
                remaining = [y for y in range(1, 23) if y not in candidate_set]
                random.shuffle(remaining)
                
                current = list(candidate)
                # Build modular sums for candidate
                sums_mod = [set() for _ in range(NUM_PRIMES)]
                for p_idx in range(NUM_PRIMES):
                    sums_mod[p_idx].add(0)
                
                for y in candidate:
                    inv_y = [mod_inverse(y, p) for p in PRIMES]
                    for p_idx in range(NUM_PRIMES):
                        p = PRIMES[p_idx]
                        w = inv_y[p_idx]
                        old_sums = sums_mod[p_idx]
                        new_sums = set()
                        for s in old_sums:
                            new_sums.add((s + w) % p)
                        sums_mod[p_idx] = old_sums | new_sums
                
                for y in remaining:
                    if time.time() > deadline - 3.0:
                        break
                    
                    inv_y = [mod_inverse(y, p) for p in PRIMES]
                    valid = True
                    for p_idx in range(NUM_PRIMES):
                        p = PRIMES[p_idx]
                        w = inv_y[p_idx]
                        old_sums = sums_mod[p_idx]
                        for s in old_sums:
                            if (s + w) % p in old_sums:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        for p_idx in range(NUM_PRIMES):
                            p = PRIMES[p_idx]
                            w = inv_y[p_idx]
                            old_sums = sums_mod[p_idx]
                            new_sums = set()
                            for s in old_sums:
                                new_sums.add((s + w) % p)
                            sums_mod[p_idx] = old_sums | new_sums
                        current.append(y)
                
                if len(current) > len(best):
                    if verify_exact(current):
                        best = current
                        improved = True
                        break
    
    # 4. Final round of random restarts if time permits
    if time.time() < deadline - 2.0:
        for _ in range(200):
            if time.time() > deadline - 1.0:
                break
            order = list(range(1, 23))
            random.shuffle(order)
            res = try_greedy_modular(order)
            if len(res) > len(best):
                if verify_exact(res):
                    best = res
    
    if best:
        return best
    
    # Fallback: return a known valid set
    return [1, 2, 4, 8, 16]
