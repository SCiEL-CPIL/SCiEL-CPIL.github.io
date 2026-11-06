"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Precompute primes up to N+1
_ALL_PRIMES = _primes(N + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_result = None
    best_size = -1
    
    def run_greedy(priority='large', seed=42):
        rng = random.Random(seed)
        points = set(range(N + 1))
        points_arr = np.array(list(points))
        removed = []
        
        # Initialize counts using NumPy
        counts = {}
        for p in _ALL_PRIMES:
            counts[p] = np.bincount(points_arr % p, minlength=p)
        
        current_size = len(points)
        
        while True:
            if time.time() > deadline - 10:
                break
            
            # Find violating primes
            violating_primes = []
            for p in _ALL_PRIMES:
                if p > current_size:
                    break
                if np.min(counts[p]) > 0:
                    violating_primes.append(p)
            
            if not violating_primes:
                break
            
            # Select prime
            if priority == 'large':
                p = max(violating_primes)
            else:
                p = min(violating_primes)
            
            # Select residue
            min_count = np.min(counts[p])
            candidates = np.where(counts[p] == min_count)[0]
            r = rng.choice(candidates.tolist())
            
            # Remove elements
            to_remove_mask = (points_arr % p == r)
            to_remove_indices = np.where(to_remove_mask)[0]
            
            if len(to_remove_indices) == 0:
                break
            
            to_remove_vals = points_arr[to_remove_indices]
            
            # Update points set
            for x in to_remove_vals:
                points.remove(int(x))
                removed.append(int(x))
            
            # Update counts
            for q in _ALL_PRIMES:
                counts[q] -= np.bincount(to_remove_vals % q, minlength=q)
            
            # Update points_arr
            points_arr = np.array(list(points))
            
            current_size -= len(to_remove_vals)
            
        # Add back phase
        rng.shuffle(removed)
        
        for x in removed:
            if time.time() > deadline - 5:
                break
            
            valid = True
            new_size = current_size + 1
            x_arr = np.array([x])
            
            for q in _ALL_PRIMES:
                if q > new_size:
                    break
                
                r_x = x % q
                if counts[q][r_x] == 0:
                    # Check if this is the only zero
                    if np.count_nonzero(counts[q] == 0) == 1:
                        valid = False
                        break
            
            if valid:
                points.add(x)
                current_size += 1
                # Update counts
                for q in _ALL_PRIMES:
                    counts[q] += np.bincount(x_arr % q, minlength=q)
        
        return list(points)

    strategies = [
        ('large', 42),
        ('small', 42),
        ('large', 123),
        ('small', 123),
        ('large', 999),
        ('small', 999),
    ]
    
    for i, (prio, seed) in enumerate(strategies):
        if time.time() > deadline - 30:
            break
        
        result = run_greedy(prio, seed)
        size = len(result)
        
        if size > best_size:
            best_size = size
            best_result = result

    if best_result is None:
        return [1]
        
    return best_result
# EVOLVE-BLOCK-END
