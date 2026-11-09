import time
import sys

def _get_square_free_numbers():
    nums = []
    for i in range(2, 41):
        is_sq_free = True
        n = i
        for p in [2, 3, 5, 7]:
            if p * p > n:
                break
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            if count >= 2:
                is_sq_free = False
                break
        if is_sq_free:
            nums.append(i)
    return nums

def _get_exponent_vector(n, primes):
    vec = []
    for p in primes:
        count = 0
        while n % p == 0:
            n //= p
            count += 1
        vec.append(count)
    return tuple(vec)

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    square_free_nums = _get_square_free_numbers()
    
    # Map numbers to exponent vectors
    candidates = []
    for n in square_free_nums:
        vec = _get_exponent_vector(n, primes)
        candidates.append((n, vec))
        
    # Sort candidates: 
    # Heuristic: Numbers with fewer prime factors are "less dense" in the lattice?
    # Actually, let's just sort by value to be deterministic.
    # Or sort by number of non-zero entries in vec (number of prime factors).
    # Primes (1 factor) are easiest to add.
    candidates.sort(key=lambda x: (sum(1 for v in x[1] if v > 0), x[0]))
    
    best_set = []
    best_size = 0
    
    def search(idx, current_set, subset_sums):
        nonlocal best_set, best_size
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning: even if we take all remaining, we can't beat best
        if len(current_set) + (len(candidates) - idx) <= best_size:
            return
            
        if idx == len(candidates):
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
            
        n, vec = candidates[idx]
        
        # Option 1: Include n
        # Check for collisions
        # new_sums = {s + vec for s in subset_sums}
        # Collision if any (s + vec) is in subset_sums
        # We can do this efficiently
        collision = False
        new_sums = set()
        for s in subset_sums:
            new_s = tuple(s[i] + vec[i] for i in range(len(primes)))
            if new_s in subset_sums:
                collision = True
                break
            new_sums.add(new_s)
            
        if not collision:
            new_subset_sums = subset_sums | new_sums
            current_set.append(n)
            search(idx + 1, current_set, new_subset_sums)
            current_set.pop()
            
        # Option 2: Exclude n
        search(idx + 1, current_set, subset_sums)

    # Initial state: empty set, subset sums = {zero vector}
    zero_vec = tuple(0 for _ in primes)
    search(0, [], {zero_vec})
    
    if not best_set:
        # Fallback: return primes
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    return [int(x) for x in best_set]
