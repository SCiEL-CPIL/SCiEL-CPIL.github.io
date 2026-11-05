import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s
    
    N = 256
    elements = list(range(1, N + 1))
    
    # Precompute conflict graph: (a,b) conflict if (a*b) % (a+b) == 0
    # For speed, precompute which pairs are "good" (non-conflicting)
    # conflict[a][b] = True if (a*b) % (a+b) == 0 (i.e., they CANNOT both be in set)
    
    # Use a more efficient representation: for each a, store set of b that conflict
    # (a*b) % (a+b) == 0 means (a+b) divides (a*b)
    # Let g = gcd(a, a+b) = gcd(a, b). Then (a+b)/g divides b.
    # So (a+b) | (a*b) iff (a+b)/gcd(a,b) | b
    
    # Actually let's just compute directly with numpy for speed
    import numpy as np
    
    a_arr = np.arange(1, N + 1, dtype=np.int64)
    # Compute conflict matrix: C[i,j] = 1 if (a_i * a_j) % (a_i + a_j) == 0
    # For pairs of distinct elements
    
    # Build conflict as bitsets for fast operations
    # conflict_mask[i] = bitmask of elements that conflict with element i
    conflict_mask = [0] * (N + 1)  # 1-indexed
    good_mask = [0] * (N + 1)  # elements that do NOT conflict with element i
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    # For each element, the "compatible" elements are those that don't conflict
    # We want a maximum independent set in the conflict graph
    
    # Strategy: Simulated Annealing with random removal + greedy refill
    # Combined with multi-trial greedy and local search
    
    def can_add(element, current_set_mask):
        """Check if element can be added to the set represented by current_set_mask"""
        return (conflict_mask[element] & current_set_mask) == 0
    
    def set_size(mask):
        return bin(mask).count('1')
    
    def greedy_build(order, max_time=None):
        """Build a set by greedily adding elements in the given order."""
        if max_time is not None and time.time() > max_time:
            return 0
        result_mask = 0
        for e in order:
            if can_add(e, result_mask):
                result_mask |= (1 << e)
        return result_mask
    
    def local_search(mask, max_time=None):
        """Try to improve by swapping or adding/removing elements."""
        current_mask = mask
        current_size = set_size(current_mask)
        
        while True:
            if max_time is not None and time.time() > max_time:
                break
            
            improved = False
            
            # Try adding elements not in set
            for e in range(1, N + 1):
                if not (current_mask & (1 << e)):
                    if can_add(e, current_mask):
                        current_mask |= (1 << e)
                        current_size += 1
                        improved = True
            
            # Try swapping: remove one element, add two others
            for e in range(1, N + 1):
                if current_mask & (1 << e):
                    # Remove e
                    candidate = current_mask & ~(1 << e)
                    # Try to add two elements
                    for f in range(1, N + 1):
                        if not (candidate & (1 << f)) and can_add(f, candidate):
                            new_candidate = candidate | (1 << f)
                            for g in range(f + 1, N + 1):
                                if not (new_candidate & (1 << g)) and can_add(g, new_candidate):
                                    if set_size(new_candidate) > current_size:
                                        current_mask = new_candidate
                                        current_size += 1
                                        improved = True
                    if improved:
                        break
            if not improved:
                break
        
        return current_mask
    
    # Multi-trial greedy with random orders
    best_mask = 0
    best_size = 0
    
    # First, try a deterministic greedy
    order = list(range(1, N + 1))
    mask = greedy_build(order)
    mask = local_search(mask, max_time=start_time + 20.0)
    best_mask = mask
    best_size = set_size(mask)
    
    # Now do multiple random trials
    rng = random.Random(42)
    
    for trial in range(100):
        if time.time() > deadline - 5.0:
            break
        
        # Random order
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        # Greedy build
        mask = greedy_build(order)
        
        # Local search
        mask = local_search(mask, max_time=time.time() + 0.5)
        
        size = set_size(mask)
        if size > best_size:
            best_mask = mask
            best_size = size
    
    # Simulated annealing phase
    if time.time() < deadline - 10.0:
        current_mask = best_mask
        current_size = best_size
        temp = 5.0
        cooling = 0.995
        
        while time.time() < deadline - 5.0 and temp > 0.1:
            # Random move: remove a random element from set, add a random element not in set
            in_set = [e for e in range(1, N + 1) if current_mask & (1 << e)]
            not_in_set = [e for e in range(1, N + 1) if not (current_mask & (1 << e))]
            
            if not in_set or not not_in_set:
                break
            
            # Choose random swap
            e_remove = rng.choice(in_set)
            e_add = rng.choice(not_in_set)
            
            new_mask = (current_mask & ~(1 << e_remove)) | (1 << e_add)
            
            # Check validity
            valid = True
            for e in range(1, N + 1):
                if new_mask & (1 << e):
                    if conflict_mask[e] & new_mask & ~(1 << e):
                        valid = False
                        break
            
            if valid:
                new_size = set_size(new_mask)
                delta = new_size - current_size
                if delta > 0 or rng.random() < math.exp(delta / temp):
                    current_mask = new_mask
                    current_size = new_size
                    if current_size > best_size:
                        best_mask = current_mask
                        best_size = current_size
            
            temp *= cooling
        
        # Local search on best
        if time.time() < deadline - 2.0:
            polished = local_search(best_mask, max_time=deadline - 1.0)
            if set_size(polished) > best_size:
                best_mask = polished
                best_size = best_size
    
    # Convert mask to list
    result = [e for e in range(1, N + 1) if best_mask & (1 << e)]
    
    # Ensure nonempty
    if not result:
        result = [1]
    
    # Validate
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                # Should not happen, but if it does, remove b
                result.pop(j)
                j -= 1
    
    return result if result else [1]
