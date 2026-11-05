import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s hard kill
    
    N = 256
    elements = list(range(1, N + 1))
    
    # Precompute conflict graph: conflict[a][b] = True if (a*b) % (a+b) == 0
    # For each a, store bitmask of b that conflict
    conflict_mask = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    def can_add(element, current_mask):
        """Check if element can be added to the set represented by current_mask"""
        return (conflict_mask[element] & current_mask) == 0
    
    def set_size(mask):
        return bin(mask).count('1')
    
    def greedy_build(order, start_mask=0):
        """Build a set by greedily adding elements in the given order to start_mask."""
        result_mask = start_mask
        for e in order:
            if can_add(e, result_mask):
                result_mask |= (1 << e)
        return result_mask
    
    def local_search_add(mask):
        """Try to add any element that fits. Returns improved mask."""
        improved = True
        while improved:
            improved = False
            for e in range(1, N + 1):
                if not (mask & (1 << e)):
                    if can_add(e, mask):
                        mask |= (1 << e)
                        improved = True
        return mask
    
    def local_search_swap(mask, max_time=None):
        """Try to improve by removing one element and adding others that fit."""
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
                    candidate = current_mask & ~(1 << e)
                    added = 0
                    for f in range(1, N + 1):
                        if not (candidate & (1 << f)) and can_add(f, candidate):
                            new_candidate = candidate | (1 << f)
                            added += 1
                            for g in range(f + 1, N + 1):
                                if not (new_candidate & (1 << g)) and can_add(g, new_candidate):
                                    if set_size(new_candidate) > current_size:
                                        current_mask = new_candidate
                                        current_size = set_size(current_mask)
                                        improved = True
                            if improved:
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current_mask
    
    # Strategy 1: Use the structural property that all odds form a valid independent set
    # Then greedily add compatible evens
    odds = [e for e in range(1, N + 1) if e % 2 == 1]
    evens = [e for e in range(1, N + 1) if e % 2 == 0]
    
    best_mask = 0
    best_size = 0
    
    # Start with all odds
    odds_mask = 0
    for e in odds:
        odds_mask |= (1 << e)
    
    # Greedily add evens
    mask = greedy_build(evens, odds_mask)
    mask = local_search_add(mask)
    best_mask = mask
    best_size = set_size(mask)
    
    # Strategy 2: Multiple random restarts with shuffled evens
    rng = random.Random(42)
    
    for trial in range(200):
        if time.time() > deadline - 10.0:
            break
        
        # Shuffle evens
        shuffled_evens = evens[:]
        rng.shuffle(shuffled_evens)
        
        # Greedily add evens to odds
        mask = greedy_build(shuffled_evens, odds_mask)
        mask = local_search_add(mask)
        
        size = set_size(mask)
        if size > best_size:
            best_mask = mask
            best_size = size
    
    # Strategy 3: Simulated annealing on the best solution
    if time.time() < deadline - 15.0:
        current_mask = best_mask
        current_size = best_size
        temp = 3.0
        cooling = 0.99
        
        while time.time() < deadline - 5.0 and temp > 0.05:
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
        
        # Final polish
        if time.time() < deadline - 2.0:
            polished = local_search_add(best_mask)
            if set_size(polished) > best_size:
                best_mask = polished
                best_size = set_size(polished)
    
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
                result.pop(j)
                j -= 1
    
    return result if result else [1]
