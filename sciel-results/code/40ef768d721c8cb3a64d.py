"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # safety margin before 90s kill
    
    N = 256
    
    # Precompute conflict_mask: conflict_mask[i] is a bitmask of indices j where i and j conflict
    # (a*b) % (a+b) == 0
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    def count_bits(mask):
        return bin(mask).count('1')
    
    # Strategy:
    # 1. Start with all odd numbers (indices 0, 2, 4, ..., 254). They are mutually compatible.
    # 2. Greedily add even numbers that are compatible with the current set.
    # 3. Local search to optimize the set of even numbers.
    
    # Phase 1: Initialize with all odds
    current_mask = 0
    for i in range(0, N, 2):
        current_mask |= (1 << i)
    
    # Phase 2: Greedy addition of evens
    # Sort evens by number of conflicts with odds (or just iterate)
    # To be more effective, we can try adding evens in order of increasing conflict count with the current set
    # But since the set grows, dynamic sorting is expensive. 
    # A simple pass adding any compatible even is a good start.
    # We can do multiple passes or sort by static conflict count with odds.
    
    # Let's compute conflict count with odds for each even
    odds_mask = current_mask
    even_indices = [i for i in range(1, N, 2)]
    
    # Sort evens by how many conflicts they have with the odds
    # This helps prioritize evens that are "less conflicting"
    def conflict_with_odds(idx):
        return count_bits(conflict_mask[idx] & odds_mask)
    
    even_indices.sort(key=conflict_with_odds)
    
    for idx in even_indices:
        if (current_mask >> idx) & 1:
            continue
        if (conflict_mask[idx] & current_mask) == 0:
            current_mask |= (1 << idx)
    
    best_mask = current_mask
    best_size = count_bits(current_mask)
    
    # Phase 3: Local Search
    # We focus on swapping even numbers. 
    # Moves:
    # 1. Add an even if possible (increases size)
    # 2. Remove an even (decreases size, only if it allows adding more? No, simple hill climb on size)
    #    Actually, if we remove an even, we might be able to add two others. 
    #    Let's stick to: Add if possible, or Swap (remove one even, add another even that was previously blocked).
    #    Or Swap (remove one even, add one odd? No, odds are all in. Removing an odd to add two evens might be beneficial.)
    #    Let's allow swapping any element with any other element to maximize size.
    
    def local_search(start_mask, deadline, max_iters=50000):
        cur_m = start_mask
        cur_sz = count_bits(cur_m)
        best_m = start_mask
        best_sz = cur_sz
        
        # Precompute list of elements in mask for faster random selection? 
        # With N=256, bit operations are fast enough.
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random move type
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random element not in current
                # Pick a random index
                idx = random.randint(0, N - 1)
                if (cur_m >> idx) & 1:
                    continue
                if (conflict_mask[idx] & cur_m) == 0:
                    new_m = cur_m | (1 << idx)
                    new_sz = cur_sz + 1
                    if new_sz > best_sz:
                        best_m = new_m
                        best_sz = new_sz
                    cur_m = new_m
                    cur_sz = new_sz
            elif move_type < 0.8:
                # Try to swap: remove one, add one
                # Pick random element to remove
                idx_rm = random.randint(0, N - 1)
                if not ((cur_m >> idx_rm) & 1):
                    continue
                
                # Pick random element to add
                idx_add = random.randint(0, N - 1)
                if idx_add == idx_rm:
                    continue
                if (cur_m >> idx_add) & 1:
                    continue
                
                # Check if idx_add is compatible with (cur_m without idx_rm)
                trial_base = cur_m & ~(1 << idx_rm)
                if (conflict_mask[idx_add] & trial_base) == 0:
                    new_m = trial_base | (1 << idx_add)
                    new_sz = cur_sz # Size remains same
                    # We only update best if size is strictly larger? 
                    # In this case size is same. 
                    # However, this move might enable future adds. 
                    # For a simple hill climber on size, we don't accept same-size moves unless we use simulated annealing.
                    # Let's try to accept same-size moves with some probability to escape local optima?
                    # Or just stick to size-increasing moves? 
                    # If we only accept size-increasing, we get stuck when no more single adds are possible.
                    # Then we need swaps that maintain size but change structure to allow future adds.
                    # Let's accept same-size swaps with 50% probability.
                    if random.random() < 0.5:
                        cur_m = new_m
                        # best_m/best_sz not updated as size didn't increase
            else:
                # Try 2-swap: remove two, add two
                # This can increase size by 0, but change structure.
                # Or we can try to add 2 if possible?
                # Let's try to find 2 elements to add that are compatible with each other and current set.
                # This is expensive. 
                # Alternative: Remove 1, Add 2? 
                # If we remove 1 element, we free up 1 "slot" in terms of conflicts. 
                # If we can add 2 elements that were only blocked by the removed one, we gain +1 size.
                
                idx_rm = random.randint(0, N - 1)
                if not ((cur_m >> idx_rm) & 1):
                    continue
                
                trial_base = cur_m & ~(1 << idx_rm)
                
                # Find candidates to add
                # We want to add 2 elements from the complement.
                # This is a search problem. 
                # Let's just try a few random pairs.
                found = 0
                new_m = trial_base
                for _ in range(10):
                    idx_add = random.randint(0, N - 1)
                    if (new_m >> idx_add) & 1:
                        continue
                    if (conflict_mask[idx_add] & new_m) == 0:
                        new_m |= (1 << idx_add)
                        found += 1
                        if found == 2:
                            break
                
                if found == 2:
                    new_sz = cur_sz + 1
                    if new_sz > best_sz:
                        best_m = new_m
                        best_sz = new_sz
                    cur_m = new_m
                    cur_sz = new_sz
                elif found == 1 and random.random() < 0.1:
                    # Accept size decrease? No, usually bad.
                    pass
        
        return best_m
    
    # Run local search multiple times with different random seeds if time permits
    # Or just one long run.
    # Since the landscape might have multiple local optima, multi-start is good.
    
    # We have already done a greedy phase. Let's run local search on the result.
    # If we have time, we can restart from scratch (just odds) with different random choices in greedy?
    # Greedy was deterministic. Let's add randomness to greedy for multi-start.
    
    # Let's do 3-5 restarts if time allows.
    # Restart 1: Deterministic greedy (already done)
    best_mask = local_search(best_mask, DEADLINE - 5, max_iters=20000)
    best_size = count_bits(best_mask)
    
    # Restart 2-5: Randomized greedy
    for restart in range(4):
        if time.time() > DEADLINE - 10:
            break
        
        # Randomized greedy: shuffle evens, add if compatible
        current_mask = 0
        for i in range(0, N, 2):
            current_mask |= (1 << i)
        
        even_indices = [i for i in range(1, N, 2)]
        random.shuffle(even_indices)
        
        for idx in even_indices:
            if (current_mask >> idx) & 1:
                continue
            if (conflict_mask[idx] & current_mask) == 0:
                current_mask |= (1 << idx)
        
        improved = local_search(current_mask, DEADLINE - 5, max_iters=10000)
        sz = count_bits(improved)
        if sz > best_size:
            best_mask = improved
            best_size = sz
    
    # Convert back to numbers
    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    
    return result
# EVOLVE-BLOCK-END
