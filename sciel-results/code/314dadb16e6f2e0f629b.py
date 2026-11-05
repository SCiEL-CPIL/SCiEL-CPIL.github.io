import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # Build conflict graph using bitmasks for speed
    # conflicts[i] is a bitmask of j such that (i+1) + (j+1) is a square
    # Index 0 corresponds to value 1
    conflicts = [0] * N
    
    for i in range(N):
        val_i = i + 1
        mask = 0
        for j in range(N):
            val_j = j + 1
            if val_i + val_j in squares:
                mask |= (1 << j)
        conflicts[i] = mask
        
    # Initial set: all x such that x % 3 == 1
    # This is a valid independent set because (1+1)%3 = 2, not a square mod 3?
    # Wait, squares mod 3 are 0, 1. 1+1=2. So 1+1 is never a square.
    # 1+2=0 mod 3. 0 is a square (3^2=9, 6^2=36). So 1 and 2 conflict.
    # 2+2=1 mod 3. 1 is a square. So 2 and 2 conflict (diagonal).
    # So {1, 4, 7...} is safe?
    # 1+1=2 (not sq). 1+4=5 (not sq). 4+4=8 (not sq).
    # Yes, residue 1 mod 3 works.
    
    current_mask = 0
    for v in vertices:
        if v % 3 == 1:
            current_mask |= (1 << (v - 1))
            
    best_mask = current_mask
    best_size = bin(current_mask).count('1')
    
    # Helper to check if adding vertex i to mask is valid
    # We need (conflicts[i] & mask) == 0
    # Note: conflicts[i] includes i itself if 2*i is a square.
    # If i is not in mask, we just check against existing members.
    
    # Simulated Annealing
    # State: mask
    # Moves: 
    # 1. Add a random vertex not in mask
    # 2. Remove a random vertex from mask
    # 3. Swap: Remove one, add one
    
    # Precompute list of indices for faster random selection
    all_indices = list(range(N))
    
    T = 1.0
    cooling = 0.9999
    
    # We will run until deadline
    while time.time() - start_time < DEADLINE:
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a vertex
            # Find candidates not in mask
            # To make this O(N) or better, we can just pick a random index and check
            # If we want to be smarter, we could maintain a list of available vertices, 
            # but that's O(N) to update. 
            # Let's just pick random indices until we find one not in mask (expected O(1) if density < 0.5)
            added = False
            for _ in range(10): # Limit retries to avoid hanging on full sets
                idx = random.choice(all_indices)
                if not (current_mask & (1 << idx)):
                    # Check if adding idx is valid
                    if (conflicts[idx] & current_mask) == 0:
                        new_mask = current_mask | (1 << idx)
                        new_size = bin(new_mask).count('1')
                        # Always accept improvement
                        current_mask = new_mask
                        added = True
                        break
            if added:
                if bin(current_mask).count('1') > best_size:
                    best_mask = current_mask
                    best_size = bin(best_mask).count('1')
                continue # Skip removal/swap if we successfully added
            
        elif move_type < 0.7:
            # Try to remove a vertex (to allow adding others later)
            # Only remove if current set is not "full" or to escape local opt
            # Actually, removal is only useful if it enables a larger set.
            # In SA, we accept worse states.
            # Let's pick a random vertex from current mask
            # Get list of set bits
            # This is O(N) per step. 
            # Optimization: maintain a list? No, let's just iterate.
            # For N=256, iterating is fast.
            
            # Find a random set bit
            # Python doesn't have a direct "random set bit" op.
            # We can use a list of indices that are set.
            set_indices = [i for i in range(N) if (current_mask >> i) & 1]
            if set_indices:
                idx = random.choice(set_indices)
                new_mask = current_mask & ~(1 << idx)
                new_size = bin(new_mask).count('1')
                
                # Acceptance criterion
                # Delta size = -1
                # Probability = exp(-(-1)/T) = exp(1/T) > 1. Always accept?
                # No, usually we minimize cost. Here we maximize size.
                # Cost = -size.
                # Delta cost = -new_size - (-current_size) = current_size - new_size = 1.
                # Prob = exp(-1/T).
                
                if random.random() < math.exp(-1.0 / T):
                    current_mask = new_mask
                    # Update best? No, it's worse.
        else:
            # Swap: Remove one, add one
            # Pick a vertex to remove
            set_indices = [i for i in range(N) if (current_mask >> i) & 1]
            if set_indices:
                rem_idx = random.choice(set_indices)
                # Pick a vertex to add
                # Must not be rem_idx
                # And must not conflict with (current_mask - rem_idx)
                
                temp_mask = current_mask & ~(1 << rem_idx)
                
                # Find a candidate to add
                # Randomly try a few
                added_idx = -1
                for _ in range(5):
                    add_idx = random.choice(all_indices)
                    if add_idx == rem_idx:
                        continue
                    if not (temp_mask & (1 << add_idx)):
                        # Check conflict with temp_mask
                        if (conflicts[add_idx] & temp_mask) == 0:
                            added_idx = add_idx
                            break
                
                if added_idx != -1:
                    new_mask = temp_mask | (1 << added_idx)
                    # Size remains same
                    # We accept swap if it "improves" structure? 
                    # In SA for maximization with fixed size swaps, we usually accept with some probability 
                    # or if it allows future additions. 
                    # Let's just accept with high probability to explore.
                    if random.random() < 0.5:
                        current_mask = new_mask
                    # Check if this specific swap leads to a better state immediately? 
                    # It doesn't change size. But it might allow additions.
                    # We'll rely on the "Add" move to exploit this.
                    
        # Update best
        current_size = bin(current_mask).count('1')
        if current_size > best_size:
            best_mask = current_mask
            best_size = current_size
            
        # Cool down
        T *= cooling
        if T < 0.01:
            T = 0.01 # Floor

    # Final verification and conversion
    best_list = []
    for i in range(N):
        if (best_mask >> i) & 1:
            best_list.append(i + 1)
            
    # Verify validity
    valid = True
    for i in range(len(best_list)):
        a = best_list[i]
        for j in range(i, len(best_list)):
            b = best_list[j]
            if a + b in squares:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback
        return list(range(1, 257, 3))
        
    return best_list
