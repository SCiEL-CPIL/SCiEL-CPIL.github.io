# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Precompute neighbor bitmasks
    # neighbor_mask[u] is a bitmask where bit v is 1 if (u-v)%m is in forbidden
    neighbor_mask = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u == v:
                continue
            if (u - v) % m in forbidden:
                mask |= (1 << v)
        neighbor_mask[u] = mask
    
    # Greedy initialization using bitmasks
    def greedy_init():
        current_mask = 0
        for v in range(m):
            if not (current_mask & neighbor_mask[v]):
                current_mask |= (1 << v)
        return current_mask
    
    best_mask = greedy_init()
    best_size = bin(best_mask).count('1')
    
    start_time = time.time()
    timeout = 580.0 # Leave 50s margin for 630s limit
    
    # Simulated Annealing
    temperature = 10.0
    cooling_rate = 0.9999
    min_temperature = 0.1
    
    current_mask = best_mask
    current_size = best_size
    
    while time.time() - start_time < timeout and temperature > min_temperature:
        # Pick a random move
        # Types of moves:
        # 1. Add a vertex (if compatible)
        # 2. Remove a vertex
        # 3. Swap: remove one, add another (or two)
        
        # We will try to find a valid add first (greedy improvement)
        # This is cheap and always improves
        added = False
        for v in range(m):
            if not (current_mask & (1 << v)) and not (current_mask & neighbor_mask[v]):
                current_mask |= (1 << v)
                current_size += 1
                added = True
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
                break # Restart loop with new state? Or just continue SA?
                # For SA, we usually accept immediate improvements.
        
        if added:
            # We found an improvement, reset temperature slightly? Or just continue.
            # Let's continue SA loop.
            continue
        
        # If no simple add, perform a random perturbation
        # Pick a random vertex to remove (if any)
        if current_size == 0:
            break
            
        # Get list of selected vertices
        selected_vertices = [i for i in range(m) if (current_mask >> i) & 1]
        
        # With some probability, do a 1-remove, 1-add or 1-remove, 2-add move
        # Let's try 1-remove, k-add where k is 1 or 2
        
        move_type = random.choice([1, 2]) # 1 or 2 additions after removal
        
        if not selected_vertices:
            break
            
        u = random.choice(selected_vertices)
        
        # Remove u
        new_mask = current_mask & ~(1 << u)
        new_size = current_size - 1
        
        # Try to add random candidates until we have added move_type vertices or tried enough
        # We need to ensure the added vertices are compatible with each other and new_mask
        # For simplicity and speed, we'll try a few random candidates
        
        added_count = 0
        attempts = 0
        max_attempts = 20
        
        while added_count < move_type and attempts < max_attempts:
            v = random.randint(0, m - 1)
            if new_mask & (1 << v):
                attempts += 1
                continue
            
            # Check compatibility with new_mask
            if new_mask & neighbor_mask[v]:
                attempts += 1
                continue
            
            # Check compatibility with previously added vertices in this move
            # Since we are adding to new_mask, we can just check against the current state of new_mask
            # But we need to track the specific vertices added in this step if we don't update new_mask immediately
            # Actually, if we update new_mask immediately, it handles compatibility with previous adds.
            
            new_mask |= (1 << v)
            new_size += 1
            added_count += 1
            attempts += 1
        
        if added_count == 0:
            # No valid add found, revert? Or accept as penalty?
            # If we can't add anything, the move is just a removal (size -1).
            # We might accept this to escape local optima.
            pass
        
        # Calculate delta
        delta = new_size - current_size
        
        # Acceptance probability
        if delta > 0 or random.random() < pow(2.718281828, delta / temperature):
            current_mask = new_mask
            current_size = new_size
            
            if current_size > best_size:
                best_mask = current_mask
                best_size = current_size
                # Reset temperature on improvement?
                # temperature = 5.0 # Optional, might help re-explore
        
        temperature *= cooling_rate
        
    # Convert best_mask to array
    result = np.array([i for i in range(m) if (best_mask >> i) & 1], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
