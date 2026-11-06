import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit (and 90s kill)
    
    n = 128
    target_min = 128
    target_max = 256
    target_len = target_max - target_min + 1  # 129 targets
    
    # Helper to evaluate a set A (represented as a numpy bool array v)
    # Returns (max_mult, num_zeros)
    def evaluate(v):
        # v is a boolean array of length 129
        # Convolution
        conv = np.convolve(v, v, mode='full')
        
        # Extract target range [128, 256]
        # conv indices: 0 to 256
        targets_ordered = conv[target_min:target_max+1]
        
        # Calculate diagonal contributions
        # t is in [128, 256]
        # t/2 is in [64, 128]
        # If t is even, t/2 is an integer. Check if v[t//2] is True.
        # We need to map target index i to t = 128 + i
        # t is even if (128 + i) % 2 == 0 => i % 2 == 0 (since 128 is even)
        
        # Create a mask for even targets in the range
        # i goes from 0 to 128
        # t = 128 + i
        # t is even iff i is even
        even_indices = np.arange(0, target_len, 2)
        
        # For these indices, check if t//2 is in A
        # t//2 = (128 + i) // 2 = 64 + i//2
        # So we need v[64 + i//2]
        
        diag = np.zeros(target_len, dtype=np.int32)
        v_indices = 64 + even_indices // 2
        diag[even_indices] = v[v_indices].astype(np.int32)
        
        # Unordered count
        # (ordered + diag) // 2
        unordered = (targets_ordered.astype(np.int32) + diag) // 2
        
        num_zeros = np.sum(unordered == 0)
        max_mult = np.max(unordered) if target_len > 0 else 0
        
        return max_mult, num_zeros

    # Objective function for search
    def cost(v):
        max_mult, num_zeros = evaluate(v)
        # Strong penalty for invalid sets (zeros in target range)
        return max_mult + 1000 * num_zeros

    # Initialize a random set
    # We want a set that is likely to cover the range.
    # Density p=0.5 usually works well for such covering problems.
    # Let's try a few random initializations and pick the best valid one.
    
    best_v = None
    best_cost = float('inf')
    best_max_mult = float('inf')
    
    # Time budget for initialization
    init_deadline = start_time + 2
    
    # Try a few random starts
    for _ in range(50):
        if time.time() > init_deadline:
            break
        v = np.random.random(129) < 0.5
        # Ensure A is not empty
        if not np.any(v):
            v[0] = True
            v[128] = True
            
        c = cost(v)
        if c < best_cost:
            best_cost = c
            best_v = v.copy()
            max_m, zeros = evaluate(v)
            if zeros == 0:
                best_max_mult = max_m
                # If we found a valid set early, we can stop initializing
                # but let's check a couple more just in case
                if max_m < 10: # Good enough
                    break
    
    if best_v is None:
        # Fallback: full set
        best_v = np.ones(129, dtype=bool)
        best_max_mult, _ = evaluate(best_v)
        best_cost = best_max_mult

    current_v = best_v.copy()
    current_cost = best_cost
    current_max_mult, current_zeros = evaluate(current_v)
    
    # Local Search (Hill Climbing)
    step_deadline = deadline
    
    # To speed up, we can precompute some things? 
    # np.convolve is fast enough for 129 size.
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < step_deadline:
        # Pick a random index to toggle
        idx = random.randint(0, 128)
        
        # Toggle
        current_v[idx] = not current_v[idx]
        
        # Evaluate
        new_max, new_zeros = evaluate(current_v)
        new_cost = new_max + 1000 * new_zeros
        
        if new_cost < current_cost:
            # Accept
            current_cost = new_cost
            no_improve_count = 0
            if new_zeros == 0:
                if new_max < best_max_mult:
                    best_max_mult = new_max
                    best_v = current_v.copy()
            else:
                # We accepted an invalid set because it had lower cost?
                # This only happens if current_cost was also very high (invalid).
                # We track best_v only if it's valid.
                pass
                
        elif new_cost == current_cost:
            # Accept to explore
            if random.random() < 0.5:
                pass # Keep the move
            else:
                # Revert
                current_v[idx] = not current_v[idx]
            no_improve_count += 1
        else:
            # Revert
            current_v[idx] = not current_v[idx]
            no_improve_count += 1
            
        # If stuck for a long time, try a random restart from best_v with noise
        if no_improve_count > max_no_improve:
            # Restart from best_v
            current_v = best_v.copy()
            current_cost = cost(current_v)
            current_max_mult, current_zeros = evaluate(current_v)
            no_improve_count = 0
            # Add some noise
            for _ in range(10):
                idx = random.randint(0, 128)
                current_v[idx] = not current_v[idx]
            current_cost = cost(current_v)
            current_max_mult, current_zeros = evaluate(current_v)

    # Final verification
    # The best_v found should be valid.
    # Convert numpy bool array to list of ints
    A_list = [i for i in range(129) if best_v[i]]
    
    return A_list
