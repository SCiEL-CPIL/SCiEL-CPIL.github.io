import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Initial small set to start with? Or start empty.
    # Starting empty is fine.
    
    # To speed up, we can try a few different strategies
    # 1. Random greedy
    # 2. Greedy with specific ordering (e.g., powers of 2 first, then fill gaps)
    
    # Let's try random greedy with many restarts.
    
    max_val = 256
    
    while time.time() < deadline:
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        current_set = []
        # sums[k] is bitset of sums for cardinality k
        # Start with k=0
        sums = [0] * (max_val + 2)  # Max size is bounded by max_val, but realistically < 20
        sums[0] = 1  # Bit 0 is set
        
        current_size = 0
        
        for x in candidates:
            # Check if x can be added
            # Condition: for all k, (sums[k] << x) & (total_mask ^ sums[k+1]) == 0
            # Note: sums[k+1] might be 0 if k+1 > current_size
            
            total_mask = 0
            for i in range(current_size + 1):
                total_mask |= sums[i]
                
            valid = True
            for k in range(current_size + 1):
                new_part = sums[k] << x
                # We need to ensure new_part does not intersect any sums[i] for i != k+1
                # The union of all sums[i] is total_mask.
                # The union of sums[i] for i != k+1 is total_mask with sums[k+1] removed.
                # Since they are disjoint, we can just mask out sums[k+1].
                # However, if k+1 > current_size, sums[k+1] is 0, so we mask out 0.
                
                # Actually, we just need to check intersection with total_mask.
                # If it intersects, does it intersect only sums[k+1]?
                # The new sums for cardinality k+1 are sums[k+1] | (sums[k] << x).
                # The old sums[k+1] are already in total_mask.
                # The new part is (sums[k] << x).
                # We need (sums[k] << x) to not intersect any sums[i] where i != k+1.
                # So (sums[k] << x) & (total_mask ^ sums[k+1]) must be 0.
                
                mask = total_mask ^ sums[k+1] if k+1 <= current_size else total_mask
                if (new_part & mask) != 0:
                    valid = False
                    break
            
            if valid:
                # Add x
                current_set.append(x)
                current_size += 1
                
                # Update sums
                # We need to update from high to low to avoid overwriting
                # sums[k] = sums[k] | (sums[k-1] << x)
                # But sums[k-1] is the old value.
                # So we must iterate k from current_size down to 1.
                
                for k in range(current_size, 0, -1):
                    sums[k] |= (sums[k-1] << x)
                
                if current_size > best_size:
                    best_size = current_size
                    best_set = list(current_set)
                    
                # If we reached a good size, we can break early?
                # No, we want to maximize. But if we run out of candidates, loop ends.
                
    # Ensure return type is list of ints
    return [int(x) for x in best_set]
