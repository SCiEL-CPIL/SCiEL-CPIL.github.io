import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_set = None
    best_size = N + 1
    
    # Helper to check and update
    # We will use two strategies:
    # 1. Randomized Greedy (Bitset)
    # 2. Greedy Max Conflict (Bitset) - limited runs
    
    # Strategy 1: Randomized Greedy
    # We run this until time runs out or we find a very small set.
    
    # To interleave, we can do 1 Max Conflict, then 100 Random Greedy, etc.
    # Or just Random Greedy, as it's simpler and proven.
    # Let's stick to Randomized Greedy with Bitsets for reliability and speed.
    
    while time.time() < deadline:
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        selected = []
        sums_mask = 0
        
        for x in order:
            # Calculate new sums mask for x given current selected
            # 2*x
            new_mask = 1 << (2 * x)
            
            # x + s for s in selected
            # Optimization: if selected is empty, skip loop
            for s in selected:
                new_mask |= 1 << (x + s)
                
            # Check collision
            if sums_mask & new_mask:
                continue
                
            # No collision, add x
            selected.append(x)
            sums_mask |= new_mask
            
        size = len(selected)
        if size < best_size:
            best_size = size
            best_set = selected
            
            # If we find a size <= 26, it's likely near optimal.
            # We can continue to try for 25, but diminishing returns.
            
    return best_set
