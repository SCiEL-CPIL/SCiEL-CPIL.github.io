import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # We will use a bitmask to represent reachable sums.
    # Bit i is 1 if sum i is reachable.
    # We care about sums up to 200.
    # mask = 1 initially (sum 0 is reachable).
    
    def try_greedy(order):
        selected = []
        mask = 1  # sum 0 is reachable
        # Precompute mask limit: bits 0 to 200
        limit_mask = (1 << 201) - 1
        
        for x in order:
            # Check if adding x creates sum 200
            # This happens if (200 - x) is currently reachable
            # x is in [1, 100], so 200-x is in [100, 199]
            if (mask >> (200 - x)) & 1:
                continue
            
            # Add x
            selected.append(x)
            # Update reachable sums
            # New sums are old_sums + x
            # We only care about sums <= 200
            new_sums_mask = (mask << x) & limit_mask
            mask |= new_sums_mask
            
        return selected

    best_set = []
    
    # 1. Deterministic Descending
    # Large numbers are "safer" as they have fewer partners to sum to 200
    order_desc = list(range(100, 0, -1))
    candidate = try_greedy(order_desc)
    if len(candidate) > len(best_set):
        best_set = candidate

    # 2. Deterministic Ascending
    order_asc = list(range(1, 101))
    candidate = try_greedy(order_asc)
    if len(candidate) > len(best_set):
        best_set = candidate

    # 3. Randomized Restarts
    # With bitmask, each run is very fast.
    elements = list(range(1, 101))
    
    # Run random restarts until time runs out
    # We can run many more restarts now.
    while time.time() < deadline:
        random.shuffle(elements)
        candidate = try_greedy(elements)
        
        if len(candidate) > len(best_set):
            best_set = candidate
            
    return best_set
