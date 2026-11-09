import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85  # 85s out of 90s
    
    # Mask for bits 0 to 200
    mask = (1 << 201) - 1
    
    def greedy_select(order):
        """
        Greedily select elements in the given order.
        An element x is included if adding it does not create a subset sum of 200.
        Uses bitset for fast reachable sums tracking.
        Returns (selected_list, reachable_bits).
        """
        selected = []
        reachable_bits = 1  # Bit 0 is set (sum 0 is reachable)
        
        for x in order:
            # Check if 200-x is reachable
            # If bit (200-x) is set in reachable_bits, then sum 200-x exists.
            # Adding x would create 200.
            if (reachable_bits >> (200 - x)) & 1:
                continue
            
            selected.append(x)
            # Update reachable bits
            # Shift left by x, mask to 200 bits
            new_bits = (reachable_bits << x) & mask
            reachable_bits |= new_bits
            
        return selected, reachable_bits

    best_set = []
    best_size = 0
    
    # 1. Try deterministic orders first
    deterministic_orders = [
        list(range(100, 0, -1)),  # Largest first
        list(range(1, 101)),      # Smallest first
    ]
    
    for order in deterministic_orders:
        selected, _ = greedy_select(order)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
            
        if time.time() > deadline:
            return best_set
    
    # 2. Try random orders to find better solutions
    # Since the bitset approach is very fast, we can do many iterations.
    # We want to maximize cardinality, so we are looking for orders that allow many elements.
    # Random shuffling might find "lucky" orders where conflicts are avoided.
    
    # Also, we can try to "repair" or "improve" existing solutions, but greedy with different orders is simpler and often effective for this size.
    
    while time.time() < deadline:
        order = list(range(1, 101))
        random.shuffle(order)
        
        selected, _ = greedy_select(order)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
            
        # Optional: If we hit a high score (e.g., 50+), we might want to try more specific heuristics,
        # but for now, random restarts are a good baseline for structural change from the removal-based approach.
        
    return best_set
