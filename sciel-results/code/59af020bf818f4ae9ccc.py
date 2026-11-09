import random
import time

def construct():
    start_time = time.time()
    best_set = []
    best_size = 0
    
    # Mask for bits 0 to 200
    mask = (1 << 201) - 1
    
    # Deterministic orders to try first
    # 1. Largest first
    # 2. Smallest first
    # 3. Middle out (50, 51, 49, 52, ...)
    # 4. Alternating parity (evens then odds, or vice versa)
    
    middle_out = []
    left, right = 1, 100
    while left <= right:
        middle_out.append(left)
        if left != right:
            middle_out.append(right)
        left += 1
        right -= 1
        
    evens = list(range(2, 101, 2))
    odds = list(range(1, 100, 2))
    evens_then_odds = evens + odds
    odds_then_evens = odds + evens
    
    orders = [
        list(range(100, 0, -1)), # Largest first
        list(range(1, 101)),     # Smallest first
        middle_out,
        evens_then_odds,
        odds_then_evens
    ]
    
    max_iterations = 50000 # Increased significantly
    
    for i in range(max_iterations):
        if i < len(orders):
            order = orders[i]
        else:
            order = list(range(1, 101))
            random.shuffle(order)
        
        selected = []
        reachable_bits = 1 # Bit 0 is set
        
        for x in order:
            # Check if 200-x is reachable
            if (reachable_bits >> (200 - x)) & 1:
                continue
            
            selected.append(x)
            # Update reachable bits
            new_bits = (reachable_bits << x) & mask
            reachable_bits |= new_bits
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
            
        if time.time() - start_time > 85:
            break
            
    return best_set
