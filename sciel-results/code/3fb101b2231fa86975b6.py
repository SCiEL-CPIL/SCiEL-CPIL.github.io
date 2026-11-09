import random
import time

def construct():
    start_time = time.time()
    best_set = []
    best_size = 0
    
    # Mask for bits 0 to 200
    mask = (1 << 201) - 1
    
    # Deterministic orders to try first
    middle_out = []
    left, right = 1, 100
    while left <= right:
        middle_out.append(left)
        if left != right:
            middle_out.append(right)
        left += 1
        right -= 1
        
    rev_middle_out = []
    left, right = 1, 100
    while left <= right:
        rev_middle_out.append(right)
        if left != right:
            rev_middle_out.append(left)
        left += 1
        right -= 1
        
    evens = list(range(2, 101, 2))
    odds = list(range(1, 100, 2))
    evens_then_odds = evens + odds
    odds_then_evens = odds + evens
    
    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        middle_out,
        evens_then_odds,
        odds_then_evens,
        rev_middle_out,
        list(range(50, 0, -1)) + list(range(51, 101)),
        list(range(1, 51)) + list(range(100, 50, -1)),
        list(range(1, 101, 2)) + list(range(100, 0, -2)),
        list(range(2, 101, 2)) + list(range(99, 0, -2)),
    ]
    
    max_iterations = 50000
    
    for i in range(max_iterations):
        if time.time() - start_time > 82:
            break
            
        if i < len(orders):
            order = orders[i]
        else:
            order = list(range(1, 101))
            random.shuffle(order)
        
        selected = []
        reachable_bits = 1
        
        for x in order:
            if (reachable_bits >> (200 - x)) & 1:
                continue
            
            selected.append(x)
            new_bits = (reachable_bits << x) & mask
            reachable_bits |= new_bits
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
            
    return best_set
