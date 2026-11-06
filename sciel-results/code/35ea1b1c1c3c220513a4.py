import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    best_set = []
    mask = (1 << 201) - 1 # Bits 0 to 200
    
    orders = []
    orders.append(list(range(100, 0, -1)))
    orders.append(list(range(1, 101)))
    
    base = list(range(1, 101))
    for _ in range(50):
        random.shuffle(base)
        orders.append(base[:])
        
    orders.append(sorted(range(1, 101), key=lambda x: x % 2))
    orders.append(sorted(range(1, 101), key=lambda x: x % 3))
    orders.append(sorted(range(1, 101), key=lambda x: x % 5))
    
    for order in orders:
        if time.time() > deadline:
            break
            
        selected = []
        reachable_bits = 1 # Bit 0 is set
        
        for x in order:
            # Check if adding x creates sum 200
            # New sums are (reachable_bits << x)
            # We need to check if bit 200 is set in the NEW sums
            # But we only add x if it doesn't create 200.
            # 200 is created if (reachable_bits >> (200-x)) & 1 is 1?
            # No, 200 is formed by s + x = 200 => s = 200 - x.
            # So we check if bit (200-x) is set in reachable_bits.
            
            if (reachable_bits >> (200 - x)) & 1:
                # 200 - x is reachable, so adding x makes 200. Skip.
                continue
            else:
                # Safe to add
                selected.append(x)
                # Update reachable bits
                new_bits = (reachable_bits << x) & mask
                reachable_bits |= new_bits
                
        if len(selected) > len(best_set):
            best_set = selected
            
    return best_set
