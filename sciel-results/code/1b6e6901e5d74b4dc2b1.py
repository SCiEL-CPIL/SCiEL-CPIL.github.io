import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Bitset representation of reachable sums
    # Bit i is 1 if sum i is reachable
    # We care about sums up to 200
    # (1 << 201) - 1 creates a mask for bits 0 to 200
    mask = (1 << 201) - 1
    reachable = 1  # Sum 0 is reachable
    
    selected = []
    
    # Greedy selection from largest to smallest
    for x in range(100, 0, -1):
        if time.time() > deadline:
            break
            
        # Check if adding x creates a subset sum of 200
        # This happens if (200 - x) is already reachable
        # We check bit (200-x) in 'reachable'
        target_bit = 200 - x
        if target_bit >= 0:
            if (reachable >> target_bit) & 1:
                # Adding x would complete the sum 200
                continue
        
        # Add x to selected set
        selected.append(x)
        
        # Update reachable sums
        # New sums are old sums shifted by x
        reachable |= (reachable << x)
        
        # Mask to keep integer size manageable
        reachable &= mask
        
    # If we have time, we could try to improve, but greedy is likely optimal
    # For now, return the greedy result
    return selected
