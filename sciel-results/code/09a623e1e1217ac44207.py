import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    selected = []
    reachable = {0}
    
    # Greedy from largest to smallest
    for x in range(100, 0, -1):
        if time.time() > deadline:
            break
            
        # Check if adding x creates a sum of 200
        # We need to check if 200 - x is in reachable
        # If 200 - x is in reachable, then some subset S of selected sums to 200-x
        # Then S U {x} sums to 200
        if 200 - x in reachable:
            continue
            
        # Safe to add x
        selected.append(x)
        
        # Update reachable sums
        # new sums are s + x for s in reachable
        # We only care about sums <= 200
        new_sums = {s + x for s in reachable if s + x <= 200}
        reachable.update(new_sums)
        
    return selected
