import time
import random

def _greedy_construction(order):
    selected = []
    reachable = [False] * 201
    reachable[0] = True
    
    for x in order:
        # Check if adding x creates sum 200
        # This is equivalent to checking if (200 - x) is in reachable
        # But we must also update reachable set if we add x
        # If 200-x is reachable, then adding x makes 200 reachable.
        
        if 200 - x >= 0 and reachable[200 - x]:
            # Adding x would create a subset sum of 200
            continue
            
        # Add x
        selected.append(x)
        
        # Update reachable sums
        # new_sums = {s + x for s in reachable if s + x <= 200}
        # reachable |= new_sums
        # To avoid modifying while iterating, we can do:
        # for s in range(200 - x, -1, -1):
        #     if reachable[s]:
        #         reachable[s + x] = True
        # Note: We only care about sums <= 200.
        # If s + x > 200, we don't care.
        
        for s in range(200 - x, -1, -1):
            if reachable[s]:
                reachable[s + x] = True
                
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    # Initial deterministic greedy
    best_set = _greedy_construction(range(100, 0, -1))
    best_size = len(best_set)
    
    # Random restarts
    # We can also try ascending order, etc.
    # But random shuffles are more likely to explore the space.
    
    # To speed up, we can pre-check if a random shuffle is likely to be worse?
    # No, just run it.
    
    count = 0
    while time.time() < deadline:
        order = list(range(1, 101))
        random.shuffle(order)
        current_set = _greedy_construction(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            # If we reached a theoretical max? 
            # Max possible is 100. But 200 constraint limits it.
            # If best_size == 100, break (impossible).
            # If best_size is high, maybe stop early?
            # Let's just keep searching until deadline.
        count += 1
        
    return best_set
