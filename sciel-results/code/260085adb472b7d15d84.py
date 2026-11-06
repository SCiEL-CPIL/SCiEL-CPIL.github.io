import random
import time

def _greedy(order):
    chosen = []
    # Max sum is 1000 + 1000 = 2000
    counts = [0] * 2001
    
    for x in order:
        # Check validity
        # New pairs: (x, x) and (x, y) for y in chosen
        # We need to ensure counts[s] + 1 <= 2 for all affected s
        # i.e., counts[s] <= 1 for all affected s
        
        valid = True
        if counts[2*x] > 1:
            valid = False
        else:
            for y in chosen:
                s = x + y
                if counts[s] > 1:
                    valid = False
                    break
        
        if valid:
            chosen.append(x)
            counts[2*x] += 1
            for y in chosen[:-1]:
                s = x + y
                counts[s] += 1
                
    return chosen

def construct():
    start_time = time.time()
    deadline = 115.0 # 5s safety margin
    
    best_set = []
    best_size = 0
    
    # Run baseline
    order = list(range(1, 1001))
    s = _greedy(order)
    best_set = s
    best_size = len(s)
    
    # Random restarts
    # We can also try to "warm start" from the best set?
    # No, greedy is order-dependent.
    
    # To make it faster, we can pre-generate orders?
    # No, just shuffle on the fly.
    
    while time.time() < deadline:
        order = list(range(1, 1001))
        random.shuffle(order)
        s = _greedy(order)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
            
    return best_set
