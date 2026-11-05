import random
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Initial greedy run to get a baseline
    chosen, counts = [], {}
    for x in range(1, 1001):
        # Check if x can be added
        valid = True
        # Check sums with existing elements
        for y in chosen:
            s = x + y
            if counts.get(s, 0) + 1 > 2:
                valid = False
                break
        if valid:
            # Check double sum
            if counts.get(2 * x, 0) + 1 > 2:
                valid = False
        
        if valid:
            chosen.append(x)
            # Update counts
            for y in chosen[:-1]:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            counts[2 * x] = counts.get(2 * x, 0) + 1
            
    best_set = chosen
    best_size = len(chosen)
    
    # Randomized greedy runs
    while time.time() < deadline:
        random.seed() # Or use a counter for reproducibility if needed, but random is fine
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        chosen = []
        counts = {}
        
        for x in candidates:
            valid = True
            # Check sums with existing elements
            for y in chosen:
                s = x + y
                if counts.get(s, 0) + 1 > 2:
                    valid = False
                    break
            if valid:
                # Check double sum
                if counts.get(2 * x, 0) + 1 > 2:
                    valid = False
            
            if valid:
                chosen.append(x)
                # Update counts
                for y in chosen[:-1]:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen
            
            # If we found a very good set, we might want to break early or continue?
            # Let's continue to see if we can find even better.
            
    return best_set
