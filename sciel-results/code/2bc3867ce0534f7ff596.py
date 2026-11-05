# EVOLVE-BLOCK-START
import random
import time

def construct():
    # Initial greedy from 100 down to 1
    selected, reachable = [], {0}
    for x in range(100, 0, -1):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    
    best_set = set(selected)
    best_size = len(best_set)
    all_elements = set(range(1, 101))
    
    def is_valid(S):
        sums = {0}
        for x in S:
            new_sums = {s + x for s in sums if s + x <= 200}
            if 200 in new_sums:
                return False
            sums |= new_sums
        return True

    start_time = time.time()
    time_limit = 85.0
    
    # Local Search: Try to increase size by swapping k elements for k+1
    # We try k=1 and k=2
    
    no_improve_time = 0.0
    last_improve_time = start_time
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Strategy 1: Remove 1, Add 2
        # Pick random element to remove
        if best_size > 0:
            out_el = random.choice(list(best_set))
            S_base = best_set - {out_el}
            
            # Pick 2 random elements from complement
            comp = list(all_elements - best_set)
            if len(comp) >= 2:
                random.shuffle(comp)
                # Try first few pairs
                for i in range(min(20, len(comp)-1)):
                    x1, x2 = comp[i], comp[i+1]
                    new_set = S_base | {x1, x2}
                    if is_valid(new_set):
                        best_set = new_set
                        best_size += 1
                        improved = True
                        last_improve_time = time.time()
                        break
        
        # Strategy 2: Remove 2, Add 3
        if not improved and best_size >= 2:
            # Pick 2 random to remove
            out_els = random.sample(list(best_set), 2)
            S_base = best_set - set(out_els)
            
            comp = list(all_elements - best_set)
            if len(comp) >= 3:
                random.shuffle(comp)
                for i in range(min(10, len(comp)-2)):
                    x1, x2, x3 = comp[i], comp[i+1], comp[i+2]
                    new_set = S_base | {x1, x2, x3}
                    if is_valid(new_set):
                        best_set = new_set
                        best_size += 1
                        improved = True
                        last_improve_time = time.time()
                        break
                        
        if not improved:
            # If no improvement for 5 seconds, try a different random seed or restart?
            # For now, just continue. The random choices will vary.
            if time.time() - last_improve_time > 5.0:
                # Maybe try a full restart with random shuffle of 1..100?
                # Greedy on random order might find a different local max.
                pass # Keeping it simple, the swap logic should explore enough.
                
    return list(best_set)
# EVOLVE-BLOCK-END
