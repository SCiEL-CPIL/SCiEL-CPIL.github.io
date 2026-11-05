import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0 # 90s limit, leave 5s margin
    
    # 1. Two-Phase Greedy Initialization
    # Phase 1: Greedy on [51, 100]
    # Phase 2: Greedy on [1, 50]
    selected = []
    reachable = 1
    
    # Phase 1
    for x in range(100, 50, -1):
        new_reach = reachable | (reachable << x)
        if not (new_reach >> 200) & 1:
            selected.append(x)
            reachable = new_reach
            
    # Phase 2
    for x in range(50, 0, -1):
        new_reach = reachable | (reachable << x)
        if not (new_reach >> 200) & 1:
            selected.append(x)
            reachable = new_reach
            
    best_set = set(selected)
    best_size = len(best_set)
    
    # 2. Local Search with Directed Moves
    current_set = best_set.copy()
    current_reachable = reachable
    
    all_elements = list(range(1, 101))
    
    def can_add(x, R):
        return not ((R >> (200 - x)) & 1)
        
    def update_reachable(R, x):
        return R | (R << x)
        
    def is_valid_set(S):
        r = 1
        for x in S:
            r |= (r << x)
            if (r >> 200) & 1:
                return False
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any single missing element
        missing = [x for x in all_elements if x not in current_set]
        random.shuffle(missing)
        
        for x in missing:
            if can_add(x, current_reachable):
                current_set.add(x)
                current_reachable = update_reachable(current_reachable, x)
                improved = True
                if len(current_set) > best_size:
                    best_set = current_set.copy()
                    best_size = len(current_set)
        
        # Phase 2: Swaps
        if not improved:
            # Try Swap 1-2 and Swap 1-3
            for swap_type in [2, 3]:
                if time.time() > deadline: break
                
                attempts = 500 if swap_type == 2 else 100
                for _ in range(attempts):
                    if time.time() > deadline: break
                    
                    if not current_set: break
                    a = random.choice(list(current_set))
                    
                    missing_now = [x for x in all_elements if x not in current_set]
                    if len(missing_now) < swap_type: break
                    
                    additions = random.sample(missing_now, swap_type)
                    
                    new_set = current_set.copy()
                    new_set.remove(a)
                    for x in additions:
                        new_set.add(x)
                    
                    if is_valid_set(new_set):
                        current_set = new_set
                        current_reachable = 1
                        for x in current_set:
                            current_reachable |= (current_reachable << x)
                        
                        improved = True
                        if len(current_set) > best_size:
                            best_set = current_set.copy()
                            best_size = len(current_set)
                        break # Restart outer loop to try additions again

    return list(best_set)
