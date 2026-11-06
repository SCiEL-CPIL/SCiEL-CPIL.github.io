import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    n = 4096
    candidates = list(range(1, n + 1))
    
    best_size = 0
    best_set = []
    
    def try_greedy(candidates_order):
        selected = []
        sums_set = set()
        
        for x in candidates_order:
            valid = True
            new_sums_set = set()
            
            # 1. 3x
            s = 3 * x
            if s in sums_set or s in new_sums_set:
                valid = False
            else:
                new_sums_set.add(s)
            
            if valid:
                # 2. 2x + a
                for a in selected:
                    s = 2 * x + a
                    if s in sums_set or s in new_sums_set:
                        valid = False
                        break
                    new_sums_set.add(s)
            
            if valid:
                # 3. x + a + b
                for i in range(len(selected)):
                    a = selected[i]
                    for j in range(i, len(selected)):
                        b = selected[j]
                        s = x + a + b
                        if s in sums_set or s in new_sums_set:
                            valid = False
                            break
                        new_sums_set.add(s)
                    if not valid:
                        break
            
            if valid:
                selected.append(x)
                sums_set.update(new_sums_set)
        
        return selected

    # 1. Deterministic greedy
    current_set = try_greedy(candidates)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set
        
    # 2. Random restarts
    while time.time() < deadline:
        random.shuffle(candidates)
        current_set = try_greedy(candidates)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return best_set
