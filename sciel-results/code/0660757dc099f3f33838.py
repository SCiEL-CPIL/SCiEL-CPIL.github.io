import random
import time

def construct():
    deadline = time.time() + 55
    best_set = set(range(66, 101))
    best_size = len(best_set)
    
    def is_valid(s):
        reachable = 1
        for x in s:
            reachable |= (reachable << x)
        return (reachable >> 200) & 1 == 0

    if not is_valid(best_set):
        selected, reachable = [], {0}
        for x in range(100, 0, -1):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected

    for pass_num in range(30):
        if time.time() > deadline:
            break
        
        candidates = list(range(1, 66))
        random.shuffle(candidates)
        
        improved = False
        for x in candidates:
            if time.time() > deadline:
                break
            
            new_set = best_set | {x}
            if is_valid(new_set):
                best_set = new_set
                best_size += 1
                improved = True
            else:
                for y in list(best_set):
                    if time.time() > deadline:
                        break
                    candidate_set = (best_set - {y}) | {x}
                    if is_valid(candidate_set):
                        best_set = candidate_set
                        improved = True
                        break
        
        if not improved:
            break
    
    return list(best_set)
