"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit
    
    def greedy(order, deadline):
        """Greedy construction given an order of elements to consider."""
        selected = []
        reachable = [False] * 201
        reachable[0] = True
        
        for x in order:
            if time.time() > deadline:
                break
            if 200 - x >= 0 and reachable[200 - x]:
                continue
            for s in range(200 - x, -1, -1):
                if reachable[s]:
                    reachable[s + x] = True
            selected.append(x)
        
        return selected, reachable
    
    best = None
    best_len = 0
    
    # Try multiple orderings
    orderings = []
    orderings.append(list(range(100, 0, -1)))
    orderings.append(list(range(1, 101)))
    
    rng = random.Random(42)
    for _ in range(80):
        perm = list(range(1, 101))
        rng.shuffle(perm)
        orderings.append(perm)
    
    for order in orderings:
        if time.time() > deadline:
            break
        result, reachable = greedy(order, deadline)
        if len(result) > best_len:
            best_len = len(result)
            best = result
            best_reachable = reachable
    
    if best is None or len(best) == 0:
        return [100]
    
    # Local improvement: try to add excluded elements
    best_set = set(best)
    excluded = [x for x in range(1, 101) if x not in best_set]
    improved = list(best)
    reachable = best_reachable[:]  # copy
    
    for x in excluded:
        if time.time() > deadline:
            break
        if 200 - x < 0 or not reachable[200 - x]:
            for s in range(200 - x, -1, -1):
                if reachable[s]:
                    reachable[s + x] = True
            improved.append(x)
            best_set.add(x)
    
    if len(improved) > best_len:
        best = improved
        best_len = len(improved)
    
    return best
# EVOLVE-BLOCK-END
