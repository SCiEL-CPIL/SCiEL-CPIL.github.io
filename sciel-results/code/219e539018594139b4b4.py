import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for evaluator overhead
    
    def greedy(order):
        selected = []
        reachable = {0}
        for x in order:
            # Check if adding x creates a sum of 200
            # We need to see if 200 is in {s + x | s in reachable, s + x <= 200}
            # This is equivalent to checking if (200 - x) is in reachable, provided 200 - x >= 0
            # But wait, the seed computes `new` set. 
            # If 200 - x is in reachable, then (200 - x) + x = 200.
            # So we just need to check if (200 - x) is in reachable.
            # However, we must also ensure that the new sums added don't exceed 200 for future steps?
            # The seed updates `reachable` with `new` which only contains sums <= 200.
            # So `reachable` always contains sums <= 200.
            # So the check `200 - x in reachable` is sufficient and faster than creating the set `new`.
            
            if (200 - x) in reachable:
                continue
            
            # Safe to add
            selected.append(x)
            # Update reachable
            # new_sums = {s + x for s in reachable if s + x <= 200}
            # reachable |= new_sums
            
            # Optimization: 
            # We can update reachable in place or create a new set.
            # Since reachable size is small (< 201), creating a new set is fine.
            new_reachable = set(reachable)
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    new_reachable.add(ns)
            reachable = new_reachable
            
        return selected

    # 1. Try the seed's order (descending)
    best = greedy(list(range(100, 0, -1)))
    best_len = len(best)
    
    # 2. Try ascending
    candidate = greedy(list(range(1, 101)))
    if len(candidate) > best_len:
        best = candidate
        best_len = len(candidate)
        
    # 3. Random shuffles
    # Estimate time per shuffle. 
    # 100 items, each check is O(1) lookup + O(200) update. ~200 ops.
    # 100 * 200 = 20,000 ops per run.
    # In Python, maybe 1-2 ms per run?
    # 50s / 0.002s = 25,000 runs possible.
    
    while time.time() < deadline:
        order = list(range(1, 101))
        random.shuffle(order)
        candidate = greedy(order)
        if len(candidate) > best_len:
            best = candidate
            best_len = len(candidate)
            
    return best
