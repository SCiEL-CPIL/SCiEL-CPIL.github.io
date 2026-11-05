"""Optimized 4-AP-free set construction using simulated annealing with bit-parallel conflicts."""
import time
import random
import math

N = 512

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Precompute for each x: list of "other triples" - for each 4-AP containing x,
    # store the tuple of the other 3 elements
    x_to_aps = [[] for _ in range(N+1)]
    
    for a in range(1, N+1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            ap = (a, a+d, a+2*d, a+3*d)
            for j, x in enumerate(ap):
                others = tuple(ap[k] for k in range(4) if k != j)
                x_to_aps[x].append(others)
    
    # Start with greedy solution
    selected = set()
    for x in range(1, N+1):
        safe = True
        for others in x_to_aps[x]:
            if all(o in selected for o in others):
                safe = False
                break
        if safe:
            selected.add(x)
    
    best_set = set(selected)
    best_size = len(selected)
    
    # Simulated annealing with add/remove/swap moves
    T = 100.0
    cooling = 0.9998
    
    random.seed(123)
    
    while time.time() < deadline:
        if T < 0.001:
            T = 100.0
        
        # Choose move type
        r = random.random()
        if r < 0.4:
            # Add a random unselected element
            unselected = [x for x in range(1, N+1) if x not in selected]
            if not unselected:
                T *= cooling
                continue
            x = random.choice(unselected)
            # Check conflicts
            conflict_count = 0
            for others in x_to_aps[x]:
                if all(o in selected for o in others):
                    conflict_count += 1
            if conflict_count == 0:
                selected.add(x)
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_set = set(selected)
            else:
                # Worsening: accept with prob exp(-conflict_count/T)
                if random.random() < math.exp(-conflict_count / T):
                    selected.add(x)
        elif r < 0.7:
            # Remove a random selected element
            if not selected:
                T *= cooling
                continue
            x = random.choice(list(selected))
            selected.discard(x)
            # This is a worsening move (size decreases by 1)
            if random.random() >= math.exp(-1.0 / T):
                selected.add(x)  # revert
        else:
            # Swap: remove one selected, add one unselected
            if not selected:
                T *= cooling
                continue
            unselected = [x for x in range(1, N+1) if x not in selected]
            if not unselected:
                T *= cooling
                continue
            x_remove = random.choice(list(selected))
            x_add = random.choice(unselected)
            # Check if swap is valid (no new 4-APs created)
            # Temporarily remove x_remove, check if x_add can be added
            selected.discard(x_remove)
            conflict = False
            for others in x_to_aps[x_add]:
                if all(o in selected for o in others):
                    conflict = True
                    break
            if not conflict:
                selected.add(x_add)
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_set = set(selected)
            else:
                # Revert swap
                selected.add(x_remove)
                selected.discard(x_add)
        
        T *= cooling
    
    return sorted(best_set)
