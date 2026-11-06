import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # Safety margin: kill at 90s, leave 3s buffer
    
    N = 100
    target = 200
    mask = (1 << (target + 1)) - 1

    def compute_reachable(S):
        r = 1
        for x in S:
            r = (r | (r << x)) & mask
        return r

    def is_valid(S):
        r = compute_reachable(S)
        return ((r >> target) & 1) == 0

    def is_valid_with_reachable(reachable):
        return ((reachable >> target) & 1) == 0

    # 1. Generate initial candidates using Randomized Greedy with incremental reachability
    best_set = set()
    best_size = 0
    
    # Base set from original code is a good deterministic start
    base_S = set(range(66, 101))
    if is_valid(base_S):
        best_set = base_S
        best_size = len(base_S)
        
    candidates = list(range(1, 101))
    
    # Run several randomized greedy trials with incremental reachability
    random.seed(42)
    num_trials = 200
    for i in range(num_trials):
        if time.time() > deadline - 15: break # Leave time for local search
        
        S_try = set()
        reachable = 1  # Start with sum 0 reachable
        
        # Random order
        order = candidates[:]
        random.shuffle(order)
        
        for x in order:
            # Check if adding x keeps target unreachable using incremental reachability
            new_reachable = (reachable | (reachable << x)) & mask
            if is_valid_with_reachable(new_reachable):
                S_try.add(x)
                reachable = new_reachable
                
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try

    # 2. Local Search Refinement with incremental reachability
    current_set = set(best_set)
    current_size = len(current_set)
    
    # Precompute reachable for current set
    current_reachable = compute_reachable(current_set)
    
    no_improvement_count = 0
    max_no_improvement = 500
    
    while time.time() < deadline - 2:
        improved = False
        
        # Try to add an element (try all, pick best valid one)
        not_in_set = [x for x in candidates if x not in current_set]
        
        # Try additions first - this is the key to increasing size
        added = False
        for x in not_in_set:
            new_reachable = (current_reachable | (current_reachable << x)) & mask
            if is_valid_with_reachable(new_reachable):
                current_set.add(x)
                current_reachable = new_reachable
                current_size += 1
                improved = True
                added = True
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
                break  # Try one addition at a time
        
        # If no addition worked, try swaps
        if not improved and current_set and not_in_set:
            # Try multiple random swaps to escape local optima
            for _ in range(10):
                if time.time() > deadline - 2:
                    break
                u = random.choice(list(current_set))
                v = random.choice(not_in_set)
                if u != v:
                    # Compute reachable after removing u and adding v
                    # This requires recomputing since removal is hard to invert
                    new_set = (current_set - {u}) | {v}
                    if is_valid(new_set):
                        # Accept swap to explore different regions
                        current_set = new_set
                        current_reachable = compute_reachable(current_set)
                        improved = True
                        break
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Try a fresh greedy from scratch with different seed
                if time.time() > deadline - 10:
                    break
                # Restart with a new random greedy
                S_try = set()
                reachable = 1
                order = candidates[:]
                random.shuffle(order)
                for x in order:
                    new_reachable = (reachable | (reachable << x)) & mask
                    if is_valid_with_reachable(new_reachable):
                        S_try.add(x)
                        reachable = new_reachable
                
                if len(S_try) >= current_size:
                    current_set = S_try
                    current_reachable = compute_reachable(current_set)
                    current_size = len(current_set)
                    no_improvement_count = 0
                else:
                    break
        else:
            no_improvement_count = 0
        
    return list(best_set)
