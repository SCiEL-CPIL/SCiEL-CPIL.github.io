import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 100
    target = 200
    
    # Helper to compute reachable sums up to target
    def compute_reachable(S, target):
        reachable = {0}
        for x in S:
            # Optimization: only add sums <= target
            new_sums = set()
            for s in reachable:
                ns = s + x
                if ns <= target:
                    new_sums.add(ns)
            reachable |= new_sums
        return reachable

    # Base set: {66, ..., 100}
    # Min sum of 3 elements: 66+67+68 = 201 > 200
    # No 2 elements sum to 200 (max pair 99+100=199)
    # So this set is valid.
    base_S = set(range(66, 101))
    reachable = compute_reachable(base_S, target)
    
    if target in reachable:
        # Fallback, though shouldn't happen
        return list(base_S)
        
    best_set = set(base_S)
    best_size = len(base_S)
    
    candidates = list(range(1, 66))
    
    # Try different orders to maximize cardinality
    orders = [
        candidates,                    # Ascending
        sorted(candidates, reverse=True), # Descending
    ]
    
    # Add a few random shuffles
    random.seed(42)
    for _ in range(20):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    for order in orders:
        if time.time() > deadline:
            break
            
        S_try = set(base_S)
        reachable_try = set(reachable)
        
        for x in order:
            if x in S_try:
                continue
                
            # Check if adding x creates a sum of 200
            # New sums are s+x for s in reachable_try
            # If any s+x == 200, i.e., s == 200-x, then invalid
            if (target - x) in reachable_try:
                continue
                
            # Also need to ensure that adding x doesn't allow combinations 
            # with previously added x's in this trial to sum to 200.
            # The reachable_try set already accounts for all subsets of S_try.
            # So if (target - x) is not in reachable_try, then no subset of S_try 
            # sums to target - x, so no subset of S_try U {x} sums to target.
            
            S_try.add(x)
            # Update reachable sums
            # We only care about sums <= target
            new_sums = set()
            for s in reachable_try:
                ns = s + x
                if ns <= target:
                    new_sums.add(ns)
            reachable_try |= new_sums
            
            # Optimization: if target is reached, we would have skipped x, 
            # so target should never be in reachable_try.
            
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try
            
    return list(best_set)
