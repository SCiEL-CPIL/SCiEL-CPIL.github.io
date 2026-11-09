import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    time_limit = 50.0  # Safety margin
    
    # Precompute all 4-APs in [1, N]
    # Each AP is a tuple of 4 numbers
    # We'll store them as a list of tuples
    # For fast checking, we can use a set of frozensets? No, we need to check if S contains ALL elements of the AP.
    # Actually, we want to check if adding x creates a 4-AP.
    # This is equivalent to checking if there exists an AP containing x such that the other 3 elements are in S.
    
    # Let's precompute for each x, the list of APs it participates in.
    # ap_list[x] = list of tuples (y1, y2, y3) where {x, y1, y2, y3} is a 4-AP.
    
    ap_list = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                # AP: a, b, c, e
                ap_list[a].append((b, c, e))
                ap_list[b].append((a, c, e))
                ap_list[c].append((a, b, e))
                ap_list[e].append((a, b, c))
    
    # Helper: Check if adding x to S creates a 4-AP
    # S is a set of integers
    def can_add(S, x):
        for y1, y2, y3 in ap_list[x]:
            if y1 in S and y2 in S and y3 in S:
                return False
        return True

    best_set = set()
    best_size = 0
    
    # Run multiple random restarts
    while time.time() - start_time < time_limit:
        S = set()
        # Random order for greedy
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        # Local search: Try to improve by swapping
        # If we have time, do a few local search steps
        if time.time() - start_time < time_limit - 5.0:
            # Simple local search: try to replace one element with another
            # Or just try to add elements that were skipped?
            # The greedy already tried all elements in random order.
            # So S is maximal with respect to that order.
            # To improve, we need to remove some elements to allow adding others.
            
            # Heuristic: Remove the element that blocks the most potential additions?
            # Or just random removal and re-greedy?
            
            # Let's try a simple hill climbing:
            # 1. Pick a random element x in S to remove.
            # 2. Try to add elements from the complement in random order.
            # 3. If size increases, keep the change.
            
            # But this is expensive.
            # Given the time budget, maybe just stick to restarts but with a better heuristic for ordering?
            
            # Alternative: Use a weighted greedy.
            # Weight of x = 1 / (number of APs containing x that are not yet "blocked")?
            # Or just random restarts is enough?
            
            # Let's try to optimize the restart loop by using a different ordering strategy.
            # Instead of pure random, maybe "center-out" or "random with bias".
            
            # For now, let's just do more restarts if possible.
            # The precomputation should make can_add faster.
            
            pass

    return sorted(best_set)
