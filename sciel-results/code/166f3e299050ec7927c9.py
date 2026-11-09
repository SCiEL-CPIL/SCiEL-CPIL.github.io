import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    time_limit = 50.0  # Safety margin
    
    # Precompute all 4-APs in [1, N]
    # For each element x, we want to know which APs it participates in.
    # Specifically, for can_add(S, x), we need to check if there exists an AP (x, y1, y2, y3)
    # such that y1, y2, y3 are all in S.
    
    # We represent the set S as a bitmask (integer) for fast membership checks.
    # bit i is 1 if i is in S.
    
    # Precompute for each x, a list of "constraints".
    # A constraint is a tuple of 3 bitmasks? No, that's not right.
    # A constraint is a tuple (y1, y2, y3).
    # We need to check: (S & (1<<y1)) != 0 AND (S & (1<<y2)) != 0 AND (S & (1<<y3)) != 0.
    # This is still 3 operations per constraint.
    
    # Can we precompute a "conflict mask"?
    # No, because the conflict depends on which 3 specific elements are present.
    
    # However, we can optimize the inner loop by using bitwise operations on the whole mask.
    # Let's precompute for each x, the list of (y1, y2, y3) tuples.
    
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
    
    # Helper: Check if adding x to S (bitmask) creates a 4-AP
    # S is an integer bitmask
    def can_add(S, x):
        for y1, y2, y3 in ap_list[x]:
            # Check if y1, y2, y3 are all in S
            # Using bit shifts
            if (S >> y1) & 1 and (S >> y2) & 1 and (S >> y3) & 1:
                return False
        return True

    best_mask = 0
    best_size = 0
    
    # Run multiple random restarts
    while time.time() - start_time < time_limit:
        S = 0
        # Random order for greedy
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(S, x):
                S |= (1 << x)
        
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_mask = S
            
        # Local search: Try to improve by swapping
        # We have time, so let's do a few local search steps.
        # Strategy: Try to find an element x NOT in S that can be added if we remove some element y from S.
        # Or simpler: Randomly remove an element from S, then try to re-add elements (including the removed one and others) in random order?
        # This is essentially a "shake" operation.
        
        # More targeted: 
        # 1. Identify "blocked" elements: elements not in S that are blocked by some subset of S.
        # 2. Pick a random blocked element x.
        # 3. Find the APs blocking x.
        # 4. Pick one of the blocking APs.
        # 5. Try to remove one element from that AP (that is in S) and see if x can be added.
        #    If size doesn't decrease (i.e., we add x and maybe others), keep it.
        
        # Given the complexity, let's stick to a simpler local search:
        # Hill Climbing: 
        # 1. Pick a random element x in S to remove.
        # 2. Create S' = S \ {x}.
        # 3. Try to add elements from (complement of S') in random order.
        # 4. If |S'| > |S|, accept.
        
        # This is O(N * len(ap_list)) per step. 
        # With N=512, and ~10 APs per element, this is ~5000 ops.
        # 50 seconds / 5000 ops = 10000 steps.
        # This is feasible.
        
        if time.time() - start_time < time_limit - 5.0:
            # Local Search
            S_local = S
            improved = True
            while improved and (time.time() - start_time < time_limit - 5.0):
                improved = False
                # Try a few random moves
                for _ in range(10):
                    if time.time() - start_time >= time_limit - 5.0:
                        break
                    
                    # Pick a random element to remove from S_local
                    elements_in_S = [i for i in range(1, N + 1) if (S_local >> i) & 1]
                    if not elements_in_S:
                        break
                    x_remove = random.choice(elements_in_S)
                    
                    S_trial = S_local & ~(1 << x_remove)
                    
                    # Try to add elements from the complement in random order
                    # We want to see if we can gain more than we lost (1 element).
                    # So if we can add at least 1 element, we break even.
                    # If we can add > 1, we improve.
                    # Actually, we just want to maximize size.
                    # Let's just re-greedy the complement?
                    # No, that's expensive.
                    # Let's just try to add the removed element back? No, that's trivial.
                    # Let's try to add random elements from the complement.
                    
                    # Better: Try to add ALL elements from the complement that can be added.
                    # This is a "greedy fill" step.
                    # But we only removed 1 element. The number of new elements we can add is likely small.
                    
                    # Let's just try to add the elements that were blocked by x_remove?
                    # We don't track that easily.
                    
                    # Simple heuristic:
                    # 1. Remove x_remove.
                    # 2. Try to add x_remove back? No.
                    # 3. Try to add random elements from the complement.
                    
                    # Let's try adding elements in random order, but stop if size increases.
                    # Actually, let's just do a full greedy pass on the complement?
                    # That's O(N^2) worst case.
                    
                    # Let's stick to: Remove x, try to add x back (fail), try to add random other elements.
                    # This is weak.
                    
                    # Alternative: 
                    # 1. Remove x.
                    # 2. Run a quick greedy fill on the complement.
                    # 3. Compare sizes.
                    
                    # Let's implement a quick greedy fill.
                    # We only care if we can get a larger set.
                    
                    # Optimization: Only try to add elements that are "close" to x_remove?
                    # No, let's just random sample from complement.
                    
                    # Let's try a different local search:
                    # "Swap" search.
                    # Pick x in S, y not in S.
                    # Remove x, add y.
                    # Check if valid.
                    # This is O(N^2 * check_cost).
                    # N^2 = 262,144. check_cost ~ 10 ops. Total ~ 2.6M ops per full sweep.
                    # 50s / 2.6M ~ 20 sweeps.
                    # This is feasible.
                    
                    # Let's do a random swap search.
                    x_remove = random.choice(elements_in_S)
                    y_add = random.choice([i for i in range(1, N + 1) if not ((S_local >> i) & 1)])
                    
                    # Try swap: remove x_remove, add y_add
                    S_trial = S_local & ~(1 << x_remove)
                    if can_add(S_trial, y_add):
                        S_trial |= (1 << y_add)
                        # Check if this swap allows adding more?
                        # For now, just accept if valid.
                        # But size is same. We need to improve size.
                        # So we need to find a sequence of swaps that increases size?
                        # Or just accept the swap and then try to add more?
                        
                        # Let's accept the swap and then try to greedily add more elements.
                        # This is getting complicated.
                        
                        # Let's go back to: Remove x, then greedy fill.
                        # But limit the greedy fill to a few iterations?
                        
                        # Let's just do the "Remove x, Greedy Fill" approach.
                        # It's the most robust.
                        
                        pass # Placeholder, will implement below

            # Let's implement the "Remove x, Greedy Fill" properly.
            # We need to be careful with time.
            
            pass # Will replace with actual implementation

    # Convert best_mask to list
    best_set = [i for i in range(1, N + 1) if (best_mask >> i) & 1]
    return sorted(best_set)
