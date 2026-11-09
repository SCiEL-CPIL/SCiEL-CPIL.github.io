import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # 1. Generate an initial maximal Sidon set using Greedy
    # This is the baseline from the seed
    selected, sums = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    
    # The greedy set is maximal by construction?
    # Yes, because we iterate 1..N. If x wasn't added, it was blocked by existing sums.
    # So [1..N] is covered.
    
    best_set = selected[:]
    
    # 2. Pruning Loop
    # We try to remove elements while maintaining maximality.
    
    def is_maximal(S):
        # Check if S is Sidon (should be true if derived from Sidon)
        # Check if every x not in S is blocked
        S_set = set(S)
        sums_S = set()
        for a in S:
            for b in S:
                if a <= b:
                    sums_S.add(a + b)
        
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Check if x is blocked
            # Blocked if 2x in sums_S OR (x + S) intersects sums_S
            if 2 * x in sums_S:
                continue
            blocked = False
            for y in S:
                if x + y in sums_S:
                    blocked = True
                    break
            if not blocked:
                return False
        return True

    # Optimization: is_maximal is O(N * |S|^2).
    # |S| ~ 16, N=256. 256 * 256 = 65536 ops. Very fast.
    # We can call it frequently.
    
    # Better: Incremental check?
    # For now, simple full check is fine if we don't do too many iterations.
    
    # Let's do a randomized restart pruning
    # We can also try to swap elements to find a smaller set?
    # Pruning only reduces size.
    
    for restart in range(100): # Try multiple random orders/removals
        if time.time() > deadline:
            break
        
        # Start from the greedy set or a perturbed version?
        # Let's stick to greedy set as base, but randomize removal order.
        S = best_set[:]
        
        # Randomize order of elements to remove
        order = S[:]
        random.shuffle(order)
        
        # Try to remove each element
        for s in order:
            if time.time() > deadline:
                break
            
            if s not in S:
                continue
                
            S_temp = [x for x in S if x != s]
            
            if is_maximal(S_temp):
                S = S_temp
                if len(S) < len(best_set):
                    best_set = S[:]
        
        # Optional: Try local swaps to escape local minima of size?
        # If we can't remove more, maybe we can swap a large element for a smaller one?
        # Size doesn't change. We care about size.
        # So only removals matter.
        
    return best_set
