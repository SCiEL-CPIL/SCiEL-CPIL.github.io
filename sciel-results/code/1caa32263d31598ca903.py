import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Precompute reciprocals
    recips = {i: Fraction(1, i) for i in range(1, 25)}
    
    def can_add(S, x):
        # S is a set of ints, x is an int not in S
        # 1. Check if recips[x] is a subsum of recips of S
        target = recips[x]
        sums = {Fraction(0)}
        for y in S:
            ry = recips[y]
            new_sums = set()
            for s in sums:
                ns = s + ry
                if ns <= target:
                    new_sums.add(ns)
            sums.update(new_sums)
            if target in sums:
                return False
        
        # 2. Check if adding x breaks validity of y in S
        # For y in S, if y < x, check if recips[y] - recips[x] is a subsum of S \ {y}
        for y in S:
            if y >= x:
                continue
            target_diff = recips[y] - recips[x]
            if target_diff <= 0:
                continue
            
            sums = {Fraction(0)}
            for z in S:
                if z == y:
                    continue
                rz = recips[z]
                new_sums = set()
                for s in sums:
                    ns = s + rz
                    if ns <= target_diff:
                        new_sums.add(ns)
                sums.update(new_sums)
                if target_diff in sums:
                    return False
        return True

    S = set()
    
    # Greedy Addition: Large to Small
    for x in range(24, 0, -1):
        if time.time() > deadline:
            break
        if can_add(S, x):
            S.add(x)
            
    best_S = set(S)
    best_size = len(S)
    
    # Local Search: Try to add elements
    # If we can't add any, try swaps to enable future additions
    # We will iterate until deadline or no improvement
    
    # To speed up, we can cache the result of can_add? No, S changes.
    
    while time.time() < deadline:
        added = False
        # Try to add any element
        for x in range(1, 25):
            if x in S:
                continue
            if can_add(S, x):
                S.add(x)
                added = True
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                break # Restart adding loop
                
        if not added:
            # Try swaps
            # We pick a random x not in S and random y in S
            # If S - {y} + {x} is valid, we do it.
            # This doesn't increase size, but might change the set to one where an element CAN be added.
            
            # Optimization: Only do swaps if we have time.
            # Each swap check is is_valid(S_new) which is expensive.
            # Let's limit swaps.
            
            # Instead of full is_valid, we can just check if the swap creates a valid set.
            # But is_valid is O(N * 2^N).
            # Maybe we can just try to add elements in different orders?
            
            # Let's try a simple heuristic: 
            # If we are stuck, remove the largest element and try to add smaller ones?
            # Or just accept the current best.
            
            # Given the time limit, and that greedy already finds 19, 
            # and 1.583 is the known best, maybe 19 is the max or close to it.
            # Let's do a few random restarts of the greedy with different orders?
            # No, time is tight.
            
            # Let's try to add elements from 1 to 24 in the local search loop.
            # If we can't add, we break out of the while loop?
            # No, maybe a swap helps.
            
            # Let's do limited swaps.
            import random
            for _ in range(5):
                if time.time() > deadline:
                    break
                x = random.choice([i for i in range(1, 25) if i not in S])
                y = random.choice(list(S))
                S_new = set(S)
                S_new.remove(y)
                S_new.add(x)
                
                # Check validity of S_new
                # We can reuse can_add logic? 
                # can_add checks if x can be added to S.
                # Here we are removing y first.
                # So we check if x can be added to S \ {y}.
                # And we also need to ensure S \ {y} is valid (it is, subset of valid set).
                # So we just need to check if x can be added to S \ {y}.
                
                if can_add(S_new, x):
                    # Wait, can_add(S_new, x) checks if x can be added to S_new.
                    # S_new is S \ {y}.
                    # If can_add returns True, then S_new U {x} is valid.
                    S = S_new
                    # Check if we can add more
                    added = False
                    for z in range(1, 25):
                        if z in S:
                            continue
                        if can_add(S, z):
                            S.add(z)
                            added = True
                            if len(S) > best_size:
                                best_S = set(S)
                                best_size = len(S)
                            break
                    if added:
                        break # Restart outer while
    
    return list(best_S)
