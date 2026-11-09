"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    n = 256
    best_total = 0
    best_A = None
    best_B = None
    
    # Run multiple trials with different random seeds and biases
    for trial in range(200):
        rng = random.Random(trial)
        
        # Bias towards A or B
        # Alternate bias or randomize it
        bias = 0.5 + (trial % 10) * 0.05 - 0.25 # Range 0.25 to 0.75
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Separate candidate lists
        list_A = list(range(1, n + 1))
        list_B = list(range(1, n + 1))
        rng.shuffle(list_A)
        rng.shuffle(list_B)
        
        while list_A or list_B:
            moved = False
            
            # Decide which set to try first
            # If one list is empty, try the other
            if not list_A:
                try_B = True
            elif not list_B:
                try_B = False
            else:
                try_B = rng.random() > bias
            
            if try_B:
                # Try to add from B
                if list_B:
                    x = list_B.pop()
                    # Check validity for B
                    can_B = True
                    if x + x in sums_B:
                        can_B = False
                    else:
                        for s in B:
                            if x + s in sums_B:
                                can_B = False
                                break
                    if can_B:
                        for s in B:
                            d1 = x - s
                            d2 = s - x
                            if d1 in diffs_A or d2 in diffs_A:
                                can_B = False
                                break
                    
                    if can_B:
                        B.append(x)
                        sums_B.add(x + x)
                        for s in B[:-1]:
                            sums_B.add(x + s)
                            diffs_B.add(x - s)
                            diffs_B.add(s - x)
                        moved = True
                    # If invalid, x is discarded (removed from list_B already)
            else:
                # Try to add from A
                if list_A:
                    x = list_A.pop()
                    # Check validity for A
                    can_A = True
                    if x + x in sums_A:
                        can_A = False
                    else:
                        for s in A:
                            if x + s in sums_A:
                                can_A = False
                                break
                    if can_A:
                        for s in A:
                            d1 = x - s
                            d2 = s - x
                            if d1 in diffs_B or d2 in diffs_B:
                                can_A = False
                                break
                    
                    if can_A:
                        A.append(x)
                        sums_A.add(x + x)
                        for s in A[:-1]:
                            sums_A.add(x + s)
                            diffs_A.add(x - s)
                            diffs_A.add(s - x)
                        moved = True
                    # If invalid, x is discarded (removed from list_A already)
            
            if not moved:
                # If we tried a pool and it was empty (or we didn't try it because other was empty?)
                # Actually, if list_A is not empty and we tried A but it was invalid, we popped it.
                # So the list shrinks.
                # If we tried B and it was empty, we would have gone to A?
                # Logic:
                # If list_A empty -> try_B True.
                # If list_B empty -> try_B False.
                # If both non-empty -> random.
                # If we pick a pool, we pop from it.
                # If valid, we add.
                # If invalid, we discard.
                # In both cases, the list shrinks.
                # So the loop will terminate.
                pass
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
            
    return [best_A, best_B]
# EVOLVE-BLOCK-END
