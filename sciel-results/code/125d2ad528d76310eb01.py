"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    n = 256
    best_total = 0
    best_A = None
    best_B = None
    
    # Use time to ensure we don't exceed limits, but aim for high restarts
    start_time = time.time()
    max_time = 50.0 # Leave buffer for evaluation overhead
    
    # Increase restarts significantly
    num_seeds = 2000
    
    for seed in range(num_seeds):
        if time.time() - start_time > max_time:
            break
            
        rng = random.Random(seed)
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        elements = list(range(1, n+1))
        rng.shuffle(elements)
        
        for x in elements:
            # Try adding to A
            can_A = True
            # Check sums
            if x + x in sums_A:
                can_A = False
            else:
                for s in A:
                    if x + s in sums_A:
                        can_A = False
                        break
            # Check differences against B
            if can_A:
                for s in A:
                    d = x - s
                    if d in diffs_B or -d in diffs_B:
                        can_A = False
                        break
            
            # Try adding to B
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
                    d = x - s
                    if d in diffs_A or -d in diffs_A:
                        can_B = False
                        break
            
            if can_A and can_B:
                # Randomly choose
                if rng.random() < 0.5:
                    A.append(x)
                    sums_A.add(x + x)
                    for s in A[:-1]:
                        sums_A.add(x + s)
                        d = x - s
                        diffs_A.add(d)
                        diffs_A.add(-d)
                else:
                    B.append(x)
                    sums_B.add(x + x)
                    for s in B[:-1]:
                        sums_B.add(x + s)
                        d = x - s
                        diffs_B.add(d)
                        diffs_B.add(-d)
            elif can_A:
                A.append(x)
                sums_A.add(x + x)
                for s in A[:-1]:
                    sums_A.add(x + s)
                    d = x - s
                    diffs_A.add(d)
                    diffs_A.add(-d)
            elif can_B:
                B.append(x)
                sums_B.add(x + x)
                for s in B[:-1]:
                    sums_B.add(x + s)
                    d = x - s
                    diffs_B.add(d)
                    diffs_B.add(-d)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
            
    return [best_A, best_B]
# EVOLVE-BLOCK-END
