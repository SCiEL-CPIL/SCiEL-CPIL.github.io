"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    best_total = 0
    best_A = []
    best_B = []
    
    # We will try multiple random restarts
    # Each restart uses a different random ordering of candidates
    # and a randomized greedy selection with balancing
    
    # Initial seed values to ensure non-empty sets if search fails
    # A simple valid pair: A=[1], B=[2] -> diffs {0}, {0} (nonzero diffs empty) -> disjoint.
    # Actually nonzero diffs of [1] is empty. [1,2] has diff {1}.
    # Let's start with best as a known valid small set.
    best_A = [1]
    best_B = [2]
    best_total = 2
    
    # Candidate pool is [1, 256]
    candidates = list(range(1, 257))
    
    # Number of restarts is dynamic based on time
    # Estimate time per restart. A greedy pass is O(N * K^2) where K is set size.
    # N=256, K~30. 256 * 900 ~ 200k ops. Very fast in Python.
    # Let's just loop until deadline.
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Randomize the order of candidates to explore different parts of the space
        random.shuffle(candidates)
        
        A = []
        B = []
        
        # State tracking for fast validation
        # Sums for A: set of a+b
        # Sums for B: set of a+b
        # Diffs for A: set of a-b (nonzero)
        # Diffs for B: set of a-b (nonzero)
        
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # To balance sizes, we can alternate turns or pick randomly from {A, B}
        # Or just iterate through candidates and assign to A or B if possible.
        # Better: iterate through candidates, try to add to A, if fails try B, if fails skip.
        # Or: maintain a target size and balance.
        
        # Let's try a "round-robin" style with randomization.
        # For each candidate x in shuffled order:
        #   Try to add to A
        #   If A is significantly larger than B, try B first?
        #   Or just random side?
        
        # Strategy:
        # For each x:
        #   valid_A = check if x can be added to A
        #   valid_B = check if x can be added to B
        #   if valid_A and valid_B:
        #       choose side with smaller size (or random if equal)
        #   elif valid_A:
        #       add to A
        #   elif valid_B:
        #       add to B
        #   else:
        #       skip
        
        for x in candidates:
            # Check A
            valid_A = True
            # Sums check for A
            # New sums: x+a for a in A, and x+x
            # If any sum already in sums_A, invalid.
            # Also need to check if x+x collides with existing sums? 
            # sums_A contains all a+b for a,b in A.
            # Adding x: new pairs are (x,x) and (x,a) for a in A.
            # So check:
            # 1. x+x not in sums_A
            # 2. x+a not in sums_A for all a in A
            # 3. Diffs: x-a not in diffs_A and a-x not in diffs_A for all a in A.
            #    Also diffs_A must not intersect diffs_B.
            
            # Optimization: Precompute checks?
            # For small sets, direct check is fine.
            
            # Check Sums A
            if x + x in sums_A:
                valid_A = False
            else:
                for a in A:
                    if x + a in sums_A:
                        valid_A = False
                        break
            
            # Check Diffs A (internal)
            if valid_A:
                for a in A:
                    d = x - a
                    if d in diffs_A:
                        valid_A = False
                        break
                    if -d in diffs_A:
                        valid_A = False
                        break
            
            # Check Diffs A vs B (disjointness)
            if valid_A:
                # New diffs are x-a and a-x.
                # They must not be in diffs_B.
                for a in A:
                    d = x - a
                    if d in diffs_B:
                        valid_A = False
                        break
                    if -d in diffs_B:
                        valid_A = False
                        break
            
            # Check B
            valid_B = True
            if x + x in sums_B:
                valid_B = False
            else:
                for b in B:
                    if x + b in sums_B:
                        valid_B = False
                        break
            
            if valid_B:
                for b in B:
                    d = x - b
                    if d in diffs_B:
                        valid_B = False
                        break
                    if -d in diffs_B:
                        valid_B = False
                        break
            
            if valid_B:
                for b in B:
                    d = x - b
                    if d in diffs_A:
                        valid_B = False
                        break
                    if -d in diffs_A:
                        valid_B = False
                        break
            
            # Decision
            if valid_A and valid_B:
                # Balance: add to smaller set
                if len(A) < len(B):
                    # Add to A
                    sums_A.add(x + x)
                    diffs_A.update() # placeholder
                    for a in A:
                        sums_A.add(x + a)
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                    A.append(x)
                elif len(B) < len(A):
                    # Add to B
                    sums_B.add(x + x)
                    for b in B:
                        sums_B.add(x + b)
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
                    B.append(x)
                else:
                    # Random choice
                    if random.random() < 0.5:
                        sums_A.add(x + x)
                        for a in A:
                            sums_A.add(x + a)
                            diffs_A.add(x - a)
                            diffs_A.add(a - x)
                        A.append(x)
                    else:
                        sums_B.add(x + x)
                        for b in B:
                            sums_B.add(x + b)
                            diffs_B.add(x - b)
                            diffs_B.add(b - x)
                        B.append(x)
            elif valid_A:
                sums_A.add(x + x)
                for a in A:
                    sums_A.add(x + a)
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
                A.append(x)
            elif valid_B:
                sums_B.add(x + x)
                for b in B:
                    sums_B.add(x + b)
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
                B.append(x)
            
            # Update best
            current_total = len(A) + len(B)
            if current_total > best_total:
                best_total = current_total
                best_A = A.copy()
                best_B = B.copy()
        
        # If we are getting stuck at a low score, we might need a different strategy
        # But random restarts with shuffling should help.
        
        # Optional: If best_total is high, we can try to optimize further?
        # No, keep it simple.
        
        # Break if we hit the deadline
        if time.time() > deadline:
            break
            
    return [best_A, best_B]
# EVOLVE-BLOCK-END
