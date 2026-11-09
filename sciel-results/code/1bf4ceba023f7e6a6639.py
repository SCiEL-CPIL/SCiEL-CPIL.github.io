import random
import time

def construct():
    N = 512
    best_set = set()
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    def is_4ap_free(S):
        # Optimization: Check only if needed? 
        # For greedy, we check incrementally.
        pass

    def greedy_greedy(order):
        S = set()
        for x in order:
            # Check if x creates a 4-AP with elements in S
            # A 4-AP ending at x: x-3d, x-2d, x-d in S
            # A 4-AP starting at x: x, x+d, x+2d, x+3d (not possible since x is max so far? No, order is arbitrary)
            # If order is random, x is not necessarily the max.
            # So we must check all 4 positions.
            # 1. x is 1st: x, x+d, x+2d, x+3d in S? No, x is being added.
            #    So check if x+d, x+2d, x+3d are in S.
            # 2. x is 2nd: x-d, x, x+d, x+2d. Check if x-d, x+d, x+2d in S.
            # 3. x is 3rd: x-2d, x-d, x, x+d. Check if x-2d, x-d, x+d in S.
            # 4. x is 4th: x-3d, x-2d, x-d, x. Check if x-3d, x-2d, x-d in S.
            
            valid = True
            # Max d such that terms are in [1, N]
            # We can just iterate d from 1 to N//3
            for d in range(1, N//3 + 1):
                # Check all 4 alignments
                # 1. x is start
                if x + 3*d <= N:
                    if (x+d in S) and (x+2*d in S) and (x+3*d in S):
                        valid = False
                        break
                # 2. x is 2nd
                if x - d >= 1 and x + 2*d <= N:
                    if (x-d in S) and (x+d in S) and (x+2*d in S):
                        valid = False
                        break
                # 3. x is 3rd
                if x - 2*d >= 1 and x + d <= N:
                    if (x-2*d in S) and (x-d in S) and (x+d in S):
                        valid = False
                        break
                # 4. x is 4th
                if x - 3*d >= 1:
                    if (x-3*d in S) and (x-2*d in S) and (x-d in S):
                        valid = False
                        break
            
            if valid:
                S.add(x)
        return S

    # Run multiple random greedy restarts
    elements = list(range(1, N+1))
    best_size = 0
    
    for i in range(1000):
        if time.time() > deadline:
            break
        random.shuffle(elements)
        S = greedy_greedy(elements)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Local Search: Try to add any remaining valid elements
    # This is safe and fast
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N+1):
            if x not in best_set:
                # Check if x can be added to best_set
                valid = True
                for d in range(1, N//3 + 1):
                    if x + 3*d <= N and (x+d in best_set) and (x+2*d in best_set) and (x+3*d in best_set):
                        valid = False; break
                    if x - d >= 1 and x + 2*d <= N and (x-d in best_set) and (x+d in best_set) and (x+2*d in best_set):
                        valid = False; break
                    if x - 2*d >= 1 and x + d <= N and (x-2*d in best_set) and (x-d in best_set) and (x+d in best_set):
                        valid = False; break
                    if x - 3*d >= 1 and (x-3*d in best_set) and (x-2*d in best_set) and (x-d in best_set):
                        valid = False; break
                if valid:
                    best_set.add(x)
                    changed = True
                    break # Restart loop to catch new possibilities? Or just continue?
                    # Breaking and restarting is safer for convergence
                    # But might be slow. Let's just continue and mark changed.
                    # Actually, adding x might block others, but not unblock them.
                    # So we can just continue.
                    
    return sorted(best_set)
