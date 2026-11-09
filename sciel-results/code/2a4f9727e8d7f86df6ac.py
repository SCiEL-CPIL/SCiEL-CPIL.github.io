import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_set = None
    best_diameter = float('inf')
    
    # Helper to check validity and get diffs
    def get_info(A):
        if len(A) < 2: return None
        diffs = [y - x for i, x in enumerate(A) for y in A[i+1:]]
        diff_set = set(diffs)
        duplicates = len(diffs) - len(diff_set)
        missing = sum(1 for d in range(1, 31) if d not in diff_set)
        return diffs, diff_set, duplicates, missing

    # 1. Generate initial valid sets using greedy with randomization
    # We will try to construct sets of size n=9 to 12
    # For each size, run a few greedy attempts or use SA
    
    # Let's use SA for each size n from 9 to 15
    for n in range(9, 16):
        if time.time() > deadline: break
        
        # Initialize random set of size n containing 0
        # To make it likely to be close to valid, we can start with a greedy set of size n
        # or just random.
        
        # Let's try a few random restarts for this size n
        for restart in range(50):
            if time.time() > deadline: break
            
            # Initial state
            # Start with 0 and n-1 random numbers
            # To help convergence, maybe start with a "spread out" set
            # e.g., 0, 1, 3, 7, 15... (doubled)
            # Or just random.
            
            # Let's use a greedy initial guess for this specific n
            # Greedy for size n:
            A = [0]
            # Force size n by adding elements greedily to cover diffs, 
            # but we might exceed n. 
            # Instead, let's just randomize.
            
            # Random init
            A = [0] + random.sample(range(1, 500), n-1)
            A.sort()
            
            # Simulated Annealing
            T = 100.0
            T_min = 1.0
            cooling = 0.995
            max_iter = 5000
            
            curr_A = A
            curr_cost = float('inf')
            
            # Calculate initial cost
            info = get_info(curr_A)
            if info:
                _, _, dups, miss = info
                curr_cost = dups * 1000 + miss * 100 + max(curr_A)
            else:
                curr_cost = 1000000
            
            best_local_A = curr_A
            best_local_cost = curr_cost
            
            for i in range(max_iter):
                if time.time() > deadline: break
                
                # Generate new state
                # Pick an index (not 0) and change value
                idx = random.randint(1, n-1)
                # New value: random in [0, max(curr_A) + 10]
                # Bias towards smaller values to minimize diameter
                max_val = max(curr_A)
                new_val = random.randint(0, max_val + 10)
                
                if new_val in curr_A:
                    continue
                
                new_A = curr_A.copy()
                new_A[idx] = new_val
                new_A.sort()
                
                # Check if 0 is still 0 (it is, we didn't touch index 0)
                
                info = get_info(new_A)
                if not info:
                    new_cost = 1000000
                else:
                    _, _, dups, miss = info
                    new_cost = dups * 1000 + miss * 100 + max(new_A)
                
                delta = new_cost - curr_cost
                if delta < 0:
                    curr_A = new_A
                    curr_cost = new_cost
                    if curr_cost < best_local_cost:
                        best_local_cost = curr_cost
                        best_local_A = curr_A
                else:
                    if random.random() < pow(2.718, -delta / T):
                        curr_A = new_A
                        curr_cost = new_cost
                
                T *= cooling
                if T < T_min:
                    T = T_min
            
            # After SA, check if we found a valid set
            info = get_info(best_local_A)
            if info:
                _, _, dups, miss = info
                if dups == 0 and miss == 0:
                    diam = max(best_local_A)
                    if diam < best_diameter:
                        best_diameter = diam
                        best_set = best_local_A

    # If no valid set found via SA, fallback to seed's greedy (modified to be fast)
    if best_set is None:
        # Use the seed's logic as a fallback
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            found = False
            # Try adding one element
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                # Try adding two elements
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        best_set = a
        best_diameter = max(best_set)

    return best_set
