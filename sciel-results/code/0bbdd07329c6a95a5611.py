import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    n = 128
    targets = np.arange(128, 257)
    
    # Initial A: [64, 128]
    A = set(range(64, 129))
    
    # Initialize counts
    counts = np.zeros(257, dtype=int)
    A_list = sorted(A)
    for i in range(len(A_list)):
        for j in range(i, len(A_list)):
            counts[A_list[i] + A_list[j]] += 1
            
    # Helper to update counts and get metrics
    def update_counts(A, counts, delta):
        # delta is +1 for add, -1 for remove
        # This is slow if done from scratch. 
        # Better to maintain counts and update incrementally.
        pass

    # We will maintain `counts` and `covered_mask`
    # covered_mask[k] is True if counts[k] > 0
    
    def calculate_metrics(counts):
        peak = np.max(counts[128:257])
        uncovered = np.sum(counts[128:257] == 0)
        ss = np.sum(counts[128:257]**2)
        return peak, uncovered, ss

    curr_peak, curr_unc, curr_ss = calculate_metrics(counts)
    best_peak = curr_peak
    best_A = set(A)
    best_ss = curr_ss
    
    # Precompute neighbors? No, just random sampling.
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Strategy 1: Try to remove non-critical elements
        # Identify candidates for removal
        # An element x is removable if for all k, counts[k] - contrib(x, k) > 0 for k in targets?
        # Actually, if counts[k] > contrib(x,k) for all k, then k remains covered.
        # contrib(x, k) is 1 if there exists y in A\{x} such that x+y=k, plus 1 if 2x=k.
        
        # To speed up, we can just try random removals and check validity.
        # Validity check: 
        # 1. Compute new counts (incremental).
        # 2. Check if any target count becomes 0.
        
        A_list = list(A)
        random.shuffle(A_list)
        
        for x in A_list:
            if len(A) <= 1: break
            # Tentatively remove x
            # Update counts
            # We need to subtract pairs (x, y) for y in A, y != x, and (x, x)
            # Note: in our count array, (x, y) with x<y is counted once. (x,x) counted once.
            # When we built counts, we iterated i<=j.
            # So removing x:
            # - For each y in A, y > x: counts[x+y] -= 1
            # - If x in A: counts[2x] -= 1
            
            # But A changes. We need to be careful.
            # Let's just recalculate counts for the trial set?
            # Size of A is ~40-60. Pairs ~1500.
            # Recalculating from scratch is O(|A|^2). 1500 ops is fast.
            # 10000 trials/sec * 1500 ops = 15M ops/sec. Python might be slow.
            # NumPy vectorization helps for the sum, but the loop is Python.
            
            # Let's try incremental update.
            # Save old counts
            old_counts = counts.copy() # 257 ints, fast
            
            # Remove x from A
            A_trial = set(A)
            A_trial.remove(x)
            
            # Update counts for removal
            # Pairs involving x: (x, y) where y in A_trial (since x removed, y != x)
            # And (x, x)
            # Wait, if we remove x, we lose:
            # 1. Pair (x, x) -> sum 2x
            # 2. Pairs (x, y) for y in A_trial. Note: in the original set A, these were (min(x,y), max(x,y)).
            #    If y > x, pair was (x, y). If y < x, pair was (y, x).
            #    In the count array, it's just index x+y.
            #    So we subtract 1 from counts[x+y] for all y in A_trial.
            #    And subtract 1 from counts[2x].
            
            for y in A_trial:
                counts[x + y] -= 1
            counts[2*x] -= 1
            
            # Check validity and metrics
            if np.any(counts[128:257] == 0):
                # Invalid, restore
                counts[:] = old_counts
            else:
                new_peak = np.max(counts[128:257])
                new_ss = np.sum(counts[128:257]**2)
                
                if new_peak < curr_peak or (new_peak == curr_peak and new_ss < curr_ss):
                    # Accept
                    A = A_trial
                    curr_peak = new_peak
                    curr_ss = new_ss
                    improved = True
                    if curr_peak < best_peak:
                        best_peak = curr_peak
                        best_A = set(A)
                        best_ss = curr_ss
                    break # Restart loop with new A
                else:
                    # Reject, restore
                    counts[:] = old_counts
        
        if improved:
            continue
            
        # Strategy 2: Try Swaps
        if len(A) > 1:
            x = random.choice(list(A))
            candidates_out = [i for i in range(n+1) if i not in A]
            if candidates_out:
                y = random.choice(candidates_out)
                
                old_counts = counts.copy()
                
                A_trial = set(A)
                A_trial.remove(x)
                A_trial.add(y)
                
                # Remove x effects
                for z in A_trial: # A_trial has y, not x
                    # Wait, A_trial is the NEW set.
                    # We are removing x from OLD A and adding y.
                    # Pairs lost: (x, z) for z in OLD A \ {x}.
                    # Pairs gained: (y, z) for z in NEW A \ {y}.
                    
                    # This is getting complicated to do incrementally with sets.
                    # Given |A| is small, let's just recalculate counts for trial.
                    pass
                
                # Recalculate counts for A_trial
                counts_trial = np.zeros(257, dtype=int)
                A_list_trial = sorted(A_trial)
                for i in range(len(A_list_trial)):
                    ai = A_list_trial[i]
                    for j in range(i, len(A_list_trial)):
                        aj = A_list_trial[j]
                        counts_trial[ai + aj] += 1
                
                if np.any(counts_trial[128:257] == 0):
                    continue # Invalid
                
                new_peak = np.max(counts_trial[128:257])
                new_ss = np.sum(counts_trial[128:257]**2)
                
                if new_peak < curr_peak or (new_peak == curr_peak and new_ss < curr_ss):
                    A = A_trial
                    counts = counts_trial
                    curr_peak = new_peak
                    curr_ss = new_ss
                    improved = True
                    if curr_peak < best_peak:
                        best_peak = curr_peak
                        best_A = set(A)
                        best_ss = curr_ss
                    break
        
        if improved:
            continue
            
        # Strategy 3: Random Perturbation (2-swap)
        if len(A) > 2:
            x1, x2 = random.sample(list(A), 2)
            candidates_out = [i for i in range(n+1) if i not in A]
            if len(candidates_out) >= 2:
                y1, y2 = random.sample(candidates_out, 2)
                
                A_trial = set(A)
                A_trial.remove(x1)
                A_trial.remove(x2)
                A_trial.add(y1)
                A_trial.add(y2)
                
                counts_trial = np.zeros(257, dtype=int)
                A_list_trial = sorted(A_trial)
                for i in range(len(A_list_trial)):
                    ai = A_list_trial[i]
                    for j in range(i, len(A_list_trial)):
                        aj = A_list_trial[j]
                        counts_trial[ai + aj] += 1
                        
                if np.any(counts_trial[128:257] == 0):
                    continue
                
                new_peak = np.max(counts_trial[128:257])
                new_ss = np.sum(counts_trial[128:257]**2)
                
                if new_peak < curr_peak or (new_peak == curr_peak and new_ss < curr_ss):
                    A = A_trial
                    counts = counts_trial
                    curr_peak = new_peak
                    curr_ss = new_ss
                    improved = True
                    if curr_peak < best_peak:
                        best_peak = curr_peak
                        best_A = set(A)
                        best_ss = curr_ss

    return sorted(list(best_A))
