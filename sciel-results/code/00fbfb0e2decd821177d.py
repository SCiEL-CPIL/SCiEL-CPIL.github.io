import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 58.0  # Safety margin for 60s limit

    N = 256

    # 1. Precompute APs and reverse mapping
    ap_indices = []
    for d in range(1, N // 2 + 1):
        for start in range(1, N - d + 1):
            indices = list(range(start - 1, N, d))
            if len(indices) >= 1:
                ap_indices.append(indices)

    num_aps = len(ap_indices)

    # Reverse mapping: for each position i in 0..N-1, which APs contain it?
    aps_containing = [[] for _ in range(N)]
    for r, indices in enumerate(ap_indices):
        for c in indices:
            aps_containing[c].append(r)

    # Convert to numpy arrays for faster indexing
    # We need to store the AP indices for each position as a numpy array
    aps_containing_np = []
    for i in range(N):
        aps_containing_np.append(np.array(aps_containing[i], dtype=np.int32))

    def thue_morse(N):
        seq = []
        for i in range(N):
            count = bin(i).count('1')
            if count % 2 == 0:
                seq.append(1)
            else:
                seq.append(-1)
        return np.array(seq, dtype=np.int8)

    def evaluate_sequence(s):
        """
        Calculate the max absolute sum for all APs given a sequence s.
        """
        sums = np.zeros(num_aps, dtype=np.int32)
        for r, indices in enumerate(ap_indices):
            sums[r] = np.sum(s[indices])
        return np.max(np.abs(sums))

    def greedy_optimize(s_init, time_budget):
        """
        Perform Greedy Coordinate Descent with tie-breaking.
        """
        s = s_init.copy()
        
        # Initialize sums
        sums = np.zeros(num_aps, dtype=np.int32)
        for r, indices in enumerate(ap_indices):
            sums[r] = np.sum(s[indices])
            
        current_max = int(np.max(np.abs(sums)))
        
        best_s = s.copy()
        best_score = current_max
        
        rng = random.Random(123)
        
        indices_list = list(range(N))
        
        # Precompute the "cost" of each position? No, cost depends on current sums.
        
        while time.time() - start_time < time_budget:
            # Randomize order
            rng.shuffle(indices_list)
            
            improved = False
            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break
                
                old_val = s[i]
                delta_val = -2 * old_val
                affected_aps = aps_containing_np[i]
                
                if len(affected_aps) == 0:
                    continue

                # Save old sums for affected APs
                old_sums = sums[affected_aps].copy()
                
                # Apply flip
                sums[affected_aps] += delta_val
                s[i] = -old_val
                
                new_max = int(np.max(np.abs(sums)))
                
                if new_max < current_max:
                    current_max = new_max
                    if current_max < best_score:
                        best_score = current_max
                        best_s = s.copy()
                    improved = True
                elif new_max == current_max:
                    # Tie-breaking: accept if it reduces the "energy" (sum of squares of sums)
                    # or just accept with low probability to escape local minima.
                    # Calculating sum of squares is O(num_aps) which is expensive.
                    # Let's use a cheaper proxy: number of APs at max?
                    # Or just random acceptance.
                    if rng.random() < 0.05:
                        improved = True
                    else:
                        sums[affected_aps] = old_sums
                        s[i] = old_val
                else:
                    sums[affected_aps] = old_sums
                    s[i] = old_val
            
            if not improved:
                break
                
        return best_s, best_score

    # Main Loop
    best_global_s = None
    best_global_score = float('inf')
    
    rng_global = random.Random(42)
    
    # Initial candidates
    s_thue = thue_morse(N)
    # Try a few random inits
    s_random1 = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
    s_random2 = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
    
    inits = [s_thue, s_random1, s_random2]
    
    # We can also try a simple heuristic: alternating signs?
    s_alt = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.int8)
    inits.append(s_alt)
    
    # Sort inits by initial score?
    # Evaluating 4 inits takes ~4 * 0.1s = 0.4s. Worth it.
    init_scores = []
    for s in inits:
        score = evaluate_sequence(s)
        init_scores.append(score)
    
    # Sort inits by score
    sorted_inits = sorted(zip(init_scores, inits), key=lambda x: x[0])
    
    restarts = 0
    while time.time() - start_time < time_limit:
        if restarts < len(sorted_inits):
            s_init = sorted_inits[restarts][1]
        else:
            # Perturb the best found so far
            if best_global_s is not None:
                s_init = best_global_s.copy()
                # Flip a few random positions
                for _ in range(5):
                    idx = rng_global.randint(0, N-1)
                    s_init[idx] = -s_init[idx]
            else:
                s_init = np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
            
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1.0:
            break
        
        # Allocate time to this run
        # If it's the first few runs, give them more time?
        # Let's just give 80% of remaining time to the current run, but cap it?
        # No, let's just run until time is up or local min.
        # The greedy loop breaks on local min.
        # So we just pass the remaining time as the budget.
        run_budget = remaining * 0.9
        
        s_best_run, score_best_run = greedy_optimize(s_init, run_budget)
        
        if score_best_run < best_global_score:
            best_global_score = score_best_run
            best_global_s = s_best_run
            
        restarts += 1
        
        if best_global_score <= 1:
            break

    if best_global_s is None:
        best_global_s = thue_morse(N)
        
    return [int(x) for x in best_global_s]
