import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin (90s total, leave buffer)
    
    N = 256
    
    # Precompute APs
    ap_indices = []
    for d in range(1, N // 2 + 1):
        for start in range(1, N - d + 1):
            indices = list(range(start - 1, N, d))
            if len(indices) >= 1:
                ap_indices.append(indices)
                
    num_aps = len(ap_indices)
    
    # Build sparse incidence structure
    # For each position i, store which APs contain it
    aps_containing = [[] for _ in range(N)]
    for r, indices in enumerate(ap_indices):
        for c in indices:
            aps_containing[c].append(r)
            
    # Convert to numpy arrays for faster access
    aps_containing_arr = []
    for i in range(N):
        aps_containing_arr.append(np.array(aps_containing[i], dtype=np.int32))
    
    def compute_sums(s):
        """Compute all AP sums given sign sequence s."""
        sums = np.zeros(num_aps, dtype=np.int32)
        for r, indices in enumerate(ap_indices):
            sums[r] = np.sum(s[indices])
        return sums
    
    def max_abs_sum(sums):
        return np.max(np.abs(sums))
    
    # Simulated annealing
    def simulated_annealing(s_init, time_budget):
        s = s_init.copy()
        sums = compute_sums(s)
        current_max = max_abs_sum(sums)
        
        best_s = s.copy()
        best_score = current_max
        
        rng = random.Random(42)
        
        # Temperature schedule: start high, cool down
        T_start = 5.0
        T_end = 0.01
        # We'll use a time-based cooling schedule
        
        t_start = time.time()
        
        while time.time() - t_start < time_budget:
            # Current temperature based on elapsed fraction
            elapsed = time.time() - t_start
            frac = elapsed / time_budget
            T = T_start * (T_end / T_start) ** frac
            
            # Pick random position to flip
            i = rng.randint(0, N - 1)
            
            old_val = s[i]
            delta = -2 * old_val
            
            # Get affected APs
            affected = aps_containing_arr[i]
            
            # Compute new sums for affected APs
            old_sums_affected = sums[affected].copy()
            new_sums_affected = old_sums_affected + delta
            
            # Compute new max
            # The new global max is max of:
            # 1. Max of unaffected APs (unchanged)
            # 2. Max of affected APs (updated)
            
            # For efficiency, we can't easily compute max of unaffected without copying.
            # Instead, we'll do a full recompute of max after tentative update.
            # This is O(num_aps) per step, which is ~100k ops.
            
            sums[affected] = new_sums_affected
            s[i] = -old_val
            
            new_max = max_abs_sum(sums)
            
            # Acceptance criterion
            if new_max < current_max:
                # Always accept improvement
                current_max = new_max
                if current_max < best_score:
                    best_score = current_max
                    best_s = s.copy()
            else:
                # Delta is the increase in max (>= 0)
                delta_max = new_max - current_max
                # Accept with probability exp(-delta_max / T)
                if rng.random() < np.exp(-delta_max / T):
                    current_max = new_max
                else:
                    # Reject: revert
                    sums[affected] = old_sums_affected
                    s[i] = old_val
        
        return best_s, best_score
    
    # Multiple restarts with different initializations
    best_global_s = None
    best_global_score = float('inf')
    
    rng_global = random.Random(123)
    
    # Generate several different initializations
    def thue_morse(N):
        seq = []
        for i in range(N):
            count = bin(i).count('1')
            seq.append(1 if count % 2 == 0 else -1)
        return np.array(seq, dtype=np.int8)
    
    def random_init():
        return np.array([rng_global.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
    
    # Try a few different starting points
    inits = [
        thue_morse(N),
        random_init(),
        random_init(),
    ]
    
    # Allocate time to each restart
    num_restarts = len(inits)
    time_per_restart = time_limit / num_restarts
    
    for init in inits:
        if time.time() - start_time > time_limit:
            break
        
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1.0:
            break
            
        s_best, score_best = simulated_annealing(init, remaining * 0.9)
        
        if score_best < best_global_score:
            best_global_score = score_best
            best_global_s = s_best
            
        if best_global_score <= 1:
            break
    
    if best_global_s is None:
        best_global_s = thue_morse(N)
        
    return [int(x) for x in best_global_s]
