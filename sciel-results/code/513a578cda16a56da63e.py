import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    N = 256
    
    # Precompute APs
    # An AP is defined by start, step, length.
    # We need all APs contained in [1, 256].
    # Indices in 0-based: 0 to 255.
    ap_indices = []
    for d in range(1, N // 2 + 1):
        for start in range(1, N - d + 1):
            # AP: start-1, start-1+d, ...
            indices = list(range(start - 1, N, d))
            if len(indices) >= 1:
                ap_indices.append(np.array(indices, dtype=np.int32))
                
    num_aps = len(ap_indices)
    
    # Build reverse mapping: for each position i, which APs contain it?
    # This is needed for fast updates if we do incremental, 
    # but for Best Improvement, we might just recompute sums for the flipped index's APs?
    # No, for Best Improvement, we evaluate the effect of flipping i.
    # Effect of flipping i:
    #   s[i] changes by -2*s[i].
    #   S[j] changes by -2*s[i] for all j in aps_containing[i].
    #   New Max = max( max(S_new), max(S_old_unaffected) )
    
    # To make Best Improvement fast, we need to know the "background" max.
    # Let's precompute aps_containing.
    aps_containing = [[] for _ in range(N)]
    for r, indices in enumerate(ap_indices):
        for c in indices:
            aps_containing[c].append(r)
            
    # Convert to numpy arrays for vectorized access
    aps_containing_arr = [np.array(lst, dtype=np.int32) for lst in aps_containing]

    def compute_sums(s):
        sums = np.zeros(num_aps, dtype=np.int32)
        for r, indices in enumerate(ap_indices):
            sums[r] = np.sum(s[indices])
        return sums

    def thue_morse(N):
        seq = []
        for i in range(N):
            count = bin(i).count('1')
            seq.append(1 if count % 2 == 0 else -1)
        return np.array(seq, dtype=np.int8)

    def best_improvement_hc(s_init, time_budget):
        s = s_init.copy()
        sums = compute_sums(s)
        current_max = np.max(np.abs(sums))
        
        best_s = s.copy()
        best_score = current_max
        
        t_start = time.time()
        
        # We will try to improve by flipping one bit at a time.
        # To speed up, we can prioritize indices that are in many "bad" APs.
        # But simple random or sequential is okay for now.
        
        improved = True
        while improved and (time.time() - t_start < time_budget):
            improved = False
            best_delta = 0
            best_idx = -1
            best_new_max = current_max
            
            # We can iterate through all indices
            for i in range(N):
                # Flip i
                delta = -2 * s[i]
                affected = aps_containing_arr[i]
                
                # Save old values for affected
                old_vals = sums[affected].copy()
                sums[affected] += delta
                
                new_max = np.max(np.abs(sums))
                
                # Revert
                sums[affected] = old_vals
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_idx = i
                    best_delta = delta
                    improved = True
            
            if improved:
                # Apply the best flip
                s[best_idx] = -s[best_idx]
                affected = aps_containing_arr[best_idx]
                sums[affected] += best_delta
                current_max = best_new_max
                
                if current_max < best_score:
                    best_score = current_max
                    best_s = s.copy()
                    
                # Check time
                if time.time() - t_start > time_budget:
                    break
        
        return best_s, best_score

    def run_ils(time_budget):
        # ILS: Iterated Local Search
        # Start with Thue-Morse
        s = thue_morse(N)
        
        # Initial Local Search
        s, score = best_improvement_hc(s, time_budget * 0.5)
        
        best_s = s
        best_score = score
        
        t_start = time.time()
        rng = random.Random(42)
        
        while time.time() - t_start < time_budget:
            # Perturb
            s_pert = s.copy()
            # Flip 1-3 random bits
            num_flips = rng.randint(1, 3)
            for _ in range(num_flips):
                idx = rng.randint(0, N - 1)
                s_pert[idx] = -s_pert[idx]
            
            # Local Search on perturbed
            remaining = time_budget - (time.time() - t_start)
            if remaining < 0.5: break
            
            s_new, score_new = best_improvement_hc(s_pert, remaining * 0.8)
            
            if score_new < best_score:
                best_score = score_new
                best_s = s_new
                s = s_new # Move to new best
                
            # If stuck, we can try more aggressive perturbation
            # But simple ILS is a good start.
            
        return best_s, best_score

    # Run ILS
    best_global_s, best_global_score = run_ils(time_limit)
    
    return [int(x) for x in best_global_s]
