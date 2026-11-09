import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 58.0  # Safety margin

    # 1. Precompute APs
    aps = []
    aps_containing = [[] for _ in range(n)]
    
    for d in range(1, n):
        for s in range(n):
            curr = s
            prefix = []
            while curr < n:
                prefix.append(curr)
                curr += d
            if len(prefix) >= 2:
                ap_id = len(aps)
                aps.append(prefix)
                for idx in prefix:
                    aps_containing[idx].append(ap_id)
    
    num_aps = len(aps)
    
    # Convert to numpy arrays for fast indexing
    aps_containing_np = [np.array(lst, dtype=np.int32) for lst in aps_containing]
    
    # Helper to compute initial sums
    def compute_sums(x):
        sums = np.zeros(num_aps, dtype=np.int32)
        for k, ap in enumerate(aps):
            s = 0
            for idx in ap:
                s += x[idx]
            sums[k] = s
        return sums

    # Initial random solution
    rng = random.Random(42)
    x = [rng.choice([-1, 1]) for _ in range(n)]
    sums = compute_sums(x)
    current_max = int(np.max(np.abs(sums)))
    
    best_x = x[:]
    best_max = current_max

    # Iterated Local Search with Multi-Flip Neighborhood
    # We will use a combination of single flips and pair swaps to escape local minima.
    
    while time.time() - start_time < time_limit:
        # Local Search Phase
        # Start from a random or perturbed solution
        x_curr = [rng.choice([-1, 1]) for _ in range(n)]
        sums_curr = compute_sums(x_curr)
        curr_max = int(np.max(np.abs(sums_curr)))
        
        T = 2.0
        cooling = 0.9995
        min_T = 0.01
        
        steps_since_improve = 0
        max_steps_no_improve = 5000
        
        while time.time() - start_time < time_limit:
            if T < min_T:
                # Reheat or restart
                if curr_max > best_max:
                    break
                else:
                    T = 2.0
                    # Perturb with a few random flips
                    for _ in range(3):
                        i = rng.randint(0, n - 1)
                        x_curr[i] = -x_curr[i]
                    sums_curr = compute_sums(x_curr)
                    curr_max = int(np.max(np.abs(sums_curr)))
                    continue

            # Choose neighborhood: 70% single flip, 30% pair swap
            if rng.random() < 0.7:
                # Single flip
                i = rng.randint(0, n - 1)
                old_val = x_curr[i]
                new_val = -old_val
                delta = -2 * old_val
                
                indices = aps_containing_np[i]
                if len(indices) == 0:
                    continue
                    
                sums_curr[indices] += delta
                new_max = int(np.max(np.abs(sums_curr)))
                delta_max = new_max - curr_max
                
                if delta_max < 0:
                    x_curr[i] = new_val
                    curr_max = new_max
                    steps_since_improve = 0
                    
                    if curr_max < best_max:
                        best_max = curr_max
                        best_x = x_curr[:]
                else:
                    if rng.random() < np.exp(-delta_max / T):
                        x_curr[i] = new_val
                        curr_max = new_max
                    else:
                        sums_curr[indices] -= delta
            else:
                # Pair swap (flip two different indices)
                # This can sometimes reduce max discrepancy by balancing out large sums
                i1 = rng.randint(0, n - 1)
                i2 = rng.randint(0, n - 1)
                if i1 == i2:
                    continue
                
                val1 = x_curr[i1]
                val2 = x_curr[i2]
                
                # Flip both
                delta1 = -2 * val1
                delta2 = -2 * val2
                
                indices1 = aps_containing_np[i1]
                indices2 = aps_containing_np[i2]
                
                if len(indices1) == 0 or len(indices2) == 0:
                    continue
                
                # Update sums for both
                sums_curr[indices1] += delta1
                sums_curr[indices2] += delta2
                
                new_max = int(np.max(np.abs(sums_curr)))
                delta_max = new_max - curr_max
                
                if delta_max < 0:
                    x_curr[i1] = -val1
                    x_curr[i2] = -val2
                    curr_max = new_max
                    steps_since_improve = 0
                    
                    if curr_max < best_max:
                        best_max = curr_max
                        best_x = x_curr[:]
                else:
                    if rng.random() < np.exp(-delta_max / T):
                        x_curr[i1] = -val1
                        x_curr[i2] = -val2
                        curr_max = new_max
                    else:
                        # Revert
                        sums_curr[indices1] -= delta1
                        sums_curr[indices2] -= delta2
            
            T *= cooling
            steps_since_improve += 1
            if steps_since_improve > max_steps_no_improve:
                break

    return best_x
