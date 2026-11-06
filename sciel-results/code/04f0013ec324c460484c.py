import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit

    # 1. Precompute APs
    # We store APs as lists of indices.
    # We also need to know, for each index i, which APs contain it.
    
    ap_indices = []
    # aps_containing[i] will be a list of AP indices (j)
    aps_containing = [[] for _ in range(N)]
    
    # Generate APs
    # Step d, Start s.
    # Length L is determined by s + (L-1)d < N.
    # We include all APs of length >= 2.
    
    ap_id = 0
    for d in range(1, N + 1):
        for s in range(0, N - d + 1):
            # Generate terms
            terms = []
            curr = s
            while curr < N:
                terms.append(curr)
                curr += d
            if len(terms) >= 2:
                ap_indices.append(terms)
                for idx in terms:
                    aps_containing[idx].append(ap_id)
                ap_id += 1
                
    num_aps = ap_id
    
    # Initialize signs
    # We will use a list of ints -1, 1
    # And a numpy array for sums for fast max calculation
    
    best_signs = None
    best_max_sum = N # Worst case
    
    def calc_max_sum(signs, sums):
        return int(np.max(np.abs(sums)))
        
    def greedy_init(rng):
        signs = np.zeros(N, dtype=np.int8)
        sums = np.zeros(num_aps, dtype=np.float32)
        
        for i in range(N):
            # Try 1 and -1
            best_s = 1
            best_cost = float('inf')
            
            for s in [1, -1]:
                signs[i] = s
                # Update sums for APs containing i
                # Note: We are building the sequence left-to-right.
                # APs containing i might have indices > i which are 0 (unset).
                # This is a problem for greedy.
                # We can't evaluate the full sum of an AP if it extends into the future.
                
                # Alternative Greedy:
                # Just pick randomly or based on local neighborhood?
                # Let's stick to random init for simplicity and rely on SA.
                pass
            
            # Simple random init
            signs[i] = 1 if rng.random() < 0.5 else -1
            
        # Calculate initial sums
        for j, terms in enumerate(ap_indices):
            s_val = 0
            for idx in terms:
                s_val += signs[idx]
            sums[j] = s_val
            
        return signs, sums

    # SA Parameters
    # We will run multiple restarts
    
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(seed)
        
        # Initialize
        signs = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        sums = np.zeros(num_aps, dtype=np.float32)
        
        for j, terms in enumerate(ap_indices):
            s_val = 0
            for idx in terms:
                s_val += signs[idx]
            sums[j] = s_val
            
        current_max = calc_max_sum(signs, sums)
        
        if current_max < best_max_sum:
            best_max_sum = current_max
            best_signs = signs.copy()
            
        # SA Loop
        T = 5.0
        steps = 0
        max_steps = 100000 # Adjust based on speed
        
        while time.time() < deadline - 2:
            steps += 1
            
            i = rng.randint(0, N - 1)
            old_sign = signs[i]
            new_sign = -old_sign
            
            # Update sums
            # Delta = new_sign - old_sign = -2 * old_sign
            delta = -2.0 * old_sign
            
            # Get APs containing i
            ap_list = aps_containing[i]
            
            if not ap_list:
                continue
                
            # Update sums for these APs
            # sums[ap_list] += delta
            # This is a vectorized update
            
            sums[ap_list] += delta
            
            # Calculate new max
            # np.max(np.abs(sums)) is O(num_aps)
            # num_aps ~ 30,000.
            # This is fast.
            
            new_max = calc_max_sum(signs, sums)
            
            # Acceptance
            accept = False
            if new_max < current_max:
                accept = True
            else:
                # If equal, we can accept to move around
                if new_max == current_max:
                    accept = rng.random() < 0.5
                else:
                    # Penalty for increase
                    # Use T to allow jumps
                    delta_obj = new_max - current_max
                    if delta_obj / T > 10:
                        accept = False
                    else:
                        accept = rng.random() < math.exp(-delta_obj / T)
                        
            if accept:
                signs[i] = new_sign
                current_max = new_max
                
                if current_max < best_max_sum:
                    best_max_sum = current_max
                    best_signs = signs.copy()
            else:
                # Revert sums
                sums[ap_list] -= delta
                
            # Annealing
            if steps % 100 == 0:
                T *= 0.95
                if T < 0.1:
                    T = 0.1
                    
            if best_max_sum <= 2:
                # Can't do better than 2 (proven impossible to be 1)
                # Actually, we proved 1 is impossible. So 2 is optimal.
                break
                
    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]
        
    return [int(s) for s in best_signs]
