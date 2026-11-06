import random
import numpy as np
import time
import math

N = 256

def get_aps():
    ap_indices = []
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                indices = [a + i * d for i in range(l)]
                ap_indices.append(indices)
    return ap_indices

ap_indices = get_aps()
num_aps = len(ap_indices)

pos_to_ap = [[] for _ in range(N)]
for ap_idx, indices in enumerate(ap_indices):
    for idx in indices:
        pos_to_ap[idx - 1].append(ap_idx)

def compute_initial_sums(s):
    """Compute initial sums for all APs efficiently using prefix sums per step."""
    S = np.zeros(num_aps, dtype=np.int16)
    
    # For each step d, compute all APs with that step
    # Group APs by step
    for d in range(1, N + 1):
        # Find all APs with step d
        for a in range(1, N + 1):
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                # Find the index of this AP in ap_indices
                # This is slow, so let's use a different approach
                pass
    
    # Better approach: precompute a mapping
    # Actually, let's just compute directly but more efficiently
    # We'll use the fact that for fixed d, we can use prefix sums
    
    # Let's rebuild: create a 2D array for each step
    # For step d, positions a, a+d, a+2d, ... form an AP
    # We can use prefix sums along each residue class mod d
    
    # Simpler: just compute each AP sum directly but vectorize where possible
    for ap_idx, indices in enumerate(ap_indices):
        s_indices = np.array(indices) - 1
        S[ap_idx] = np.sum(s[s_indices])
    
    return S

def polish(s, S, hist, current_max, deadline, rng, max_polish_steps=5000):
    """Final polish pass with tighter acceptance criteria."""
    polish_steps = 0
    temp = 0.5
    cooling_rate = 0.99
    
    while time.time() < deadline and polish_steps < max_polish_steps:
        j = rng.randint(0, N - 1)
        
        old_val = s[j]
        delta = -2 * old_val
        
        ap_indices_for_j = pos_to_ap[j]
        
        old_max = current_max
        max_new_abs_in_affected = 0
        
        for ap_idx in ap_indices_for_j:
            old_sum = S[ap_idx]
            new_sum = old_sum + delta
            
            old_abs = abs(old_sum)
            new_abs = abs(new_sum)
            
            hist[old_abs] -= 1
            hist[new_abs] += 1
            
            S[ap_idx] = new_sum
            
            if new_abs > max_new_abs_in_affected:
                max_new_abs_in_affected = new_abs
        
        if hist[old_max] == 0:
            if max_new_abs_in_affected > old_max:
                new_max = max_new_abs_in_affected
            else:
                scan_result = old_max - 1
                while scan_result > 0 and hist[scan_result] == 0:
                    scan_result -= 1
                new_max = max(max_new_abs_in_affected, scan_result)
        else:
            new_max = max(old_max, max_new_abs_in_affected)
        
        cost_diff = new_max - old_max
        
        if cost_diff <= 0:
            accept = True
        else:
            if temp > 0.01:
                p = math.exp(-cost_diff / temp)
                accept = (rng.random() < p)
            else:
                accept = False
        
        if accept:
            s[j] = -old_val
            current_max = new_max
        else:
            for ap_idx in ap_indices_for_j:
                new_sum = S[ap_idx]
                old_sum = new_sum - delta
                
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                hist[new_abs] -= 1
                hist[old_abs] += 1
                
                S[ap_idx] = old_sum
        
        temp *= cooling_rate
        polish_steps += 1
    
    return s, S, hist, current_max

def construct():
    deadline = time.time() + 55.0
    best_seq = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    start_temp = 5.0
    end_temp = 0.001
    cooling_rate = 0.995
    
    while time.time() < deadline:
        s = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        
        S = np.zeros(num_aps, dtype=np.int16)
        for ap_idx, indices in enumerate(ap_indices):
            s_indices = np.array(indices) - 1
            S[ap_idx] = np.sum(s[s_indices])
            
        hist = np.zeros(N + 1, dtype=np.int32)
        abs_S = np.abs(S)
        np.add.at(hist, abs_S, 1)
        
        current_max = int(np.max(abs_S))
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
            
        temp = start_temp
        steps_in_restart = 0
        max_steps_per_restart = 15000
        
        while time.time() < deadline and steps_in_restart < max_steps_per_restart:
            j = rng.randint(0, N - 1)
            
            old_val = s[j]
            delta = -2 * old_val
            
            ap_indices_for_j = pos_to_ap[j]
            
            old_max = current_max
            
            max_new_abs_in_affected = 0
            
            for ap_idx in ap_indices_for_j:
                old_sum = S[ap_idx]
                new_sum = old_sum + delta
                
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                hist[old_abs] -= 1
                hist[new_abs] += 1
                
                S[ap_idx] = new_sum
                
                if new_abs > max_new_abs_in_affected:
                    max_new_abs_in_affected = new_abs
            
            if hist[old_max] == 0:
                if max_new_abs_in_affected > old_max:
                    new_max = max_new_abs_in_affected
                else:
                    scan_result = old_max - 1
                    while scan_result > 0 and hist[scan_result] == 0:
                        scan_result -= 1
                    new_max = max(max_new_abs_in_affected, scan_result)
            else:
                new_max = max(old_max, max_new_abs_in_affected)
            
            cost_diff = new_max - old_max
            
            if cost_diff <= 0:
                accept = True
            else:
                if temp > 0:
                    p = math.exp(-cost_diff / temp)
                    accept = (rng.random() < p)
                else:
                    accept = False
            
            if accept:
                s[j] = -old_val
                current_max = new_max
                if current_max < best_score:
                    best_score = current_max
                    best_seq = s.copy()
            else:
                for ap_idx in ap_indices_for_j:
                    new_sum = S[ap_idx]
                    old_sum = new_sum - delta
                    
                    old_abs = abs(old_sum)
                    new_abs = abs(new_sum)
                    
                    hist[new_abs] -= 1
                    hist[old_abs] += 1
                    
                    S[ap_idx] = old_sum
            
            temp *= cooling_rate
            steps_in_restart += 1
            
            if temp < end_temp:
                break
        
        # Final polish pass
        if time.time() < deadline - 5.0:
            s, S, hist, current_max = polish(s, S, hist, current_max, deadline - 5.0, rng)
            if current_max < best_score:
                best_score = current_max
                best_seq = s.copy()
    
    return [int(x) for x in best_seq]
