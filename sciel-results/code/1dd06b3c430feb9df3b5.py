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

pos_to_ap_np = [np.array(lst, dtype=np.int32) if lst else np.empty(0, dtype=np.int32) for lst in pos_to_ap]

def thue_morse(N):
    """Generate Thue-Morse sequence: 1 if bit_count(i) is even, -1 otherwise."""
    seq = np.zeros(N, dtype=np.int8)
    for i in range(N):
        parity = bin(i).count('1') & 1
        seq[i] = 1 if parity == 0 else -1
    return seq

def compute_sums(s):
    """Compute sums for all APs."""
    S = np.zeros(num_aps, dtype=np.int16)
    for ap_idx, indices in enumerate(ap_indices):
        s_indices = np.array(indices, dtype=np.int32) - 1
        S[ap_idx] = np.sum(s[s_indices])
    return S

def compute_hist(S):
    """Compute histogram of absolute sums."""
    abs_S = np.abs(S)
    hist = np.zeros(N + 1, dtype=np.int32)
    np.add.at(hist, abs_S, 1)
    return hist, int(np.max(abs_S))

def local_search(s, S, hist, current_max, deadline, rng, max_steps=20000):
    """Simulated annealing local search with histogram-based max tracking."""
    temp = 3.0
    cooling_rate = 0.997
    end_temp = 0.005
    
    best_s = s.copy()
    best_max = current_max
    steps = 0
    
    while time.time() < deadline and steps < max_steps:
        j = rng.randint(0, N - 1)
        
        old_val = s[j]
        delta = -2 * old_val
        
        ap_idx_list = pos_to_ap_np[j]
        
        old_max = current_max
        max_new_abs_in_affected = 0
        
        for ap_idx in ap_idx_list:
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
            if temp > end_temp:
                p = math.exp(-cost_diff / temp)
                accept = (rng.random() < p)
            else:
                accept = False
        
        if accept:
            s[j] = -old_val
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_s = s.copy()
        else:
            for ap_idx in ap_idx_list:
                new_sum = S[ap_idx]
                old_sum = new_sum - delta
                
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                hist[new_abs] -= 1
                hist[old_abs] += 1
                
                S[ap_idx] = old_sum
        
        temp *= cooling_rate
        steps += 1
    
    return best_s, best_max

def construct():
    deadline = time.time() + 55.0
    best_seq = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    # Use Thue-Morse as primary initialization
    s_init = thue_morse(N)
    S = compute_sums(s_init)
    hist, current_max = compute_hist(S)
    
    if current_max < best_score:
        best_score = current_max
        best_seq = s_init.copy()
    
    # Run local search from Thue-Morse start
    if time.time() < deadline - 5.0:
        s_opt, opt_max = local_search(s_init.copy(), S.copy(), hist.copy(), current_max, deadline - 5.0, rng)
        if opt_max < best_score:
            best_score = opt_max
            best_seq = s_opt
    
    # Try a few more random restarts if time permits
    remaining_deadline = deadline - 2.0
    num_restarts = 3
    restart_budget = (remaining_deadline - time.time()) / num_restarts if time.time() < remaining_deadline else 0
    
    for r in range(num_restarts):
        if time.time() > remaining_deadline:
            break
        
        s = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        S = compute_sums(s)
        hist, current_max = compute_hist(S)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        if time.time() < remaining_deadline - 1.0:
            s_opt, opt_max = local_search(s, S, hist, current_max, remaining_deadline, rng, max_steps=10000)
            if opt_max < best_score:
                best_score = opt_max
                best_seq = s_opt
    
    return [int(x) for x in best_seq]
