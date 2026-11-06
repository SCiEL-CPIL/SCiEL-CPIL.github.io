import random
import numpy as np
import time
import math

N = 256

def build_ap_data():
    """Build all APs and precompute data structures for fast local search."""
    ap_starts = []
    ap_steps = []
    ap_lengths = []
    
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                ap_starts.append(a - 1)
                ap_steps.append(d)
                ap_lengths.append(l)
    
    num_aps = len(ap_starts)
    
    # Build pos_to_ap: for each position, which APs contain it
    pos_to_ap = [[] for _ in range(N)]
    for ap_idx in range(num_aps):
        start = ap_starts[ap_idx]
        d = ap_steps[ap_idx]
        l = ap_lengths[ap_idx]
        for i in range(l):
            pos = start + i * d
            pos_to_ap[pos].append(ap_idx)
    
    pos_to_ap_np = [np.array(lst, dtype=np.int32) if lst else np.empty(0, dtype=np.int32) 
                    for lst in pos_to_ap]
    
    return ap_starts, ap_steps, ap_lengths, pos_to_ap_np, num_aps

ap_starts, ap_steps, ap_lengths, pos_to_ap_np, num_aps = build_ap_data()

def compute_initial_sums(s):
    """Compute sums for all APs using prefix sums per (step, residue)."""
    S = np.empty(num_aps, dtype=np.int16)
    
    # For each step d, build prefix sums for each residue class
    # Then each AP sum is a difference of two prefix sum values
    
    # Group APs by step
    aps_by_step = [[] for _ in range(N + 1)]
    for ap_idx in range(num_aps):
        d = ap_steps[ap_idx]
        aps_by_step[d].append(ap_idx)
    
    for d in range(1, N + 1):
        if not aps_by_step[d]:
            continue
        
        # Build prefix sums for each residue class mod d
        # For residue r, positions are r, r+d, r+2d, ...
        prefix = {}
        for r in range(d):
            positions = list(range(r, N, d))
            if not positions:
                continue
            vals = s[positions]
            ps = np.zeros(len(vals) + 1, dtype=np.int16)
            np.cumsum(vals, out=ps[1:])
            prefix[r] = ps
        
        # Now compute each AP sum with step d
        for ap_idx in aps_by_step[d]:
            start = ap_starts[ap_idx]
            l = ap_lengths[ap_idx]
            r = start % d
            idx_in_residue = start // d
            ps = prefix[r]
            S[ap_idx] = ps[idx_in_residue + l] - ps[idx_in_residue]
    
    return S

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
        
        S = compute_initial_sums(s)
        
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
                for ap_idx in ap_idx_list:
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
    
    return [int(x) for x in best_seq]
