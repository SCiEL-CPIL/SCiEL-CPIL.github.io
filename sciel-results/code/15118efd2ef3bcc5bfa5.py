import random
import numpy as np
import time

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

def construct():
    deadline = time.time() + 55.0
    best_seq = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        s = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        
        # Calculate initial sums S
        S = np.zeros(num_aps, dtype=np.int16)
        for ap_idx, indices in enumerate(ap_indices):
            s_indices = np.array(indices) - 1
            S[ap_idx] = np.sum(s[s_indices])
            
        # Build histogram of absolute sums
        # Max sum is N=256
        hist = np.zeros(N + 1, dtype=np.int32)
        abs_S = np.abs(S)
        np.add.at(hist, abs_S, 1)
        
        current_max = int(np.max(abs_S))
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        no_improve_count = 0
        max_no_improve = 5000  # Restart if stuck
        
        while time.time() < deadline:
            # Pick random position to flip
            j = rng.randint(0, N - 1)
            
            old_val = s[j]
            # delta is the change in sum for APs containing j
            # new_sum = old_sum - old_val + (-old_val) = old_sum - 2*old_val
            delta = -2 * old_val
            
            ap_indices_for_j = pos_to_ap[j]
            
            # Track if current_max needs update
            max_new_abs = 0
            
            for ap_idx in ap_indices_for_j:
                old_sum = S[ap_idx]
                new_sum = old_sum + delta
                
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                # Update histogram
                hist[old_abs] -= 1
                hist[new_abs] += 1
                
                # Update S
                S[ap_idx] = new_sum
                
                if new_abs > max_new_abs:
                    max_new_abs = new_abs
            
            # Update current_max
            # If the bucket for current_max is empty, we need to find the new max
            if hist[current_max] == 0:
                # The new max must be <= current_max
                # We can scan down from current_max
                # Or we know max_new_abs is the max of the *changed* elements.
                # But the global max could be an unchanged element that was < current_max.
                # So we scan down from current_max.
                while current_max > 0 and hist[current_max] == 0:
                    current_max -= 1
                # It is possible that max_new_abs > current_max? 
                # No, because if new_abs > old_current_max, then hist[new_abs] > 0, 
                # and the scan would stop there if it started high enough?
                # Wait. If we removed the only element at current_max, and added one at max_new_abs.
                # If max_new_abs > current_max, then hist[max_new_abs] > 0.
                # But we are scanning DOWN from current_max. We would miss max_new_abs if it's higher.
                # So we should set current_max = max(current_max_after_scan, max_new_abs)
                # Actually, if max_new_abs > current_max, then the new global max is at least max_new_abs.
                # Since all other elements were <= current_max < max_new_abs, the new global max IS max_new_abs.
                if max_new_abs > current_max:
                    current_max = max_new_abs
            else:
                # current_max bucket is not empty, so current_max is still a valid max candidate.
                # But maybe a new element is larger?
                if max_new_abs > current_max:
                    current_max = max_new_abs
            
            # If we improved, accept the flip
            if current_max < best_score:
                s[j] = -old_val
                best_score = current_max
                best_seq = s.copy()
                no_improve_count = 0
            else:
                # Revert
                # s[j] was not updated yet, so it's still old_val.
                # We need to revert S and hist.
                for ap_idx in ap_indices_for_j:
                    new_sum = S[ap_idx]
                    old_sum = new_sum - delta
                    
                    new_abs = abs(new_sum)
                    old_abs = abs(old_sum)
                    
                    hist[new_abs] -= 1
                    hist[old_abs] += 1
                    
                    S[ap_idx] = old_sum
                
                no_improve_count += 1
                
                if no_improve_count > max_no_improve:
                    break
    
    return [int(x) for x in best_seq]
