import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0
    
    # Precompute APs
    # AP is defined by start s and step d.
    # Indices: s, s+d, ..., < n
    aps = []
    index_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n):
        for s in range(n):
            # Generate indices
            # Optimization: just store the list
            idxs = list(range(s, n, d))
            if idxs:
                ap_id = len(aps)
                aps.append(idxs)
                for idx in idxs:
                    index_to_aps[idx].append(ap_id)
    
    num_aps = len(aps)
    
    # Initialize signs with Thue-Morse
    # Thue-Morse: s_i = (-1)^{popcount(i)}
    signs = np.array([(-1) ** (bin(i).count('1') % 2) for i in range(n)], dtype=np.int8)
    
    # Initialize sums
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    for k, idxs in enumerate(aps):
        ap_sums[k] = np.sum(signs[idxs])
        
    # Frequency of absolute sums
    # Max possible sum is 256
    max_possible = n
    freq_abs = np.zeros(max_possible + 1, dtype=np.int32)
    
    for s in ap_sums:
        freq_abs[abs(s)] += 1
        
    current_max = int(np.max(np.abs(ap_sums)))
    
    best_signs = signs.copy()
    best_max = current_max
    
    rng = random.Random(42)
    
    T = 5.0
    cooling_rate = 0.9999
    
    steps = 0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Pick random index to flip
        i = rng.randrange(n)
        old_sign = signs[i]
        delta = -2 * old_sign
        
        affected_aps = index_to_aps[i]
        
        # Get old sums for affected APs
        old_vals = ap_sums[affected_aps]
        new_vals = old_vals + delta
        
        # Update frequency array
        # Vectorized update for freq_abs is tricky with overlapping indices?
        # No, affected_aps are distinct APs.
        # But we need to decrement old and increment new.
        
        # Efficient update:
        # 1. Decrement freq for old_vals
        # 2. Increment freq for new_vals
        # 3. Update ap_sums
        
        # Since affected_aps is a list of indices into ap_sums:
        old_abs = np.abs(old_vals)
        new_abs = np.abs(new_vals)
        
        # Update freq_abs
        # We can use bincount or loops.
        # Given small size (256), a loop is fine, but numpy is faster.
        # However, we need to handle the max update carefully.
        
        # Let's do it in a loop for simplicity and correctness of max tracking
        # 256 iterations is negligible.
        
        local_max_increase = False
        new_max_candidate = current_max
        
        for idx_ap, old_v, new_v in zip(affected_aps, old_vals, new_vals):
            oa = abs(old_v)
            na = abs(new_v)
            
            freq_abs[oa] -= 1
            freq_abs[na] += 1
            
            ap_sums[idx_ap] = new_v
            
            if na > new_max_candidate:
                new_max_candidate = na
        
        # Check if current_max is still valid
        actual_new_max = new_max_candidate
        if freq_abs[current_max] == 0:
            # Find new max
            # Since we only changed 256 values, the new max is either new_max_candidate or lower
            # If freq_abs[current_max] became 0, the max must have dropped or been replaced.
            # We know new_max_candidate is the max of the *new* values of affected APs.
            # But the global max might be in an unaffected AP.
            # If freq_abs[current_max] == 0, then no AP has sum current_max.
            # So the new max is max(new_max_candidate, max of unaffected).
            # But we don't track max of unaffected separately.
            # However, if freq_abs[current_max] == 0, then the previous max is gone.
            # The new max is the largest k such that freq_abs[k] > 0.
            # We can search downwards from current_max.
            
            # Optimization: We know that if na > current_max for any affected, we would have set new_max_candidate > current_max.
            # If new_max_candidate == current_max and freq_abs[current_max] == 0, it means the only APs with max sum were affected and decreased.
            # So we must scan down.
            
            # But wait, if new_max_candidate < current_max, it means all affected APs decreased in abs sum (or stayed same but not max).
            # So the new max is either new_max_candidate (if it's the highest new value) or some unaffected AP.
            # If freq_abs[current_max] == 0, then no AP has current_max.
            # So the new max is the largest k < current_max with freq_abs[k] > 0, OR new_max_candidate if it's larger.
            # Actually, new_max_candidate is the max of the *new* values of affected APs.
            # The global new max is max(new_max_candidate, max_unaffected).
            # But max_unaffected <= current_max (since they didn't change and no AP has current_max anymore? No, unaffected APs had sums <= current_max).
            # If freq_abs[current_max] == 0, then all APs with sum current_max were affected.
            # So max_unaffected < current_max.
            # So the new global max is max(new_max_candidate, max_unaffected).
            # Since we don't know max_unaffected, we just scan down from current_max - 1?
            # Or just scan down from current_max.
            
            # Efficient scan:
            # We know freq_abs[k] for all k.
            # Start from current_max and go down until freq_abs[k] > 0.
            
            # But we also have new_max_candidate.
            # If new_max_candidate > current_max, we handled it above (actual_new_max = new_max_candidate).
            # Here we are in the case where we might need to lower the max.
            
            # Let's just find the max k where freq_abs[k] > 0.
            # Since the range is small (256), a reverse loop is fast.
            
            # Actually, if new_max_candidate is the max of the new affected values,
            # and freq_abs[current_max] == 0,
            # then the new global max is max(new_max_candidate, second_highest_unaffected).
            # But second_highest_unaffected < current_max.
            # So if new_max_candidate >= current_max, we are fine.
            # If new_max_candidate < current_max, we need to find the highest k < current_max with freq_abs[k] > 0.
            
            # Let's just do a reverse scan from current_max.
            # It's at most 256 steps.
            
            # Wait, if I just scan from current_max down to 0:
            # while freq_abs[actual_new_max] == 0:
            #    actual_new_max -= 1
            
            # But actual_new_max might be new_max_candidate which is > current_max?
            # No, we are in the branch where we check if current_max is gone.
            # If new_max_candidate > current_max, freq_abs[current_max] might still be > 0 (if some unaffected AP had current_max).
            # In that case, the new max is new_max_candidate.
            
            # Let's simplify:
            # 1. If new_max_candidate > current_max:
            #    actual_new_max = new_max_candidate
            # 2. Else:
            #    If freq_abs[current_max] == 0:
            #       # Max dropped
            #       # Find new max
            #       # It must be <= current_max - 1 (since freq_abs[current_max] is 0)
            #       # But it could be new_max_candidate?
            #       # new_max_candidate is the max of the new affected values.
            #       # So the global max is max(new_max_candidate, max_unaffected).
            #       # max_unaffected < current_max.
            #       # So global max is max(new_max_candidate, max_unaffected).
            #       # We can just scan down from current_max - 1?
            #       # No, new_max_candidate might be larger than max_unaffected.
            #       # So we should check if freq_abs[new_max_candidate] > 0?
            #       # Yes, because new_max_candidate is the abs value of one of the affected APs.
            #       # So freq_abs[new_max_candidate] is at least 1.
            #       # So the new max is at least new_max_candidate.
            #       # And it is at most current_max (since unaffected are < current_max).
            #       # So we can scan down from current_max to new_max_candidate?
            #       # Or just scan down from current_max.
            
            # Let's just scan down from current_max.
            while freq_abs[actual_new_max] == 0:
                actual_new_max -= 1
                
        else:
            actual_new_max = new_max_candidate
            
        # Acceptance
        cost_diff = actual_new_max - current_max
        
        if cost_diff < 0:
            accept = True
        else:
            if T < 1e-5:
                accept = False
            else:
                if cost_diff / T > 50:
                    prob = 0.0
                else:
                    prob = math.exp(-cost_diff / T)
                accept = rng.random() < prob
        
        if accept:
            signs[i] = -signs[i]
            current_max = actual_new_max
            
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            # Revert ap_sums
            ap_sums[affected_aps] = old_vals
            # Revert freq_abs
            for idx_ap, old_v, new_v in zip(affected_aps, old_vals, new_vals):
                oa = abs(old_v)
                na = abs(new_v)
                freq_abs[na] -= 1
                freq_abs[oa] += 1
            # signs[i] remains old
            # current_max remains old
            
        T *= cooling_rate
        steps += 1
        
    return best_signs.tolist()
