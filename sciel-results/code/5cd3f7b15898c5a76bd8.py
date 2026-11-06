import time
import random
import sys

def construct():
    n = 256
    deadline = time.time() + 85.0  # Safety margin
    
    # 1. Generate all arithmetic progressions
    # AP defined by start s, step k, length m
    # 1 <= s, 1 <= k, s + (m-1)k <= n
    # We store the list of indices for each AP
    aps = []
    for k in range(1, n + 1):
        for s in range(1, n + 1, k):
            # Max length m such that s + (m-1)k <= n
            max_m = (n - s) // k + 1
            # We consider all lengths from 1 to max_m
            # Actually, the task says "any ... length".
            # An AP of length L is a subset of an AP of length L+1?
            # No, the sum is over the specific length.
            # We need to check all possible APs.
            # Number of APs: sum_{k=1}^n sum_{s=1}^n (n-s)//k + 1
            # This is roughly n log n. For n=256, it's small.
            current_ap = []
            for m in range(1, max_m + 1):
                # Append the new element to the current AP
                current_ap.append(s + (m - 1) * k)
                # Store a copy of the current AP
                aps.append(list(current_ap))
                
    num_aps = len(aps)
    
    # 2. Precompute for each index i, which APs contain it
    # ap_indices[i] = list of indices of APs in 'aps' that contain i
    ap_indices = [[] for _ in range(n)]
    for ap_idx, ap in enumerate(aps):
        for idx in ap:
            ap_indices[idx - 1].append(ap_idx) # 0-based index for x

    # 3. Initialize signs
    rng = random.Random(42)
    x = [rng.choice([-1, 1]) for _ in range(n)]
    
    # 4. Initialize sums
    sums = [0] * num_aps
    for ap_idx, ap in enumerate(aps):
        s_val = 0
        for idx in ap:
            s_val += x[idx - 1]
        sums[ap_idx] = s_val

    def get_max_abs():
        m = 0
        for s_val in sums:
            abs_val = abs(s_val)
            if abs_val > m:
                m = abs_val
        return m

    current_max = get_max_abs()
    best_x = x[:]
    best_max = current_max

    # Local Search
    # We will try to flip indices that are in APs with high absolute sums
    # To be efficient, we sample candidates
    
    no_improve_count = 0
    
    while time.time() < deadline:
        # Identify "hot" APs
        # Instead of sorting all, just pick a threshold
        threshold = current_max * 0.8
        hot_indices = set()
        for ap_idx, s_val in enumerate(sums):
            if abs(s_val) >= threshold:
                for idx in aps[ap_idx]:
                    hot_indices.add(idx - 1)
        
        if not hot_indices:
            hot_indices = set(range(n))
        
        # Sample candidates from hot_indices
        # If hot_indices is large, sample; else use all
        candidates = list(hot_indices)
        if len(candidates) > 50:
            candidates = rng.sample(candidates, 50)
        else:
            rng.shuffle(candidates)
            
        best_flip_idx = -1
        best_flip_max = current_max
        best_new_sums = None # We don't store full sums, just the max value
        
        # Evaluate candidates
        for i in candidates:
            # Simulate flip at i
            # Update sums for APs containing i
            affected_aps = ap_indices[i]
            
            # Store old values to revert
            old_sums = []
            for ap_idx in affected_aps:
                old_sums.append(sums[ap_idx])
                
            # Apply flip
            flip_val = -2 * x[i]
            for ap_idx in affected_aps:
                sums[ap_idx] += flip_val
            
            # Calculate new max
            # Optimization: We only need to check if max decreased.
            # But max can increase if we made a small sum large.
            # So we must check all sums?
            # No, only the affected ones changed.
            # Unaffected ones have same abs values.
            # So new_max = max( max_unaffected, max_affected_new )
            
            # To get max_unaffected fast, we'd need a data structure.
            # Given n=256, num_aps ~ 2000, scanning all is fast enough in C but slow in Py.
            # Let's just scan all sums.
            m = 0
            for s_val in sums:
                abs_val = abs(s_val)
                if abs_val > m:
                    m = abs_val
            
            if m < best_flip_max:
                best_flip_max = m
                best_flip_idx = i
                # We don't keep the sums, we revert and re-apply if needed?
                # Or we can just revert and remember the index.
                
            # Revert
            for j, ap_idx in enumerate(affected_aps):
                sums[ap_idx] = old_sums[j]
                
        if best_flip_idx != -1:
            # Commit the flip
            i = best_flip_idx
            x[i] *= -1
            for ap_idx in ap_indices[i]:
                sums[ap_idx] += -2 * x[i] # Wait, x[i] is already flipped.
                # Original sum S. New sum S + 2*old_x[i].
                # We flipped x[i] -> -x[i].
                # Change is -2*x[i]_new?
                # Let's trace:
                # x_old = 1. Flip to -1. Change -2.
                # x_new = -1. -2 * x_new = 2. Wrong.
                # Change is -2 * x_old.
                # Or simply: sums[ap] += -2 * x[i] (where x[i] is the NEW value? No)
                # Let's stick to:
                # delta = -2 * x_old
                # x_new = -x_old
                # So delta = 2 * x_new
                # Let's re-verify.
                # S = sum(x).
                # Flip x_i to -x_i.
                # S' = S - 2*x_i.
                # So we add -2*x_i (old value).
                
            # My previous simulation:
            # flip_val = -2 * x[i] (where x[i] was OLD value in the simulation block)
            # Then x[i] was NOT updated in the simulation block, only sums were.
            # So in the commit block, x[i] is still OLD.
            # So we should do:
            # delta = -2 * x[i]
            # sums[ap] += delta
            # x[i] = -x[i]
            
            # Let's redo the commit logic carefully.
            # In the loop:
            #   x[i] is OLD.
            #   flip_val = -2 * x[i].
            #   sums[ap] += flip_val.
            #   This calculates S_new.
            #   Then we revert sums.
            #   x[i] is still OLD.
            
            # So for commit:
            delta = -2 * x[best_flip_idx]
            for ap_idx in ap_indices[best_flip_idx]:
                sums[ap_idx] += delta
            x[best_flip_idx] *= -1
            
            current_max = best_flip_max
            if current_max < best_max:
                best_max = current_max
                best_x = x[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If stuck, try a random flip to escape
            if no_improve_count > 10:
                i = rng.randint(0, n - 1)
                delta = -2 * x[i]
                for ap_idx in ap_indices[i]:
                    sums[ap_idx] += delta
                x[i] *= -1
                current_max = get_max_abs()
                if current_max < best_max:
                    best_max = current_max
                    best_x = x[:]
                no_improve_count = 0
            elif no_improve_count > 50:
                # Restart with new random seed?
                # Or just break if we've been stuck for a long time
                pass

    return best_x

# Test locally if needed
# if __name__ == "__main__":
#     res = construct()
#     print(len(res))
#     # Verify
#     n = 256
#     max_sum = 0
#     for k in range(1, n+1):
#         for s in range(1, n+1, k):
#             curr = 0
#             for j in range(n):
#                 idx = s + j*k
#                 if idx > n: break
#                 curr += res[idx-1]
#                 if abs(curr) > max_sum: max_sum = abs(curr)
#     print("Max Discrepancy:", max_sum)
