import numpy as np
import random
import time
import math

N = 256

def construct():
    deadline = time.time() + 55.0
    rng = random.Random(42)
    
    # 1. Precompute APs
    # We store APs as (start, step, length) to save memory and allow fast sum calculation
    # But for local search, we need to know which APs are affected by a flip at position j.
    # So we need pos_to_ap_indices: list of AP indices for each position.
    
    ap_starts = []
    ap_steps = []
    ap_lengths = []
    pos_to_ap = [[] for _ in range(N)]
    
    ap_id = 0
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                start_idx = a - 1
                ap_starts.append(start_idx)
                ap_steps.append(d)
                ap_lengths.append(l)
                
                # Add this AP to all positions it covers
                for i in range(l):
                    pos = start_idx + i * d
                    pos_to_ap[pos].append(ap_id)
                
                ap_id += 1
                
    num_aps = ap_id
    
    # Convert pos_to_ap to numpy arrays for slightly faster iteration?
    # Actually, lists of ints are fine. Let's keep them as lists for now, 
    # but we can convert to np.array if needed. Lists are usually faster for small iterations in Python.
    
    # 2. Fast Initial Sum Computation using Prefix Sums
    # Group AP indices by step
    aps_by_step = [[] for _ in range(N + 1)]
    for idx in range(num_aps):
        d = ap_steps[idx]
        aps_by_step[d].append(idx)
        
    def compute_initial_sums(s):
        S = np.zeros(num_aps, dtype=np.int16)
        
        for d in range(1, N + 1):
            if not aps_by_step[d]:
                continue
                
            # Build prefix sums for each residue class mod d
            # prefix[r] will be an array of length (number of elements in class r) + 1
            # prefix[r][k] = sum of s at positions r, r+d, ..., r+(k-1)d
            prefix = {}
            for r in range(d):
                positions = list(range(r, N, d))
                if not positions:
                    continue
                vals = s[positions]
                ps = np.zeros(len(vals) + 1, dtype=np.int16)
                np.cumsum(vals, out=ps[1:])
                prefix[r] = ps
                
            # Compute sums for APs with this step
            for ap_idx in aps_by_step[d]:
                start = ap_starts[ap_idx]
                l = ap_lengths[ap_idx]
                r = start % d
                # Index in the residue class
                # start = r + k*d => k = (start - r) // d = start // d
                idx_in_residue = start // d
                ps = prefix[r]
                S[ap_idx] = ps[idx_in_residue + l] - ps[idx_in_residue]
                
        return S

    def build_hist(S):
        hist = np.zeros(N + 1, dtype=np.int32)
        abs_S = np.abs(S)
        # np.add.at is slow for this, simple loop or bincount is better
        # Since values are small (0..256), we can use bincount
        hist = np.bincount(abs_S, minlength=N+1).astype(np.int32)
        current_max = int(np.max(abs_S))
        return hist, current_max

    best_seq = None
    best_score = float('inf')
    
    # Run Simulated Annealing
    # We will run multiple restarts if time permits, but one long run is often good.
    # Let's try 1 long run first, then restart if time allows.
    
    num_restarts = 1
    # If we have lots of time, maybe more restarts? 
    # But SA with good cooling is usually enough.
    
    for restart in range(num_restarts):
        # Initialize
        if restart == 0:
            # Start with a good heuristic: Thue-Morse or Random
            # Thue-Morse has low discrepancy for powers of 2, might be a good start
            s = np.ones(N, dtype=np.int8)
            for i in range(1, N):
                c = bin(i).count('1')
                s[i] = 1 if c % 2 == 0 else -1
        else:
            s = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
            
        S = compute_initial_sums(s)
        hist, current_max = build_hist(S)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
            
        temp = 5.0
        cooling_rate = 0.995
        end_temp = 0.001
        
        steps = 0
        max_steps = 200000 # Safety cap, but time is the real limit
        
        while time.time() < deadline and temp > end_temp and steps < max_steps:
            j = rng.randint(0, N - 1)
            old_val = s[j]
            delta = -2 * old_val
            
            affected_aps = pos_to_ap[j]
            
            old_max = current_max
            max_new_abs = 0
            
            # Update sums and histogram
            for ap_idx in affected_aps:
                old_sum = S[ap_idx]
                new_sum = old_sum + delta
                S[ap_idx] = new_sum
                
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                hist[old_abs] -= 1
                hist[new_abs] += 1
                
                if new_abs > max_new_abs:
                    max_new_abs = new_abs
            
            # Determine new_max
            if hist[old_max] == 0:
                # The old max bucket is empty, we need to find the new max
                # It's either max_new_abs or something lower
                # Scan down from old_max - 1
                scan = old_max - 1
                while scan > 0 and hist[scan] == 0:
                    scan -= 1
                new_max = max(max_new_abs, scan)
            else:
                # The old max is still present, so new_max is at least old_max
                # It could be higher if max_new_abs > old_max
                new_max = max(old_max, max_new_abs)
            
            cost_diff = new_max - old_max
            
            accept = False
            if cost_diff <= 0:
                accept = True
            else:
                if temp > 0:
                    p = math.exp(-cost_diff / temp)
                    if rng.random() < p:
                        accept = True
            
            if accept:
                s[j] = -old_val
                current_max = new_max
                if current_max < best_score:
                    best_score = current_max
                    best_seq = s.copy()
            else:
                # Revert
                for ap_idx in affected_aps:
                    new_sum = S[ap_idx]
                    old_sum = new_sum - delta
                    S[ap_idx] = old_sum
                    
                    new_abs = abs(new_sum)
                    old_abs = abs(old_sum)
                    
                    hist[new_abs] -= 1
                    hist[old_abs] += 1
            
            temp *= cooling_rate
            steps += 1
            
            # If we have time, we can do a few more restarts
            # But let's stick to one long run for stability unless it converges too fast.
            # If temp drops below end_temp, we break.
            
    if best_seq is None:
        best_seq = np.ones(N, dtype=np.int8)
        
    return [int(x) for x in best_seq]
