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
    
    # Group APs by step
    aps_by_step = [[] for _ in range(N + 1)]
    for ap_idx in range(num_aps):
        d = ap_steps[ap_idx]
        aps_by_step[d].append(ap_idx)
    
    for d in range(1, N + 1):
        if not aps_by_step[d]:
            continue
        
        # Build prefix sums for each residue class mod d
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

def polish(s, S, current_max, deadline, rng):
    """Final high-precision polish pass with stricter acceptance criteria."""
    best_score = current_max
    best_seq = s.copy()
    
    # Phase 1: Strict local search - only accept improvements
    steps = 0
    max_steps = 5000
    while time.time() < deadline and steps < max_steps:
        improved = False
        for j in range(N):
            old_val = s[j]
            delta = -2 * old_val
            ap_idx_list = pos_to_ap_np[j]
            
            if len(ap_idx_list) == 0:
                continue
            
            # Compute new max if we flip
            old_max = current_max
            max_new_abs_in_affected = 0
            
            # Save old values
            old_sums = S[ap_idx_list].copy()
            
            for ap_idx in ap_idx_list:
                new_sum = old_sums[ap_idx] + delta
                new_abs = abs(new_sum)
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
            
            if new_max < old_max:
                # Accept improvement
                for ap_idx in ap_idx_list:
                    S[ap_idx] = old_sums[ap_idx] + delta
                s[j] = -old_val
                current_max = new_max
                improved = True
                if current_max < best_score:
                    best_score = current_max
                    best_seq = s.copy()
        
        if not improved:
            break
        steps += 1
    
    # Phase 2: Try pairs of flips
    if time.time() < deadline:
        for _ in range(2000):
            if time.time() >= deadline:
                break
            j1 = rng.randint(0, N - 1)
            j2 = rng.randint(0, N - 1)
            if j1 == j2:
                continue
            
            old_val1 = s[j1]
            old_val2 = s[j2]
            delta1 = -2 * old_val1
            delta2 = -2 * old_val2
            
            ap1 = pos_to_ap_np[j1]
            ap2 = pos_to_ap_np[j2]
            
            if len(ap1) == 0 or len(ap2) == 0:
                continue
            
            # Compute combined effect
            old_max = current_max
            max_new_abs = 0
            
            # Build a map of AP index to sum change
            changes = {}
            for ap_idx in ap1:
                changes[ap_idx] = changes.get(ap_idx, 0) + delta1
            for ap_idx in ap2:
                changes[ap_idx] = changes.get(ap_idx, 0) + delta2
            
            for ap_idx, chg in changes.items():
                new_sum = S[ap_idx] + chg
                new_abs = abs(new_sum)
                if new_abs > max_new_abs:
                    max_new_abs = new_abs
            
            # Estimate new max
            affected_indices = list(changes.keys())
            old_affected_abs = [abs(S[idx]) for idx in affected_indices]
            
            # Find max among unaffected
            mask = np.ones(num_aps, dtype=bool)
            mask[affected_indices] = False
            if np.any(mask):
                unaffected_max = int(np.max(np.abs(S[mask])))
            else:
                unaffected_max = 0
            
            # Find max among affected after change
            new_affected_max = max_new_abs
            
            new_max = max(unaffected_max, new_affected_max)
            
            if new_max < old_max:
                # Apply
                for ap_idx, chg in changes.items():
                    S[ap_idx] += chg
                s[j1] = -old_val1
                s[j2] = -old_val2
                current_max = new_max
                if current_max < best_score:
                    best_score = current_max
                    best_seq = s.copy()
    
    return best_seq, best_score

def construct():
    deadline = time.time() + 55.0
    best_seq = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    start_temp = 8.0
    end_temp = 0.005
    cooling_rate = 0.997
    
    # Initial random solution
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
    max_steps_per_restart = 20000
    
    while time.time() < deadline and steps_in_restart < max_steps_per_restart:
        j = rng.randint(0, N - 1)
        
        old_val = s[j]
        delta = -2 * old_val
        
        ap_idx_list = pos_to_ap_np[j]
        
        if len(ap_idx_list) == 0:
            temp *= cooling_rate
            steps_in_restart += 1
            continue
        
        old_max = current_max
        max_new_abs_in_affected = 0
        
        # Vectorized update
        old_sums = S[ap_idx_list]
        new_sums = old_sums + delta
        
        old_abs = np.abs(old_sums)
        new_abs = np.abs(new_sums)
        
        # Update histogram
        for oa in old_abs:
            hist[oa] -= 1
        for na in new_abs:
            hist[na] += 1
        
        S[ap_idx_list] = new_sums
        
        for na in new_abs:
            if na > max_new_abs_in_affected:
                max_new_abs_in_affected = na
        
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
            # Revert
            S[ap_idx_list] = old_sums
            for na in new_abs:
                hist[na] -= 1
            for oa in old_abs:
                hist[oa] += 1
        
        temp *= cooling_rate
        steps_in_restart += 1
        
        if temp < end_temp:
            break
    
    # Final polish
    if time.time() < deadline - 5.0 and best_seq is not None:
        s2 = best_seq.copy()
        S2 = compute_initial_sums(s2)
        current_max2 = int(np.max(np.abs(S2)))
        
        # Rebuild hist for polish
        hist2 = np.zeros(N + 1, dtype=np.int32)
        abs_S2 = np.abs(S2)
        np.add.at(hist2, abs_S2, 1)
        
        polished_seq, polished_score = polish(s2, S2, current_max2, deadline - 2.0, rng)
        
        if polished_score < best_score:
            best_score = polished_score
            best_seq = polished_seq
    
    return [int(x) for x in best_seq]
