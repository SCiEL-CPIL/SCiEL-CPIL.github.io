import time
import numpy as np
import random
import math

N = 256

def construct():
    deadline = time.time() + 85.0
    rng = random.Random(12345)
    
    # Precompute all APs and their indices
    ap_indices_list = []
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                indices = tuple(a + i * d for i in range(l))
                ap_indices_list.append(indices)
    
    num_aps = len(ap_indices_list)
    
    # Build flat incidence structure: for each position, list of AP indices
    pos_to_ap = [[] for _ in range(N)]
    for ap_idx, indices in enumerate(ap_indices_list):
        for idx in indices:
            pos_to_ap[idx - 1].append(ap_idx)
    
    # Build flat arrays for fast vectorized access
    ap_starts = np.zeros(N + 1, dtype=np.int64)
    total_entries = 0
    for p in range(N):
        ap_starts[p] = total_entries
        total_entries += len(pos_to_ap[p])
    ap_starts[N] = total_entries
    
    ap_flat = np.zeros(total_entries, dtype=np.int32)
    for p in range(N):
        start = ap_starts[p]
        end = ap_starts[p + 1]
        ap_flat[start:end] = pos_to_ap[p]
    
    # Build AP membership for initial computation
    ap_member_starts = np.zeros(num_aps + 1, dtype=np.int64)
    total_members = 0
    for ap_idx, indices in enumerate(ap_indices_list):
        ap_member_starts[ap_idx] = total_members
        total_members += len(indices)
    ap_member_starts[num_aps] = total_members
    
    ap_members_flat = np.zeros(total_members, dtype=np.int32)
    for ap_idx, indices in enumerate(ap_indices_list):
        start = ap_member_starts[ap_idx]
        end = ap_member_starts[ap_idx + 1]
        ap_members_flat[start:end] = np.array(indices, dtype=np.int32) - 1
    
    def compute_initial_sums(s):
        S = np.zeros(num_aps, dtype=np.int16)
        for ap_idx in range(num_aps):
            start = ap_member_starts[ap_idx]
            end = ap_member_starts[ap_idx + 1]
            S[ap_idx] = np.sum(s[ap_members_flat[start:end]])
        return S
    
    def build_hist(S):
        hist = np.zeros(N + 1, dtype=np.int32)
        abs_S = np.abs(S)
        np.add.at(hist, abs_S, 1)
        return hist, int(np.max(abs_S))
    
    def try_flip(j, s, S, hist, current_max, temp, rng):
        old_val = s[j]
        delta = -2 * old_val
        
        start = ap_starts[j]
        end = ap_starts[j + 1]
        ap_idx_list = ap_flat[start:end]
        
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
            if temp > 0.01:
                p = math.exp(-cost_diff / temp)
                accept = (rng.random() < p)
            else:
                accept = False
        
        if accept:
            s[j] = -old_val
            return True, new_max, delta
        else:
            for ap_idx in ap_idx_list:
                new_sum = S[ap_idx]
                old_sum = new_sum - delta
                
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                hist[new_abs] -= 1
                hist[old_abs] += 1
                
                S[ap_idx] = old_sum
            
            return False, current_max, delta
    
    def greedy_improve(s, S, hist, current_max, deadline, rng, max_rounds=5):
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            for j in range(N):
                old_max = current_max
                old_val = s[j]
                delta = -2 * old_val
                
                start = ap_starts[j]
                end = ap_starts[j + 1]
                ap_idx_list = ap_flat[start:end]
                max_new_abs_in_affected = 0
                
                for ap_idx in ap_idx_list:
                    old_sum = S[ap_idx]
                    new_sum = old_sum + delta
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
                
                if new_max < current_max:
                    for ap_idx in ap_idx_list:
                        old_sum = S[ap_idx]
                        new_sum = old_sum + delta
                        
                        old_abs = abs(old_sum)
                        new_abs = abs(new_sum)
                        
                        hist[old_abs] -= 1
                        hist[new_abs] += 1
                        
                        S[ap_idx] = new_sum
                    
                    s[j] = -old_val
                    current_max = new_max
                    improved = True
            
            if not improved:
                break
        
        return s, S, hist, current_max
    
    def perturb(s, S, hist, current_max, rng, num_flips=8):
        for _ in range(num_flips):
            j = rng.randint(0, N - 1)
            old_val = s[j]
            delta = -2 * old_val
            
            start = ap_starts[j]
            end = ap_starts[j + 1]
            ap_idx_list = ap_flat[start:end]
            
            for ap_idx in ap_idx_list:
                old_sum = S[ap_idx]
                new_sum = old_sum + delta
                
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                hist[old_abs] -= 1
                hist[new_abs] += 1
                
                S[ap_idx] = new_sum
            
            s[j] = -old_val
        
        abs_S = np.abs(S)
        current_max = int(np.max(abs_S))
        hist = np.zeros(N + 1, dtype=np.int32)
        np.add.at(hist, abs_S, 1)
        
        return s, S, hist, current_max
    
    def greedy_sequential_construction(rng):
        """Build sequence position by position, choosing sign that minimizes current max discrepancy."""
        s = np.zeros(N, dtype=np.int8)
        S = np.zeros(num_aps, dtype=np.int16)
        hist = np.zeros(N + 1, dtype=np.int32)
        current_max = 0
        
        for pos in range(N):
            best_sign = 1
            best_max = float('inf')
            
            for sign in [1, -1]:
                # Tentatively assign sign
                s[pos] = sign
                delta = sign
                
                start = ap_starts[pos]
                end = ap_starts[pos + 1]
                ap_idx_list = ap_flat[start:end]
                
                max_new_abs = 0
                for ap_idx in ap_idx_list:
                    new_sum = S[ap_idx] + delta
                    new_abs = abs(new_sum)
                    if new_abs > max_new_abs:
                        max_new_abs = new_abs
                
                tentative_max = max(current_max, max_new_abs)
                
                if tentative_max < best_max:
                    best_max = tentative_max
                    best_sign = sign
            
            # Commit best sign
            s[pos] = best_sign
            delta = best_sign
            
            start = ap_starts[pos]
            end = ap_starts[pos + 1]
            ap_idx_list = ap_flat[start:end]
            
            for ap_idx in ap_idx_list:
                old_sum = S[ap_idx]
                new_sum = old_sum + delta
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                hist[old_abs] -= 1
                hist[new_abs] += 1
                S[ap_idx] = new_sum
            
            abs_S = np.abs(S)
            current_max = int(np.max(abs_S))
        
        return s, S, hist, current_max
    
    def local_search(s, S, hist, current_max, deadline, rng, max_steps=20000):
        temp = 3.0
        cooling_rate = 0.997
        steps = 0
        
        while time.time() < deadline and steps < max_steps:
            j = rng.randint(0, N - 1)
            
            accepted, new_max, _ = try_flip(j, s, S, hist, current_max, temp, rng)
            
            if accepted:
                current_max = new_max
            
            temp *= cooling_rate
            steps += 1
            
            if temp < 0.001:
                break
        
        return s, S, hist, current_max
    
    # Main search loop with multiple strategies
    best_seq = None
    best_score = float('inf')
    
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        
        # Strategy 1: Greedy sequential construction
        s, S, hist, current_max = greedy_sequential_construction(rng)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # Improve with greedy hill-climbing
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            s, S, hist, current_max = greedy_improve(s, S, hist, current_max, time.time() + remaining_time * 0.15, rng)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # Strategy 2: Simulated annealing
        remaining_time = deadline - time.time()
        if remaining_time > 3.0:
            sa_deadline = time.time() + remaining_time * 0.4
            s, S, hist, current_max = local_search(s, S, hist, current_max, sa_deadline, rng)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # Strategy 3: Perturbation + re-optimization
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            s, S, hist, current_max = perturb(s, S, hist, current_max, rng, num_flips=12)
            
            if time.time() < deadline - 3.0:
                s, S, hist, current_max = greedy_improve(s, S, hist, current_max, time.time() + (deadline - time.time()) * 0.25, rng)
            
            if time.time() < deadline - 2.0:
                s, S, hist, current_max = local_search(s, S, hist, current_max, time.time() + (deadline - time.time()) * 0.5, rng)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # Strategy 4: Random restart
        if time.time() > deadline - 10.0:
            s = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
            S = compute_initial_sums(s)
            hist, current_max = build_hist(S)
            
            if current_max < best_score:
                best_score = current_max
                best_seq = s.copy()
            
            if time.time() < deadline - 3.0:
                s, S, hist, current_max = greedy_improve(s, S, hist, current_max, time.time() + (deadline - time.time()) * 0.4, rng)
            
            if current_max < best_score:
                best_score = current_max
                best_seq = s.copy()
        
        if time.time() > deadline - 3.0:
            break
    
    # Final polish
    if best_seq is not None and time.time() < deadline - 2.0:
        s_final = best_seq.copy()
        S_final = compute_initial_sums(s_final)
        hist_final, current_max_final = build_hist(S_final)
        
        s_final, S_final, hist_final, current_max_final = greedy_improve(
            s_final, S_final, hist_final, current_max_final, deadline - 1.0, rng, max_rounds=10
        )
        
        if time.time() < deadline - 1.0:
            s_final, S_final, hist_final, current_max_final = local_search(
                s_final, S_final, hist_final, current_max_final, deadline - 0.5, rng, max_steps=5000
            )
        
        if current_max_final < best_score:
            best_score = current_max_final
            best_seq = s_final.copy()
    
    return [int(x) for x in best_seq]
