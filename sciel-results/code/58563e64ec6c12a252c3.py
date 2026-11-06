import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 320
    
    # 1. Precomputation
    points_list = [(x, y) for x in range(N) for y in range(N)]
    num_points = N * N
    
    xs = np.array([p[0] for p in points_list], dtype=np.int32)
    ys = np.array([p[1] for p in points_list], dtype=np.int32)
    
    # Compute squared distances matrix
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    # Map point to index
    def get_idx(x, y):
        return x * N + y

    def valid_additions(S_set, cnt, violations):
        """Vectorized check: which points can be added without creating isosceles triangles?"""
        if not S_set:
            return np.arange(num_points)
        
        S_arr = np.array(list(S_set), dtype=np.int32)
        k = len(S_arr)
        
        # For each candidate point p, check:
        # 1. All distances from p to points in S must be distinct
        # 2. For each point q in S, dist(p,q) must not already be in cnt[q]
        
        # Compute distances from all candidates to all S points
        # dist_sq[candidates, S_arr] shape: (num_points, k)
        d_mat = dist_sq[:, S_arr]  # (num_points, k)
        
        # Check 1: distances from each candidate to S points must be distinct
        # Sort and check for duplicates
        d_sorted = np.sort(d_mat, axis=1)
        has_dup = np.any(d_sorted[:, 1:] == d_sorted[:, :-1], axis=1)
        
        # Check 2: for each q in S, dist(p,q) not in cnt[q]
        # cnt[q] is a dict of distance -> count
        # We need: for each candidate p and each S point q, dist_sq[p,q] not in cnt[q]
        # This is a bit tricky to vectorize fully, so we do it in a loop over S
        valid = ~has_dup.copy()
        
        for idx_q, q in enumerate(S_arr):
            if cnt[q]:  # if q has any distances recorded
                # For all candidates p, check if dist_sq[p,q] is in cnt[q]
                d_to_q = dist_sq[:, q]  # (num_points,)
                # Check membership in cnt[q]
                cnt_q = cnt[q]
                # Vectorized: check each distance value
                # This requires checking each unique distance
                # Alternative: build a boolean array of forbidden distances
                max_d = max(cnt_q.keys()) if cnt_q else 0
                if max_d < num_points * num_points:
                    forbidden = np.zeros(max_d + 1, dtype=bool)
                    for d_val in cnt_q:
                        forbidden[d_val] = True
                    # Now check: for each candidate p, is d_to_q[p] in forbidden?
                    # d_to_q[p] could be larger than max_d, in which case it's fine
                    d_clipped = np.minimum(d_to_q, max_d)
                    is_forbidden = forbidden[d_clipped] & (d_to_q <= max_d)
                    valid &= ~is_forbidden
        
        return np.where(valid)[0]

    def greedy_init():
        """Greedy construction with random ordering."""
        indices = list(range(num_points))
        random.shuffle(indices)
        selected = []
        cnt = {}
        violations = {}
        
        for idx in indices:
            if not selected:
                selected.append(idx)
                cnt[idx] = {}
                violations[idx] = 0
                continue
            
            # Check if idx can be added
            S_arr = np.array(selected, dtype=np.int32)
            d_to_sel = dist_sq[idx, S_arr]
            
            # Check 1: distinct distances
            if len(set(d_to_sel.tolist())) != len(d_to_sel):
                continue
            
            # Check 2: for each selected point, distance not already present
            valid = True
            for i, sel_idx in enumerate(selected):
                d = d_to_sel[i]
                if d in cnt[sel_idx]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Accept
            selected.append(idx)
            cnt[idx] = {}
            violations[idx] = 0
            for i, sel_idx in enumerate(selected[:-1]):
                d = d_to_sel[i]
                cnt[idx][d] = cnt[idx].get(d, 0) + 1
                cnt[sel_idx][d] = cnt[sel_idx].get(d, 0) + 1
                
        return [points_list[i] for i in selected]

    # Run multiple greedy initializations
    best_points = None
    best_count = 0
    time_per_greedy = 5.0
    max_greedy = int((deadline - start_time - 280) / time_per_greedy)
    max_greedy = max(1, min(max_greedy, 20))
    
    for _ in range(max_greedy):
        if time.time() > deadline - 280:
            break
        p = greedy_init()
        if len(p) > best_count:
            best_points = p
            best_count = len(p)
    
    if best_points is None:
        best_points = greedy_init()
        best_count = len(best_points)
    
    # 2. Local Search: repeatedly try to add valid points
    S_set = set(get_idx(p[0], p[1]) for p in best_points)
    cnt = {i: {} for i in S_set}
    violations = {i: 0 for i in S_set}
    
    # Build cnt
    S_list = list(S_set)
    for i in S_list:
        for j in S_list:
            if i == j:
                continue
            d = dist_sq[i, j]
            if d in cnt[i]:
                violations[i] += cnt[i][d]
                cnt[i][d] += 1
            else:
                cnt[i][d] = 1
    
    current_size = len(S_set)
    best_valid_size = current_size
    best_valid_set = set(S_set)
    
    # Local search: keep adding valid points until none remain
    while time.time() < deadline - 20:
        valid_pts = valid_additions(S_set, cnt, violations)
        if len(valid_pts) == 0:
            break
        
        # Pick a random valid point to add
        p = random.choice(valid_pts.tolist())
        
        # Add p
        S_set.add(p)
        cnt[p] = {}
        violations[p] = 0
        
        for q in S_set:
            if q == p:
                continue
            d = dist_sq[p, q]
            cnt[p][d] = cnt[p].get(d, 0) + 1
            cnt[q][d] = cnt[q].get(d, 0) + 1
        
        current_size += 1
        if current_size > best_valid_size:
            best_valid_size = current_size
            best_valid_set = set(S_set)
    
    # 3. Simulated Annealing with swap moves for further improvement
    S = set(best_valid_set)
    cnt = {i: {} for i in S}
    violations = {i: 0 for i in S}
    V = 0
    
    S_list = list(S)
    for i in S_list:
        for j in S_list:
            if i == j:
                continue
            d = dist_sq[i, j]
            if d in cnt[i]:
                violations[i] += cnt[i][d]
                cnt[i][d] += 1
            else:
                cnt[i][d] = 1
    V = sum(violations.values())
    
    current_size = len(S)
    lambda_factor = 50.0
    current_energy = V - lambda_factor * current_size
    best_energy = current_energy
    best_sa_set = set(S)
    best_sa_size = current_size if V == 0 else best_valid_size
    
    T_start = 500.0
    T_end = 0.1
    num_iterations = 0
    max_iter_time = deadline - 15 - time.time()
    
    if max_iter_time > 5:
        while time.time() < deadline - 15:
            T = T_start * (T_end / T_start) ** (num_iterations / 500000)
            if T < T_end:
                T = T_end
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a point
                candidates = [i for i in range(num_points) if i not in S]
                if not candidates:
                    num_iterations += 1
                    continue
                p = random.choice(candidates)
                
                # Calculate delta
                delta_V = 0
                d_vals = dist_sq[p, list(S)]
                freq = {}
                for d in d_vals:
                    freq[d] = freq.get(d, 0) + 1
                for c in freq.values():
                    delta_V += c * (c - 1) // 2
                
                for q in S:
                    d = dist_sq[q, p]
                    k = cnt[q].get(d, 0)
                    delta_V += k
                
                delta_E = delta_V - lambda_factor
                
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    S.add(p)
                    cnt[p] = freq
                    v_p = 0
                    for c in freq.values():
                        v_p += c * (c - 1) // 2
                    violations[p] = v_p
                    
                    for q in S:
                        if q == p:
                            continue
                        d = dist_sq[q, p]
                        k = cnt[q].get(d, 0)
                        violations[q] += k
                        cnt[q][d] = k + 1
                    
                    V += delta_V
                    current_size += 1
                    current_energy += delta_E
                    
                    if V == 0 and current_size > best_sa_size:
                        best_sa_size = current_size
                        best_sa_set = set(S)
                        
            elif move_type < 0.8:
                # Remove a point
                if not S:
                    num_iterations += 1
                    continue
                p = random.choice(list(S))
                
                delta_V = 0
                delta_V -= violations[p]
                for q in S:
                    if q == p:
                        continue
                    d = dist_sq[q, p]
                    k = cnt[q][d]
                    delta_V -= (k - 1)
                
                delta_E = delta_V + lambda_factor
                
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    S.remove(p)
                    del cnt[p]
                    del violations[p]
                    
                    for q in S:
                        d = dist_sq[q, p]
                        k = cnt[q][d]
                        violations[q] -= (k - 1)
                        cnt[q][d] = k - 1
                        if cnt[q][d] == 0:
                            del cnt[q][d]
                    
                    V += delta_V
                    current_size -= 1
                    current_energy += delta_E
                    
                    if V == 0 and current_size > best_sa_size:
                        best_sa_size = current_size
                        best_sa_set = set(S)
                        
            else:
                # Swap: remove one, add one
                if not S:
                    num_iterations += 1
                    continue
                candidates = [i for i in range(num_points) if i not in S]
                if not candidates:
                    num_iterations += 1
                    continue
                p_remove = random.choice(list(S))
                p_add = random.choice(candidates)
                
                # Calculate delta for remove
                delta_V_rem = 0
                delta_V_rem -= violations[p_remove]
                for q in S:
                    if q == p_remove:
                        continue
                    d = dist_sq[q, p_remove]
                    k = cnt[q][d]
                    delta_V_rem -= (k - 1)
                
                # Calculate delta for add (after removal)
                S_temp = S - {p_remove}
                delta_V_add = 0
                d_vals = dist_sq[p_add, list(S_temp)]
                freq = {}
                for d in d_vals:
                    freq[d] = freq.get(d, 0) + 1
                for c in freq.values():
                    delta_V_add += c * (c - 1) // 2
                
                for q in S_temp:
                    d = dist_sq[q, p_add]
                    k = cnt[q].get(d, 0)
                    # Adjust for the removed point
                    if d in cnt[q]:
                        # cnt[q][d] includes p_remove if dist_sq[p_remove, q] == d
                        if dist_sq[p_remove, q] == d:
                            k -= 1
                    delta_V_add += max(0, k)
                
                delta_V = delta_V_rem + delta_V_add
                delta_E = delta_V  # size doesn't change
                
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    # Apply swap
                    S.remove(p_remove)
                    del cnt[p_remove]
                    del violations[p_remove]
                    
                    for q in S:
                        d = dist_sq[q, p_remove]
                        k = cnt[q][d]
                        violations[q] -= (k - 1)
                        cnt[q][d] = k - 1
                        if cnt[q][d] == 0:
                            del cnt[q][d]
                    
                    S.add(p_add)
                    cnt[p_add] = {}
                    violations[p_add] = 0
                    v_add = 0
                    for c in freq.values():
                        v_add += c * (c - 1) // 2
                    violations[p_add] = v_add
                    
                    for q in S:
                        if q == p_add:
                            continue
                        d = dist_sq[q, p_add]
                        k = cnt[q].get(d, 0)
                        violations[q] += k
                        cnt[q][d] = k + 1
                    
                    V += delta_V
                    current_energy += delta_E
                    
                    if V == 0 and current_size > best_sa_size:
                        best_sa_size = current_size
                        best_sa_set = set(S)
            
            num_iterations += 1
    
    # Final greedy fill on the best SA result
    final_S = set(best_sa_set)
    final_cnt = {i: {} for i in final_S}
    final_violations = {i: 0 for i in final_S}
    S_list = list(final_S)
    for i in S_list:
        for j in S_list:
            if i == j:
                continue
            d = dist_sq[i, j]
            if d in final_cnt[i]:
                final_violations[i] += final_cnt[i][d]
                final_cnt[i][d] += 1
            else:
                final_cnt[i][d] = 1
    
    while time.time() < deadline - 5:
        valid_pts = valid_additions(final_S, final_cnt, final_violations)
        if len(valid_pts) == 0:
            break
        p = random.choice(valid_pts.tolist())
        final_S.add(p)
        final_cnt[p] = {}
        final_violations[p] = 0
        for q in final_S:
            if q == p:
                continue
            d = dist_sq[p, q]
            final_cnt[p][d] = final_cnt[p].get(d, 0) + 1
            final_cnt[q][d] = final_cnt[q].get(d, 0) + 1
    
    if len(final_S) > best_sa_size:
        best_sa_set = final_S
        best_sa_size = len(final_S)
    
    if best_sa_size >= best_valid_size:
        return [points_list[i] for i in best_sa_set]
    else:
        return [points_list[i] for i in best_valid_set]

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
