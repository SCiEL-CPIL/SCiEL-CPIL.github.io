import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    # 1. Precomputation
    points_list = [(x, y) for x in range(N) for y in range(N)]
    num_points = N * N
    
    xs = np.array([p[0] for p in points_list], dtype=np.int32)
    ys = np.array([p[1] for p in points_list], dtype=np.int32)
    
    # Compute squared distances
    # dx = xs[:, None] - xs[None, :]
    # This creates a 4096x4096 array.
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    # Map point to index
    def get_idx(x, y):
        return x * N + y

    # 2. Initial Greedy Solution
    # We need a valid starting point for the SA.
    # The previous greedy logic is good for this.
    def greedy_init():
        indices = list(range(num_points))
        random.shuffle(indices)
        selected_indices = []
        dist_sets = [] # list of sets of distances from selected_indices[i] to others
        
        for idx in indices:
            if not selected_indices:
                selected_indices.append(idx)
                dist_sets.append(set())
                continue
                
            d_to_selected = dist_sq[idx, selected_indices]
            
            # Check 1: Distances from idx to selected must be distinct
            # If any distance repeats, adding idx creates an isosceles triangle with vertex idx
            if len(set(d_to_selected.tolist())) != len(d_to_selected):
                continue
            
            # Check 2: For each selected point j, dist(idx, j) must not be in dist_sets[j]
            # If dist(idx, j) is in dist_sets[j], it means there exists k in selected such that dist(j, k) == dist(j, idx)
            # This creates an isosceles triangle with vertex j.
            valid = True
            for i, sel_idx in enumerate(selected_indices):
                d = d_to_selected[i]
                if d in dist_sets[i]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Accept
            selected_indices.append(idx)
            dist_sets.append(set(d_to_selected.tolist()))
            
            # Update existing dist_sets
            for i, sel_idx in enumerate(selected_indices[:-1]):
                dist_sets[i].add(d_to_selected[i])
                
        return [points_list[i] for i in selected_indices]

    best_points = greedy_init()
    best_count = len(best_points)
    
    # If we have time, run another greedy to see if we get better
    if time.time() < deadline - 200:
        p2 = greedy_init()
        if len(p2) > best_count:
            best_points = p2
            best_count = len(best_points)

    # 3. Simulated Annealing
    # State:
    # S: set of indices
    # cnt: dict of dicts: cnt[A][d] = count of points in S at distance d from A
    # violations: dict: violations[A] = number of isosceles triangles with vertex A
    # V: total violations = sum(violations.values())
    
    def init_state(pts):
        S = set()
        for p in pts:
            S.add(get_idx(p[0], p[1]))
        
        cnt = {i: {} for i in S}
        violations = {i: 0 for i in S}
        V = 0
        
        # Build state
        S_list = list(S)
        for i in S_list:
            for j in S_list:
                if i == j: continue
                d = dist_sq[i, j]
                if d in cnt[i]:
                    # Adding this j to i's distance profile
                    # The number of pairs (j, k) with same distance d increases by current count
                    violations[i] += cnt[i][d]
                    cnt[i][d] += 1
                else:
                    cnt[i][d] = 1
        
        # Recalculate V to be sure
        V = sum(violations.values())
        return S, cnt, violations, V

    S, cnt, violations, V = init_state(best_points)
    current_size = len(S)
    lambda_factor = 100.0 # Weight for size in energy function
    current_energy = V - lambda_factor * current_size
    best_energy = current_energy
    best_valid_set = set(S)
    best_valid_size = current_size if V == 0 else 0
    
    # SA Parameters
    T_start = 1000.0
    T_end = 1.0
    num_iterations = 0
    max_iterations = 5000000 # Will be limited by time
    
    while time.time() < deadline - 10:
        # Decay temperature
        # Linear or exponential? Exponential is standard.
        # We don't know exactly how many iterations we can do, so we can just use a fixed schedule based on time or iterations.
        # Let's use a simple decay per iteration.
        # If we do 1M iters, T goes from 1000 to 1.
        # T = T_start * (T_end/T_start) ** (iter / max_iter)
        # But we don't know max_iter.
        # Let's just use T = T_start * 0.99999 ** num_iterations
        T = T_start * (0.99999 ** num_iterations)
        if T < T_end:
            T = T_end
            
        # Generate Move
        # 1. Add
        # 2. Remove
        # 3. Swap
        
        move_type = random.random()
        
        if move_type < 0.5 and len(S) < num_points:
            # Add
            candidates = [i for i in range(num_points) if i not in S]
            if not candidates:
                continue
            p = random.choice(candidates)
            
            # Calculate Delta
            delta_V = 0
            delta_size = 1
            
            # 1. Violations for p
            # Distances from p to all q in S
            d_vals = dist_sq[p, list(S)]
            # Count frequencies
            freq = {}
            for d in d_vals:
                freq[d] = freq.get(d, 0) + 1
            for c in freq.values():
                delta_V += c * (c - 1) // 2
            
            # 2. Violations for existing points
            for q in S:
                d = dist_sq[q, p]
                k = cnt[q].get(d, 0)
                delta_V += k
            
            delta_E = delta_V - lambda_factor * delta_size
            
            # Acceptance
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Apply Move
                S.add(p)
                # Update cnt and violations
                # For p
                cnt[p] = freq
                violations[p] = delta_V - sum(cnt[q].get(dist_sq[q,p],0) for q in S if q != p) # This is wrong, we already calculated delta_V
                # Actually, violations[p] is just the first part of delta_V
                # Let's store it properly
                v_p = 0
                for c in freq.values():
                    v_p += c * (c - 1) // 2
                violations[p] = v_p
                
                for q in S:
                    if q == p: continue
                    d = dist_sq[q, p]
                    k = cnt[q].get(d, 0)
                    violations[q] += k
                    cnt[q][d] = k + 1
                
                V += delta_V
                current_size += 1
                current_energy += delta_E
                
                if V == 0:
                    if current_size > best_valid_size:
                        best_valid_size = current_size
                        best_valid_set = set(S)
                        
        elif move_type < 0.8 and len(S) > 0:
            # Remove
            p = random.choice(list(S))
            
            # Calculate Delta
            delta_V = 0
            delta_size = -1
            
            # 1. Remove violations for p
            delta_V -= violations[p]
            
            # 2. Update violations for existing points
            for q in S:
                if q == p: continue
                d = dist_sq[q, p]
                k = cnt[q][d]
                # Removing p reduces count from k to k-1
                # Violations change by -(k-1)
                delta_V -= (k - 1)
            
            delta_E = delta_V - lambda_factor * delta_size
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Apply Move
                S.remove(p)
                # Update cnt and violations
                # For p
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
                
                if V == 0:
                    if current_size > best_valid_size:
                        best_valid_size = current_size
                        best_valid_set = set(S)
                        
        else:
            # Swap
            if len(S) > 0:
                candidates = [i for i in range(num_points) if i not in S]
                if not candidates:
                    continue
                p_remove = random.choice(list(S))
                p_add = random.choice(candidates)
                
                # This is complex. We can simulate Remove then Add.
                # But we need to check validity of the combined move.
                # For simplicity, let's just do sequential Remove and Add in the next iterations if needed.
                # Or just skip Swap for now to keep code robust.
                # Given the high dimension, Add/Remove might be sufficient.
                pass

        num_iterations += 1
        
    # Return best valid set
    if best_valid_size > 0:
        return [points_list[i] for i in best_valid_set]
    else:
        return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
