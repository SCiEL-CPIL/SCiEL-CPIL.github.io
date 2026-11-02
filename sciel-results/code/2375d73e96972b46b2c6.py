"""Combined: min-degree greedy + Tomita MIS + enhanced swap local search for kissing number d=11."""
import math
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    D = 11

    # Generate all candidates: weight-4 (5280), weight-2 (220), weight-1 (22)
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)  # 5522

    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # conflict[i,j] = 1 if i and j CANNOT both be selected
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # Fast Greedy Initialization
    def greedy_init():
        remaining = np.ones(n, dtype=bool)
        selected = []
        while remaining.any():
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[~remaining] = n + 1
            min_deg = deg.min()
            if min_deg >= n:
                break
            mask = (deg == min_deg) & remaining
            picks = np.where(mask)[0]
            if len(picks) == 0:
                break
            new_idx = picks[0]
            selected.append(new_idx)
            remaining = remaining & (conflict[new_idx] == 0)
            remaining[new_idx] = False
        return selected

    init_sel = greedy_init()
    x = np.zeros(n, dtype=np.uint8)
    for i in init_sel:
        x[i] = 1
    
    current_count = len(init_sel)
    current_violations = 0 # Greedy ensures validity
    
    # Track best valid solution
    best_x = x.copy()
    best_count = current_count
    
    # Simulated Annealing
    Lambda = 100.0
    T_start = 50.0
    T_end = 0.1
    max_steps = 150000
    
    for step in range(max_steps):
        if time.time() > deadline - 5.0:
            break
            
        alpha = (T_end / T_start) ** (1.0 / max_steps)
        T = T_start * (alpha ** step)
        
        move_type = np.random.random()
        old_count = current_count
        old_viol = current_violations
        old_E = Lambda * old_viol - old_count
        
        new_count = old_count
        new_viol = old_viol
        changes = [] # List of (index, old_val) to revert
        
        if move_type < 0.4:
            unselected = np.where(x == 0)[0]
            if len(unselected) == 0: continue
            i = unselected[np.random.randint(len(unselected))]
            delta_v = int(np.dot(conflict[i], x))
            new_viol = old_viol + delta_v
            new_count = old_count + 1
            changes.append((i, 0))
            
        elif move_type < 0.8:
            selected = np.where(x == 1)[0]
            if len(selected) == 0: continue
            i = selected[np.random.randint(len(selected))]
            delta_v = -int(np.dot(conflict[i], x))
            new_viol = old_viol + delta_v
            new_count = old_count - 1
            changes.append((i, 1))
            
        else:
            selected = np.where(x == 1)[0]
            unselected = np.where(x == 0)[0]
            if len(selected) == 0 or len(unselected) == 0: continue
            i_remove = selected[np.random.randint(len(selected))]
            i_add = unselected[np.random.randint(len(unselected))]
            
            viol_remove = int(np.dot(conflict[i_remove], x))
            viol_add_with_old = int(np.dot(conflict[i_add], x))
            delta_v = -viol_remove + (viol_add_with_old - int(conflict[i_add, i_remove]))
            new_viol = old_viol + delta_v
            new_count = old_count
            changes.append((i_remove, 1))
            changes.append((i_add, 0))
        
        new_E = Lambda * new_viol - new_count
        delta_E = new_E - old_E
        
        accepted = False
        if delta_E <= 0:
            accepted = True
        else:
            prob = math.exp(-delta_E / T)
            if np.random.random() < prob:
                accepted = True
        
        if accepted:
            for idx, old_val in changes:
                x[idx] = 1 - old_val
            
            current_violations = new_viol
            current_count = new_count
            
            if new_viol == 0 and new_count > best_count:
                best_count = new_count
                best_x = x.copy()
        
    if best_count == 0:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)
        
    result = candidates[best_x.astype(bool)]
    return result.astype(np.int64)
