import random
import time
import itertools
import math
import numpy as np

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)

# Precompute squared distances
coords = np.array(ALL_POINTS)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2 = np.sum(diff**2, axis=2)

# Precompute bad pairs for each point
BAD_PAIRS = [set() for _ in range(N_PTS)]

def _precompute_bad_pairs():
    i_idx, j_idx = np.triu_indices(N_PTS, k=1)
    for p in range(N_PTS):
        dp = dist2[p, :]
        dp_i = dp[i_idx]
        dp_j = dp[j_idx]
        dij = dist2[i_idx, j_idx]
        cond = (dp_i == dp_j) | (dp_i == dij) | (dp_j == dij)
        bad_indices = np.where(cond)[0]
        for k in bad_indices:
            u, v = i_idx[k], j_idx[k]
            if u > v:
                u, v = v, u
            BAD_PAIRS[p].add((u, v))

_start_time = time.time()
_precompute_bad_pairs()

def _can_add(p, S_set):
    """Check if p can be added to S_set (represented as a set of indices)."""
    for u, v in BAD_PAIRS[p]:
        if u in S_set and v in S_set:
            return False
    return True

def _get_addable(S_set):
    """Get set of points that can be added to S_set."""
    addable = set()
    for p in range(N_PTS):
        if p not in S_set:
            if _can_add(p, S_set):
                addable.add(p)
    return addable

def _greedy_initialization(deadline):
    """Greedy construction of a maximal no-isosceles set."""
    S = []
    S_set = set()
    C = set(range(N_PTS))
    
    while C:
        if time.time() > deadline:
            break
        
        best_p = -1
        best_new_C_size = float('inf')
        best_new_C = None
        
        candidates = list(C)
        random.shuffle(candidates)
        
        for p in candidates:
            new_C = set()
            for q in C:
                if q == p:
                    continue
                valid = True
                for u in S:
                    a, b = (p, u) if p < u else (u, p)
                    if (a, b) in BAD_PAIRS[q]:
                        valid = False
                        break
                if valid:
                    new_C.add(q)
            
            new_C_size = len(new_C)
            if new_C_size < best_new_C_size:
                best_new_C_size = new_C_size
                best_p = p
                best_new_C = new_C
        
        if best_p == -1:
            break
            
        S.append(best_p)
        S_set.add(best_p)
        C = best_new_C
        
    return S

def _local_search(S_initial, deadline, penalty_weight=10.0):
    """
    Local search to minimize |S| + penalty * N_addable.
    """
    S_set = set(S_initial)
    all_points_set = set(range(N_PTS))
    
    addable = _get_addable(S_set)
    current_energy = len(S_set) + penalty_weight * len(addable)
    best_set = set(S_set)
    best_energy = current_energy
    
    temp = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        if temp < 0.01:
            temp = 0.01
            
        # Generate a move: Prefer swaps that might reduce size
        if addable and S_set:
            # 70% Swap, 30% Add/Remove
            if random.random() < 0.7:
                p_out = random.choice(list(S_set))
                p_in = random.choice(list(addable))
                new_S_set = S_set - {p_out} | {p_in}
            else:
                if random.random() < 0.5 and addable:
                    p = random.choice(list(addable))
                    new_S_set = S_set | {p}
                elif S_set:
                    p = random.choice(list(S_set))
                    new_S_set = S_set - {p}
                else:
                    continue
        elif addable:
            p = random.choice(list(addable))
            new_S_set = S_set | {p}
        elif S_set:
            p = random.choice(list(S_set))
            new_S_set = S_set - {p}
        else:
            break
            
        new_addable_set = _get_addable(new_S_set)
        new_energy = len(new_S_set) + penalty_weight * len(new_addable_set)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or random.random() < math.exp(-delta_e / temp):
            S_set = new_S_set
            addable = new_addable_set
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_set = set(S_set)
        
        temp *= cooling
        
    # Ensure maximality
    final_S = set(best_set)
    while True:
        addable = _get_addable(final_S)
        if not addable:
            break
        p = random.choice(list(addable))
        final_S.add(p)
        
    return list(final_S)

def construct():
    start_time = time.time()
    deadline = start_time + 300.0
    
    best_set = None
    best_size = float('inf')
    
    # Iterated Local Search
    current_best_init = None
    
    while time.time() < deadline - 5.0:
        if current_best_init is None:
            S_init = _greedy_initialization(deadline - 5.0)
        else:
            # Perturb current best
            S_init = list(current_best_init)
            # Randomly swap 1-2 points
            num_swaps = random.randint(1, 2)
            for _ in range(num_swaps):
                if len(S_init) > 1:
                    idx1, idx2 = random.sample(range(len(S_init)), 2)
                    S_init[idx1], S_init[idx2] = S_init[idx2], S_init[idx1]
                else:
                    # If small, just random restart
                    S_init = _greedy_initialization(deadline - 5.0)
                    break
        
        S_opt = _local_search(S_init, deadline - 2.0)
        
        if len(S_opt) < best_size:
            best_size = len(S_opt)
            best_set = S_opt
            current_best_init = S_opt
        elif best_set is None:
            best_set = S_opt
            current_best_init = S_opt
            
    if best_set is None:
        best_set = _greedy_initialization(deadline)
        
    return [ALL_POINTS[i] for i in best_set]
