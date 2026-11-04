import numpy as np
import itertools
import random
import time

D = 11
DEADLINE_S = 145.0  # Safety margin for 150s limit

def generate_candidates(max_norm_sq=5):
    vectors = []
    norms_sq = []
    
    # Norm squared 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
                norms_sq.append(2)
                
    # Norm squared 3
    for combo in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(v)
            norms_sq.append(3)
            
    # Norm squared 4
    # Type 1: four +/-1
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(v)
            norms_sq.append(4)
    # Type 2: one +/-2
    for i in range(D):
        for s in (2, -2):
            v = [0] * D
            v[i] = s
            vectors.append(v)
            norms_sq.append(4)
            
    # Norm squared 5
    # Type 1: five +/-1
    if max_norm_sq >= 5:
        for combo in itertools.combinations(range(D), 5):
            for signs in itertools.product([1, -1], repeat=5):
                v = [0] * D
                for idx, s in zip(combo, signs):
                    v[idx] = s
                vectors.append(v)
                norms_sq.append(5)
    # Type 2: one +/-2, one +/-1
    if max_norm_sq >= 5:
        for i in range(D):
            for j in range(D):
                if i == j: continue
                for s2 in (2, -2):
                    for s1 in (1, -1):
                        v = [0] * D
                        v[i] = s2
                        v[j] = s1
                        vectors.append(v)
                        norms_sq.append(5)

    V = np.array(vectors, dtype=np.int64)
    norms_sq_arr = np.array(norms_sq, dtype=np.int64)
    return V, norms_sq_arr

def construct():
    start_time = time.time()
    
    V, norms_sq = generate_candidates(max_norm_sq=5)
    N_total = len(V)
    
    best_k = 0
    best_set_indices = []
    
    # Try different R^2 values
    # R^2 = 2, 3, 4, 5
    for R_sq in [2, 3, 4, 5]:
        if time.time() - start_time > DEADLINE_S:
            break
            
        # Filter candidates with norm^2 <= R_sq
        mask = norms_sq <= R_sq
        indices = np.where(mask)[0]
        N = len(indices)
        if N == 0:
            continue
            
        V_sub = V[indices]
        norms_sub = norms_sq[indices]
        
        # Compute Gram matrix for this subset
        G = V_sub @ V_sub.T
        
        # Compute conflict mask
        sum_norms = norms_sub[:, None] + norms_sub[None, :]
        threshold = sum_norms - R_sq
        conflict = 2 * G > threshold
        np.fill_diagonal(conflict, False)
        
        # Degrees
        degrees = np.sum(conflict, axis=1)
        
        # Greedy Init
        order = np.argsort(degrees)
        selected = []
        selected_mask = np.zeros(N, dtype=bool)
        
        for idx in order:
            if time.time() - start_time > DEADLINE_S:
                break
            if selected_mask[idx]:
                continue
            if not np.any(conflict[idx] & selected_mask):
                selected.append(idx)
                selected_mask[idx] = True
                
        if len(selected) > best_k:
            best_k = len(selected)
            best_set_indices = [indices[i] for i in selected]
            
        # Local Search with Simulated Annealing
        selected_arr = np.array(selected)
        counts = conflict[:, selected_arr].sum(axis=1)
        
        # SA Parameters
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.995
        max_iterations = 20000
        
        current_size = len(selected)
        best_local_size = current_size
        best_local_indices = list(selected)
        
        for it in range(max_iterations):
            if time.time() - start_time > DEADLINE_S:
                break
            if T < T_min:
                break
            
            # Pick a random node from selected to remove
            if not selected:
                break
            u_pos = random.randint(0, len(selected) - 1)
            u = selected[u_pos]
            
            # Remove u
            selected.pop(u_pos)
            selected_mask[u] = False
            selected_arr = np.delete(selected_arr, u_pos)
            counts -= conflict[:, u]
            
            # Find addable nodes
            candidates = np.where(conflict[:, u])[0]
            valid_candidates = [v for v in candidates if not selected_mask[v] and counts[v] == 0]
            
            new_size = len(selected)
            
            if valid_candidates:
                # Greedily add as many as possible
                random.shuffle(valid_candidates)
                for v in valid_candidates:
                    if selected_mask[v]: continue
                    if not np.any(conflict[v] & selected_mask):
                        selected.append(v)
                        selected_mask[v] = True
                        selected_arr = np.append(selected_arr, v)
                        counts += conflict[:, v]
                        new_size += 1
            
            delta = new_size - current_size
            
            # Accept if improvement or with probability exp(-delta/T)
            if delta >= 0 or (delta < 0 and random.random() < np.exp(-delta / T)):
                current_size = new_size
                if current_size > best_local_size:
                    best_local_size = current_size
                    best_local_indices = list(selected)
            else:
                # Revert
                selected.insert(u_pos, u)
                selected_mask[u] = True
                selected_arr = np.insert(selected_arr, u_pos, u)
                counts += conflict[:, u]
                # Remove any nodes that were added (shouldn't be any if delta < 0 and we reverted)
            
            T *= cooling_rate
        
        # Update global best
        if best_local_size > best_k:
            best_k = best_local_size
            best_set_indices = [indices[i] for i in best_local_indices]

    if not best_set_indices:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    return V[best_set_indices]

if __name__ == "__main__":
    res = construct()
    print(f"Found {res.shape[0]} vectors")
