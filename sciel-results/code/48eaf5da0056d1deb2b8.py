"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate all integer vectors with norm squared = 4 (norm 2)
    # These are the primary candidates for the kissing configuration.
    candidates = []
    
    # Type 1: Two entries are +/-1, rest 0.
    for i in range(D):
        for j in range(i + 1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
                    
    # Type 2: One entry is +/-2, rest 0.
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int64)
    N = len(candidates)
    
    if N == 0:
        return np.array([]).reshape(0, D)

    # Compute norms squared
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # The problem states: min pairwise distance >= max vector norm.
    # If we assume all vectors have the same norm (which is optimal for kissing numbers),
    # then max_norm_sq = 4.
    # Condition: ||v_i - v_j||^2 >= 4.
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2<v_i, v_j> = 8 - 2<v_i, v_j>.
    # 8 - 2<v_i, v_j> >= 4  =>  4 >= 2<v_i, v_j>  =>  <v_i, v_j> <= 2.
    
    # However, if we mix norms, the condition is harder.
    # Let's stick to norm=2 vectors for now, as they form the D11 lattice roots.
    
    # Compute dot product matrix
    dots = candidates @ candidates.T
    
    # Conflict if dot > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    def greedy_init():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.float64)
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] <= min_deg + 1e-9]
            
            # Tie-break: pick the one with fewest conflicts with other candidates?
            # Just pick the first for speed, or random.
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            to_remove = np.unique(np.concatenate(([chosen], neighbors[chosen])))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Update degrees
                # A vertex u in rem loses a neighbor if it was connected to chosen
                # But chosen is removed, so we just need to update degrees for the alive set.
                # Actually, the degree in the induced subgraph of 'alive' decreases by 1 for each neighbor of chosen.
                for u in neighbors[chosen]:
                    if alive[u]:
                        degrees[u] -= 1
                        
        return in_set

    def local_search(in_set_init, max_time=10.0):
        """Local search with add/swap moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        
        # Precompute conflict counts for current set
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
            
        ls_deadline = time.time() + max_time
        no_improve = 0
        T = 4.0
        cooling = 0.9995
        min_T = 0.01
        
        while time.time() < ls_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]
            
            if len(not_in_set) == 0 or len(in_set_indices) == 0:
                break
                
            move_type = rng.random()
            
            if move_type < 0.5:
                # ADD move
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = min_count
                delta = 1 - k
                
                accept = False
                if delta > 0: accept = True
                elif delta == 0: accept = rng.random() < 0.3
                else: accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
                    
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
            else:
                # SWAP move
                out_counts = count_conflicts[not_in_set]
                min_out = np.min(out_counts)
                min_out_cands = not_in_set[out_counts == min_out]
                v_add = min_out_cands[rng.integers(len(min_out_cands))]
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                
                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]
                
                if len(other_conflicts) == 0:
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    count_conflicts[neighbors[v_remove]] -= 1
                    count_conflicts[neighbors[v_add]] += 1
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                    delta = 1 - len(to_remove)
                    accept = rng.random() < np.exp(delta / T)
                    
                    if accept:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[neighbors[v_add]] += 1
                        
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            no_improve = 0
                        else:
                            no_improve += 1
                            
            if no_improve > 1000:
                # Perturbation
                if len(in_set_indices) > 5:
                    n_perturb = max(3, len(in_set_indices) // 10)
                    perturb_victims = in_set_indices[rng.choice(len(in_set_indices), size=n_perturb, replace=False)]
                    in_set[perturb_victims] = False
                    current_size -= len(perturb_victims)
                    for u in perturb_victims:
                        count_conflicts[neighbors[u]] -= 1
                    for v in perturb_victims:
                        if not np.any(conflicts[v] & in_set):
                            in_set[v] = True
                            current_size += 1
                            count_conflicts[neighbors[v]] += 1
                T = max(T, 3.0)
                no_improve = 0
                
            T *= cooling
            if T < min_T: T = min_T
            
        return local_best_in_set, local_best_size

    # Run multiple restarts
    n_restarts = 0
    max_restarts = 200
    
    while time.time() < deadline - 5 and n_restarts < max_restarts:
        n_restarts += 1
        
        in_set = greedy_init()
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
            
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 2: break
        
        ls_time = min(remaining_time * 0.5, 5.0)
        ls_result, ls_size = local_search(in_set, max_time=ls_time)
        
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
            
    final_indices = np.where(best_in_set)[0]
    return candidates[final_indices]

# EVOLVE-BLOCK-END
