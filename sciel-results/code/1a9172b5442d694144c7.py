"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def sa_local_search(in_set_init, max_time=10.0):
        """Min-conflict simulated annealing with 2-for-1 swaps and perturbation."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 3.0
        cooling = 0.9998
        min_T = 0.03
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            # Choose move type: 70% 1-for-0, 20% 2-for-1, 10% perturbation
            move_type = rng.random()
            
            if move_type < 0.70:
                # 1-for-0: add a vertex, remove its conflicts
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                    
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)
                    
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
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.90:
                # 2-for-1: remove one in-set vertex, try to add two non-conflicting vertices
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) < 1:
                    continue
                    
                # Pick a vertex to remove (prefer high-degree in the induced subgraph)
                remove_v = in_set_indices[rng.integers(len(in_set_indices))]
                
                # Find candidates that could be added (not in set, not conflicting with each other)
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    continue
                    
                # Filter candidates that don't conflict with each other
                # and don't conflict with the remaining in_set after removing remove_v
                remaining_in_set = in_set.copy()
                remaining_in_set[remove_v] = False
                
                # Candidates must not conflict with remaining_in_set
                cand_mask = np.zeros(N4, dtype=bool)
                for u in np.where(remaining_in_set)[0]:
                    cand_mask |= conflicts[u]
                candidates = np.where(~in_set & ~cand_mask)[0]
                
                if len(candidates) < 2:
                    continue
                    
                # Try to find a pair of non-conflicting candidates
                # Sample a few pairs to keep it fast
                max_pairs_to_try = 5
                found_pair = None
                for _ in range(max_pairs_to_try):
                    i, j = rng.integers(len(candidates), size=2)
                    if i == j:
                        continue
                    ci, cj = candidates[i], candidates[j]
                    if not conflicts[ci, cj]:
                        found_pair = (ci, cj)
                        break
                
                if found_pair is None:
                    stuck_counter += 1
                    continue
                    
                ci, cj = found_pair
                
                # Calculate delta: remove 1, add 2 => net +1
                # But we need to check if ci/cj conflict with in_set after removing remove_v
                # We already filtered, so they should be fine
                delta = 1  # net gain
                
                accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    # Remove remove_v
                    in_set[remove_v] = False
                    for u in neighbors[remove_v]:
                        count_conflicts[u] -= 1
                    
                    # Add ci and cj
                    for v_add in [ci, cj]:
                        to_remove = np.where(conflicts[v_add] & in_set)[0]
                        if len(to_remove) > 0:
                            in_set[to_remove] = False
                            for u in to_remove:
                                count_conflicts[neighbors[u]] -= 1
                        
                        in_set[v_add] = True
                        count_conflicts[neighbors[v_add]] += 1
                    
                    current_size = in_set.sum()
                    
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
                    
            else:
                # Perturbation: remove 3-5 random in-set vertices
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) < 3:
                    continue
                    
                n_remove = min(rng.integers(3, 6), len(in_set_indices))
                to_remove = in_set_indices[rng.choice(len(in_set_indices), size=n_remove, replace=False)]
                
                for v in to_remove:
                    in_set[v] = False
                    for u in neighbors[v]:
                        count_conflicts[u] -= 1
                
                current_size = in_set.sum()
                stuck_counter = 0
                # Increase temperature after perturbation
                T = max(T, 2.5)
            
            if stuck_counter > 1000:
                T = max(T, 3.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 80
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.5, 12.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Polish pass on best found so far
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 8 and sa_size > current_size:
            polish_time = min(remaining_time * 0.4, 10.0)
            if polish_time > 3:
                polish_result, polish_size = sa_local_search(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    # Final polish with longer time on the best solution
    remaining_time = deadline - 3 - time.time()
    if remaining_time > 5 and best_size > 0:
        final_polish_time = min(remaining_time * 0.8, 15.0)
        final_result, final_size = sa_local_search(best_in_set, max_time=final_polish_time)
        if final_size > best_size:
            best_size = final_size
            best_in_set = final_result
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
