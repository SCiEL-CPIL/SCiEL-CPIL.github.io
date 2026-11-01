"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Strategy:
    # Use vectors with norm squared = 4 (Norm = 2).
    # Constraint: min pairwise dist >= max norm.
    # If all norm 2, need dist^2 >= 4 => u.v <= 2.
    
    # Type A: Weight 1 vectors in {-2, 0, 2}^11. Norm^2 = 4.
    # Count: 22. All compatible.
    
    # Type C: Weight 4 vectors in {-1, 0, 1}^11. Norm^2 = 4.
    # Count: 5280.
    # Compatible with Type A always.
    # Conflict between Type C if u.v > 2.
    
    # Generate W4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors
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
        
    # Compute conflict matrix
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    # Multiple restarts of Greedy + Min-Conflict SA
    n_restarts = 0
    max_restarts = 30
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Min-Degree Init
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int32)
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            neighbors_chosen = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], neighbors_chosen)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        current_size = in_set.sum()
        
        # 2. Min-Conflict Simulated Annealing
        T = 2.0
        cooling = 0.999
        min_T = 0.1
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in range(N4):
            if in_set[i]:
                count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        
        while time.time() < deadline - 2:
            # Find v not in set with min count_conflicts
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
                
            counts = count_conflicts[not_in_set]
            min_count = np.min(counts)
            
            # Candidates with min count
            min_cands = not_in_set[counts == min_count]
            v = min_cands[rng.integers(len(min_cands))]
            
            k = min_count
            delta = 1 - k
            
            accept = False
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.7
            else:
                accept = rng.random() < np.exp(delta / T)
                
            if accept:
                # Apply move: Add v, Remove neighbors in set
                to_remove = np.where(conflicts[v] & in_set)[0]
                
                if len(to_remove) > 0:
                    # Remove to_remove from in_set
                    in_set[to_remove] = False
                    # Update count_conflicts for neighbors of removed nodes
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                
                # Add v to in_set
                in_set[v] = True
                current_size += delta
                
                # Update count_conflicts for neighbors of v
                count_conflicts[neighbors[v]] += 1
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                    
                # Reheat if stuck
                if stuck_counter > 2000:
                    T = max(T, 1.0)
                    stuck_counter = 0
            else:
                stuck_counter += 1
                if stuck_counter > 2000:
                    T = max(T, 1.0)
                    stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        if local_best_size > best_size:
            best_size = local_best_size
            best_in_set = local_best_in_set.copy()
            
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
