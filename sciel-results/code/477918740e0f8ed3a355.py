"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
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
        
    # Compute conflict matrix for W4
    # Conflict if u.v > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Compute degrees
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    rng = np.random.default_rng(42)
    
    def greedy_mis(conflicts, deadline, rng, degrees):
        """
        Greedy Maximum Independent Set with degree-based ordering.
        """
        N = len(conflicts)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()
        
        selected_indices = []
        
        while np.any(alive):
            if time.time() > deadline:
                break
                
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
                
            # Find min degree among alive
            min_deg = np.min(deg[alive_indices])
            candidates = alive_indices[deg[alive_indices] == min_deg]
            
            # Random tie-breaking
            chosen = candidates[rng.integers(len(candidates))]
            selected_indices.append(chosen)
            
            # Find neighbors to remove
            neighbors = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            
            # Update alive
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            remaining_alive = np.where(alive)[0]
            if len(remaining_alive) > 0:
                removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
                deg[remaining_alive] -= removed_conflicts

        return selected_indices
    
    def local_search_mis(conflicts, deadline, rng, initial_set):
        """
        Local Search for Maximum Independent Set using 1-swap and 2-swap moves.
        """
        N = len(conflicts)
        current_set = set(initial_set)
        current_size = len(current_set)
        best_set = set(initial_set)
        best_size = current_size
        
        # Precompute conflict bitsets for fast membership checks
        # conflicts is a boolean matrix N x N
        
        # For speed, represent the current set as a boolean array
        current_mask = np.zeros(N, dtype=bool)
        for v in current_set:
            current_mask[v] = True
            
        # Precompute list of conflicts for each node
        conflict_lists = [np.where(conflicts[i])[0] for i in range(N)]
        
        step = 0
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            step += 1
            
            # Try 1-swap: remove one, add one
            if current_size > 0:
                # Pick a random vertex to remove
                v_remove = list(current_set)[rng.integers(len(current_set))]
                
                # Compute which vertices can be added after removing v_remove
                # A vertex u can be added if it doesn't conflict with any other vertex in current_set - {v_remove}
                # This is equivalent to: u is not in current_set, and u is not a neighbor of any v in current_set - {v_remove}
                
                # Efficiently compute candidates:
                # Start with all vertices not in current_set
                candidates_mask = ~current_mask
                candidates_mask[v_remove] = True  # v_remove itself is not a candidate to add
                
                # For each v in current_set - {v_remove}, mark its neighbors as not candidates
                for v in current_set:
                    if v == v_remove:
                        continue
                    neighbors = conflict_lists[v]
                    candidates_mask[neighbors] = False
                    candidates_mask[v] = False  # Already in set
                
                candidates = np.where(candidates_mask)[0]
                
                if len(candidates) > 0:
                    # Pick a random candidate
                    v_add = candidates[rng.integers(len(candidates))]
                    
                    # Check if this swap is valid (it should be by construction)
                    # And if it improves (it won't for 1-swap, size stays same)
                    # So we accept it if it doesn't break anything (which it doesn't by construction)
                    # But we want to explore, so we accept with some probability or always for mixing
                    if rng.random() < 0.5:  # Accept half the time to explore
                        current_set.remove(v_remove)
                        current_set.add(v_add)
                        current_mask[v_remove] = False
                        current_mask[v_add] = True
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
            else:
                no_improve_count += 1
            
            # Try 2-swap: remove two, add two (can improve size)
            if current_size >= 2:
                # Pick two random vertices to remove
                v1_remove = list(current_set)[rng.integers(len(current_set))]
                remaining = list(current_set - {v1_remove})
                if remaining:
                    v2_remove = remaining[rng.integers(len(remaining))]
                    
                    # Compute candidates after removing both
                    candidates_mask = ~current_mask
                    candidates_mask[v1_remove] = True
                    candidates_mask[v2_remove] = True
                    
                    for v in current_set:
                        if v == v1_remove or v == v2_remove:
                            continue
                        neighbors = conflict_lists[v]
                        candidates_mask[neighbors] = False
                        candidates_mask[v] = False
                    
                    candidates = np.where(candidates_mask)[0]
                    
                    if len(candidates) >= 2:
                        # Pick two random candidates
                        idx1 = rng.integers(len(candidates))
                        idx2 = rng.integers(len(candidates))
                        while idx2 == idx1:
                            idx2 = rng.integers(len(candidates))
                        
                        v1_add = candidates[idx1]
                        v2_add = candidates[idx2]
                        
                        # Check if the two new vertices are compatible with each other
                        if not conflicts[v1_add, v2_add]:
                            # Accept if improves size (which it does: -2 + 2 = 0, wait, size stays same)
                            # Actually 2-swap keeps size same. To improve, we need asymmetric swaps.
                            # Let's do: remove 1, add 2 (if possible)
                            pass
            elif current_size == 1:
                # Try to add one more
                v_remove = list(current_set)[0]
                candidates_mask = ~current_mask
                candidates_mask[v_remove] = True
                
                # No other vertices to conflict with
                candidates = np.where(candidates_mask)[0]
                
                if len(candidates) > 0:
                    v_add = candidates[rng.integers(len(candidates))]
                    current_set.add(v_add)
                    current_mask[v_add] = True
                    current_size = len(current_set)
                    if current_size > best_size:
                        best_set = set(current_set)
                        best_size = current_size
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
            else:
                # Empty set, add a random vertex
                v_add = rng.integers(N)
                current_set.add(v_add)
                current_mask[v_add] = True
                current_size = 1
                no_improve_count += 1
        
        return list(best_set)
    
    # Run multiple greedy restarts to get good initial solutions
    best_greedy = []
    for i in range(5):
        if time.time() > deadline - 30:
            break
        greedy_result = greedy_mis(conflicts, min(deadline - 30, time.time() + 5), rng, degrees)
        if len(greedy_result) > len(best_greedy):
            best_greedy = greedy_result
    
    # Use local search to improve
    if best_greedy:
        ls_deadline = deadline - 5
        ls_result = local_search_mis(conflicts, ls_deadline, rng, best_greedy)
        if len(ls_result) > len(best_greedy):
            best_greedy = ls_result
    
    if not best_greedy:
        return vectors_v1
    
    result_v4 = vectors_v4[best_greedy]
    
    # Combine with W1
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
