"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin

    # Generate candidates with squared norm 4
    candidates = []

    # Type 1: One 2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = s * 2
            candidates.append(v)

    # Type 2: Four ±1's with even sum (to ensure dot products are integers and structure is preserved)
    # Actually, the condition for kissing number in this context usually implies finding a maximal
    # set of vectors with min distance >= max norm.
    # If we restrict to norm^2=4, we need dot <= 2.
    # Let's include all Type 2 vectors. The parity constraint might be too restrictive or irrelevant
    # if we just check dot products. Let's generate all 4-subsets with signs.
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = [0] * D
            for pos, sign in zip(support, signs):
                v[pos] = sign
            candidates.append(v)

    n_cands = len(candidates)
    cand_arr = np.array(candidates, dtype=np.int32)

    # Precompute conflict matrix
    # Conflict if dot > 2
    dot_matrix = cand_arr @ cand_arr.T
    conflict = dot_matrix > 2
    np.fill_diagonal(conflict, False)
    
    # Convert to uint8 for faster boolean operations and summing
    conflict_u8 = conflict.astype(np.uint8)

    best_vecs = None
    best_count = 0

    # Helper function to run a greedy independent set algorithm
    def run_greedy(seed, use_spectral_init=False):
        if time.time() > deadline - 5:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(n_cands, dtype=bool)
        
        # Degrees: number of conflicts
        deg = conflict_u8.sum(axis=1).astype(np.int64)
        
        selected = []
        
        # Spectral initialization: 
        # If use_spectral_init, we can try to bias the selection towards nodes with low "spectral radius" 
        # or simply use the eigenvectors of the Laplacian to identify clusters.
        # For now, let's stick to a modified greedy that uses spectral info for tie-breaking or initial ordering.
        # A simpler structural change: Use the Fiedler vector (2nd smallest eigenvector of Laplacian) 
        # to partition the graph and solve subproblems? That's complex.
        
        # Let's try a different structural approach: 
        # Randomized Greedy with "Lookahead" or "Weighted" selection.
        # Or, simply run many randomized greedy passes with different tie-breaking strategies.
        
        # Strategy: 
        # 1. Compute a "score" for each node based on its degree and the degrees of its neighbors.
        # 2. At each step, pick the active node with the minimum score.
        
        # Let's implement a standard greedy MIS with random tie-breaking, but run it many times.
        # The inspiration mentions spectral clustering. Let's try to use the eigenvectors of the 
        # adjacency matrix to identify large cliques/clusters in the compatibility graph.
        # Compatibility graph edges = 1 - conflict.
        
        # However, computing eigenvectors of a 5000x5000 matrix might be slow.
        # Let's check the size.
        # Type 1: 22 vectors.
        # Type 2: C(11,4) * 16 = 330 * 16 = 5280 vectors.
        # Total ~ 5302 vectors.
        # 5300x5300 matrix eigendecomposition is O(N^3) ~ 1.5e11 ops. Too slow.
        
        # Alternative: Use power iteration to find the top eigenvectors?
        # Or, use a simpler heuristic: 
        # The "Spectral" idea might be overkill if we can just do better local search.
        # Let's combine the "Increased restarts and random tie-breaking" (from knowledge) 
        # with a "Local Search" refinement (from inspiration 1).
        
        # Greedy Pass
        while active.any():
            if time.time() > deadline - 5:
                break
                
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            
            # Get degrees of active nodes
            active_deg = deg[active_indices]
            
            # Find min degree
            min_deg = np.min(active_deg)
            
            # Find all active nodes with min degree
            candidates_for_pick = active_indices[active_deg == min_deg]
            
            # Pick one randomly
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            # Remove idx and its neighbors from active
            # Neighbors are those where conflict[idx] is True
            neighbors_mask = conflict_u8[idx].astype(bool)
            
            # Nodes to remove: idx and neighbors that are active
            removed_mask = np.zeros(n_cands, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            # Update degrees
            # For each remaining active node j, deg[j] -= number of removed neighbors
            removed_indices = np.where(removed_mask)[0]
            if len(removed_indices) > 0:
                # Count how many of the removed indices are neighbors of each active node
                # conflict_u8[removed_indices, :] has shape (K, N)
                # Summing along axis 0 gives the number of removed neighbors for each node j
                removal_counts = conflict_u8[removed_indices, :].sum(axis=0).astype(np.int64)
                
                # Update deg for active nodes
                deg[active] -= removal_counts[active]
            
            # Update active mask
            active[removed_mask] = False
            
        return selected

    # Run multiple greedy passes
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
            
        selected = run_greedy(seed=trial * 137 + 42)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_vecs = np.array([cand_arr[i] for i in selected], dtype=np.int64)

    # Local Search Refinement
    if best_vecs is not None and best_count > 0:
        # Map vector content to index for quick lookup? 
        # Or just work with indices.
        current_selected_set = set(selected)
        
        # Precompute conflict list for selected nodes?
        # We need to find a node i in selected, and nodes j, k not in selected 
        # such that:
        # 1. j, k are compatible with each other
        # 2. j, k are compatible with all nodes in selected \ {i}
        # 3. i is removed
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            selected_list = list(current_selected_set)
            rng_ls = np.random.default_rng()
            rng_ls.shuffle(selected_list)
            
            for i in selected_list:
                if time.time() > deadline - 5:
                    break
                
                # Find all nodes j not in selected that are compatible with selected \ {i}
                # A node j is compatible with selected \ {i} if it doesn't conflict with any k in selected \ {i}
                
                # To speed this up, we can precompute for each node j, the set of selected nodes it conflicts with.
                # But that's expensive to recompute.
                
                # Let's iterate through all unselected nodes.
                # For a node j to be a candidate for addition (after removing i), it must:
                # - Not be in selected
                # - Not conflict with any k in selected where k != i
                
                # Let's find the set of "blocked" nodes for the current selected set.
                # A node is blocked if it conflicts with at least one selected node.
                
                # Instead of checking all N nodes, let's look at neighbors of i.
                # If we remove i, a node j can be added if it was only blocked by i.
                # So j must conflict with i, and not conflict with any other selected node.
                
                neighbors_of_i = np.where(conflict_u8[i])[0]
                
                # Candidates j are neighbors of i that are not in selected
                # AND do not conflict with any other node in selected.
                
                compatible_with_rest = []
                for j in neighbors_of_i:
                    if j in current_selected_set:
                        continue
                    
                    # Check if j conflicts with any k in selected \ {i}
                    conflicts_with_other = False
                    for k in current_selected_set:
                        if k == i:
                            continue
                        if conflict_u8[j, k]:
                            conflicts_with_other = True
                            break
                    
                    if not conflicts_with_other:
                        compatible_with_rest.append(j)
                
                # Now, we have a list of nodes that can be added if we remove i.
                # We want to add at least 2 of them that are compatible with each other.
                # Or even 1 if it increases the count? No, we remove 1 and add 2 to net +1.
                # Or remove 1 and add 3 to net +2.
                
                # Let's try to find a pair (j1, j2) in compatible_with_rest that are compatible.
                # If we find one, we remove i and add j1, j2.
                
                # To make this efficient, we can check pairs.
                # If len(compatible_with_rest) is small, this is fast.
                
                found_pair = False
                n_cand = len(compatible_with_rest)
                for a_idx in range(n_cand):
                    if time.time() > deadline - 5:
                        break
                    j1 = compatible_with_rest[a_idx]
                    for b_idx in range(a_idx + 1, n_cand):
                        j2 = compatible_with_rest[b_idx]
                        # Check if j1 and j2 are compatible
                        if not conflict_u8[j1, j2]:
                            # We found a valid swap
                            current_selected_set.remove(i)
                            current_selected_set.add(j1)
                            current_selected_set.add(j2)
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                
                if found_pair:
                    break # Restart the while loop with the new set
            
            # Also try simple addition if possible (shouldn't happen if greedy was maximal, but maybe not)
            if not improved:
                # Try to add any node that is compatible with all selected
                # This is a "maximal" check. If greedy produced a maximal set, this should fail.
                # But if we did swaps, we might have created space for more?
                # Actually, if we swap 1->2, the set is still maximal w.r.t the new composition?
                # Not necessarily. We should check for addable nodes.
                
                # For efficiency, only check this occasionally or if the set is small.
                # Let's skip this to save time, as the swap logic should handle improvements.
                pass

        if len(current_selected_set) > best_count:
            best_count = len(current_selected_set)
            best_vecs = np.array([cand_arr[i] for i in sorted(current_selected_set)], dtype=np.int64)

    if best_vecs is None or len(best_vecs) == 0:
        # Fallback
        base_vecs = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    base_vecs.append(v)
        best_vecs = np.array(base_vecs, dtype=np.int64)

    return best_vecs
# EVOLVE-BLOCK-END
