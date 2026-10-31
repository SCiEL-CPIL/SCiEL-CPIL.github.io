import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit (15s buffer)
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    # C(11,4) * 2^4 = 330 * 16 = 5280 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    N1 = len(vectors_v1)
    N = N4 + N1
    
    # Build the full conflict matrix for all N vectors
    # Two vectors conflict if their dot product > 2 (since norm^2 = 4, distance^2 = 8 - 2*dot)
    # We need distance >= norm, i.e., distance^2 >= norm^2 = 4
    # distance^2 = ||u||^2 + ||v||^2 - 2*u.v = 4 + 4 - 2*u.v = 8 - 2*u.v
    # We need 8 - 2*u.v >= 4 => u.v <= 2
    # So conflict if u.v > 2
    
    all_vectors = np.vstack([vectors_v4, vectors_v1])
    
    # Compute dot products
    dots = all_vectors @ all_vectors.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to int8 for faster matrix operations
    conflicts_int = conflicts.astype(np.int8)
    
    best_selected = np.zeros(N, dtype=bool)
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Strategy: Multiple approaches
    # 1. Greedy MIS with randomization
    # 2. Local search with add/swap operations
    # 3. Simulated annealing-style approach
    
    max_restarts = 50
    
    for r in range(max_restarts):
        if time.time() > deadline - 30:
            break
        
        # Randomized greedy MIS
        alive = np.ones(N, dtype=bool)
        selected_indices = []
        
        # Compute initial degrees
        degrees = conflicts_int.sum(axis=1).astype(np.int32).copy()
        
        # Randomly shuffle the order of processing for diversity
        order = np.arange(N)
        rng.shuffle(order)
        
        for idx in order:
            if time.time() > deadline - 30:
                break
            if not alive[idx]:
                continue
            
            # Select this vertex
            selected_indices.append(idx)
            
            # Remove it and its neighbors
            neighbors = np.where(conflicts_int[idx] > 0)[0]
            to_remove = np.unique(np.concatenate(([idx], neighbors)))
            alive[to_remove] = False
            
            # Update degrees for remaining vertices
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # For each remaining vertex, subtract 1 if it was connected to any removed vertex
                # conflicts_int[rem][:, to_remove] > 0 gives which removed vertices were neighbors
                deg_change = (conflicts_int[rem][:, to_remove] > 0).sum(axis=1)
                degrees[rem] -= deg_change
        
        # Local search on the selected set
        selected_mask = np.zeros(N, dtype=bool)
        selected_mask[selected_indices] = True
        unselected_mask = ~selected_mask
        
        # Phase 1: Add any compatible unselected vector
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            # For each unselected vertex, check if it conflicts with any selected vertex
            conflict_counts = conflicts_int.dot(selected_mask)
            valid_adds = (conflict_counts == 0) & unselected_mask
            
            if np.any(valid_adds):
                # Add one at a time to avoid conflicts between added vertices
                add_indices = np.where(valid_adds)[0]
                # Pick the one with fewest conflicts among unselected (heuristic)
                # Or just pick the first one
                idx = add_indices[0]
                selected_mask[idx] = True
                unselected_mask[idx] = False
                improved = True
        
        # Phase 2: Swap operations
        # Try to swap u in S with v in U if v has fewer conflicts with S than u does
        # This can enable adding more vertices later
        swapped = True
        while swapped and time.time() < deadline - 10:
            swapped = False
            S_indices = np.where(selected_mask)[0]
            U_indices = np.where(unselected_mask)[0]
            
            if len(S_indices) == 0 or len(U_indices) == 0:
                break
            
            # For each u in S, compute its conflict count with S
            # For each v in U, compute its conflict count with S
            # If v has 0 conflicts with S minus {u}, we can swap
            
            # Vectorized approach:
            # conflict_counts_S = conflicts_int.dot(selected_mask) gives conflicts with all of S
            # For u in S: conflicts with S minus {u} = conflict_counts_S[u] - (1 if u conflicts with itself, but we excluded diagonal)
            # Actually, conflict_counts_S[u] counts conflicts with all selected, including potentially itself? No, diagonal is 0.
            # So conflict_counts_S[u] is the number of selected vertices (other than u) that conflict with u.
            
            # For a swap u -> v to be valid: v must not conflict with any selected vertex except possibly u
            # i.e., conflicts_int[v].dot(selected_mask) - conflicts_int[v][u] == 0
            # i.e., v conflicts with no selected vertex other than possibly u
            
            conflict_with_S = conflicts_int.dot(selected_mask)  # shape (N,)
            
            # For each u in S, find v in U such that conflict_with_S[v] == conflicts_int[v, u]
            # (i.e., v's only conflict with S is with u, or no conflict at all)
            
            # This is expensive to do naively. Let's do a simpler heuristic:
            # Try random swaps and see if they help
            
            # Pick a random u from S
            u = S_indices[rng.integers(len(S_indices))]
            
            # Find v in U that has 0 conflicts with S minus {u}
            S_minus_u = selected_mask.copy()
            S_minus_u[u] = False
            
            conflict_with_S_minus_u = conflicts_int.dot(S_minus_u)
            valid_swaps = (conflict_with_S_minus_u == 0) & unselected_mask
            
            if np.any(valid_swaps):
                v_indices = np.where(valid_swaps)[0]
                v = v_indices[rng.integers(len(v_indices))]
                
                # Tentatively swap
                selected_mask[u] = False
                selected_mask[v] = True
                unselected_mask[u] = True
                unselected_mask[v] = False
                
                # Try to add
                conflict_counts_new = conflicts_int.dot(selected_mask)
                valid_adds_new = (conflict_counts_new == 0) & unselected_mask
                
                if np.any(valid_adds_new):
                    # Commit swap and add
                    add_idx = np.where(valid_adds_new)[0][0]
                    selected_mask[add_idx] = True
                    unselected_mask[add_idx] = False
                    swapped = True
                else:
                    # Revert swap
                    selected_mask[u] = True
                    selected_mask[v] = False
                    unselected_mask[u] = False
                    unselected_mask[v] = True
        
        count = np.sum(selected_mask)
        if count > best_count:
            best_count = count
            best_selected = selected_mask.copy()
        
        # Print progress occasionally
        if r % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Restart {r}: best={best_count}, current={count}, elapsed={elapsed:.1f}s")
    
    # Final local search on the best solution
    if time.time() < deadline - 5:
        selected_mask = best_selected.copy()
        unselected_mask = ~selected_mask
        
        # Add phase
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            conflict_counts = conflicts_int.dot(selected_mask)
            valid_adds = (conflict_counts == 0) & unselected_mask
            if np.any(valid_adds):
                idx = np.where(valid_adds)[0][0]
                selected_mask[idx] = True
                unselected_mask[idx] = False
                improved = True
        
        # Swap phase
        swapped = True
        while swapped and time.time() < deadline - 2:
            swapped = False
            S_indices = np.where(selected_mask)[0]
            U_indices = np.where(unselected_mask)[0]
            
            if len(S_indices) == 0 or len(U_indices) == 0:
                break
            
            u = S_indices[rng.integers(len(S_indices))]
            S_minus_u = selected_mask.copy()
            S_minus_u[u] = False
            
            conflict_with_S_minus_u = conflicts_int.dot(S_minus_u)
            valid_swaps = (conflict_with_S_minus_u == 0) & unselected_mask
            
            if np.any(valid_swaps):
                v_indices = np.where(valid_swaps)[0]
                v = v_indices[rng.integers(len(v_indices))]
                
                selected_mask[u] = False
                selected_mask[v] = True
                unselected_mask[u] = True
                unselected_mask[v] = False
                
                conflict_counts_new = conflicts_int.dot(selected_mask)
                valid_adds_new = (conflict_counts_new == 0) & unselected_mask
                
                if np.any(valid_adds_new):
                    add_idx = np.where(valid_adds_new)[0][0]
                    selected_mask[add_idx] = True
                    unselected_mask[add_idx] = False
                    swapped = True
                else:
                    selected_mask[u] = True
                    selected_mask[v] = False
                    unselected_mask[u] = False
                    unselected_mask[v] = True
        
        count = np.sum(selected_mask)
        if count > best_count:
            best_count = count
            best_selected = selected_mask.copy()
    
    result = all_vectors[best_selected]
    return result
