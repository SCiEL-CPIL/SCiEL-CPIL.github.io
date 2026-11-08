"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

D = 11
# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 145  # safety margin from 150s

    # Generate all weight-4 vectors in R^11
    # A weight-4 vector has exactly 4 non-zero entries, each in {-1, 1}
    # Total count: C(11,4) * 2^4 = 330 * 16 = 5280
    vectors = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    n = len(vectors)
    
    if n < 2:
        return vectors
    
    # Compute Gram matrix
    G = vectors @ vectors.T
    
    # Build adjacency: conflict if G[i,j] > 2
    # For weight-4 vectors, dot product is in {-4, -3, -2, -1, 0, 1, 2, 3, 4}
    # Condition: min pairwise distance >= max norm
    # dist^2 = ||u||^2 + ||v||^2 - 2*u.v = 4 + 4 - 2*u.v = 8 - 2*u.v
    # max norm = sqrt(4) = 2, so max norm^2 = 4
    # We need dist >= max_norm => dist^2 >= max_norm^2 => 8 - 2*u.v >= 4 => 2*u.v <= 4 => u.v <= 2
    # So conflict if u.v > 2, i.e., u.v >= 3
    conflict = G > 2
    np.fill_diagonal(conflict, False)
    
    # We want the largest independent set in this conflict graph.
    # Strategy: Multiple restarts with different greedy heuristics.
    
    best_in_set = None
    best_count = 0
    
    # Pre-compute degrees for initial state
    initial_degrees = conflict.sum(axis=1)
    
    np.random.seed(42)  # Reproducibility
    
    # Phase 1: Vectorized random restarts (fast, diverse)
    # Use batched greedy: at each step, find all remaining vertices that don't conflict with current set
    num_random_restarts = 5000
    for r in range(num_random_restarts):
        if time.time() > deadline - 20:
            break
        
        # Random ordering
        order = np.random.permutation(n)
        
        in_set = np.zeros(n, dtype=bool)
        remaining = np.ones(n, dtype=bool)
        
        # Vectorized greedy: iterate through order, add if compatible
        # Instead of per-element loop, we can batch process
        # But for simplicity and speed, use a faster per-step approach
        
        count = 0
        # Pre-allocate for speed
        for idx in order:
            if not remaining[idx]:
                continue
            in_set[idx] = True
            count += 1
            # Remove neighbors from remaining using vectorized operation
            remaining[conflict[idx]] = False
            remaining[idx] = False
        
        if count > best_count:
            best_count = count
            best_in_set = in_set.copy()
        
        # Early termination if we've used too much time
        if time.time() > deadline - 20:
            break
    
    # Phase 2: Degree-ordered greedy (structured, often better than random)
    if time.time() < deadline - 15:
        # Sort by initial degree (ascending) - low degree vertices are "safer" to pick
        deg_order = np.argsort(initial_degrees)
        
        # Try multiple variants: pure degree, degree + noise
        for variant in range(50):
            if time.time() > deadline - 15:
                break
            
            if variant == 0:
                order = deg_order
            elif variant == 1:
                order = deg_order[::-1]  # high degree first
            else:
                # Add noise to degree ordering
                noise = np.random.uniform(0, 1, size=n)
                noisy_degrees = initial_degrees.astype(float) + noise * 10
                order = np.argsort(noisy_degrees)
            
            in_set = np.zeros(n, dtype=bool)
            remaining = np.ones(n, dtype=bool)
            
            count = 0
            for idx in order:
                if not remaining[idx]:
                    continue
                in_set[idx] = True
                count += 1
                remaining[conflict[idx]] = False
                remaining[idx] = False
            
            if count > best_count:
                best_count = count
                best_in_set = in_set.copy()
    
    # Phase 3: Local search - add back compatible vertices
    if time.time() < deadline - 5 and best_in_set is not None:
        in_set = best_in_set.copy()
        
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            
            # Find removed vertices that have no conflicts with current in_set
            has_conflict = conflict[:, in_set].any(axis=1)
            candidates = (~in_set) & ~has_conflict
            
            if not candidates.any():
                break
            
            # Add all candidates that don't conflict with each other (greedy)
            cand_indices = np.where(candidates)[0]
            added = 0
            for v_idx in cand_indices:
                if time.time() > deadline - 5:
                    break
                if not conflict[v_idx, in_set].any():
                    in_set[v_idx] = True
                    added += 1
            
            if added == 0:
                break
        
        best_count = in_set.sum()
        best_in_set = in_set
    
    # Phase 4: Swap-based local search
    if time.time() < deadline - 3 and best_in_set is not None:
        in_set = best_in_set.copy()
        in_set_list = np.where(in_set)[0]
        removed_list = np.where(~in_set)[0]
        
        if len(removed_list) > 0:
            # For each removed vertex, check if it can be added by removing some in_set vertices
            # Simple 1-for-1 swap: can we replace one in_set vertex with one removed vertex?
            # Better: can we add a removed vertex by removing a small set of in_set vertices?
            
            # Pre-compute: for each removed vertex, which in_set vertices does it conflict with?
            # conflict[v, in_set] gives the conflicts
            
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                
                # Try to add each removed vertex
                for v_idx in removed_list:
                    if time.time() > deadline - 3:
                        break
                    
                    # Which in_set vertices does v_idx conflict with?
                    conflicts_with_set = np.where(conflict[v_idx, in_set])[0]
                    
                    if len(conflicts_with_set) == 0:
                        # No conflicts, just add it
                        in_set[v_idx] = True
                        removed_list = np.where(~in_set)[0]
                        improved = True
                        continue
                    
                    # Try to remove conflicting vertices and add v_idx
                    # Only worth it if we remove fewer than we gain (but we gain 1, so need to remove 0)
                    # For 1-for-1: not useful since we remove 1 and add 1
                    # For 1-for-k: remove k in_set vertices to add 1 removed vertex - only if k=0
                    # So skip for now, or try 2-for-1: remove 2 in_set, add 1 removed + maybe another?
                    # This is complex. Let's try a different approach.
            
            # Alternative: try to find a removed vertex that conflicts with few in_set vertices
            # and see if we can "trade" it
            conflict_counts = np.array([np.sum(conflict[v, in_set]) for v in removed_list])
            
            # Sort removed vertices by conflict count (ascending)
            if len(removed_list) > 0:
                sorted_removed = removed_list[np.argsort(conflict_counts)]
                
                for v_idx in sorted_removed[:50]:  # Try top 50
                    if time.time() > deadline - 3:
                        break
                    
                    conflicts_with_set = np.where(conflict[v_idx, in_set])[0]
                    
                    if len(conflicts_with_set) == 0:
                        in_set[v_idx] = True
                        improved = True
                    elif len(conflicts_with_set) == 1:
                        # Remove the one conflicting vertex, add v_idx
                        # Net change: 0, but might enable other additions
                        # Skip for now as it's neutral
                        pass
        
        best_count = in_set.sum()
        best_in_set = in_set
    
    result = vectors[best_in_set]
    return result
# EVOLVE-BLOCK-END
