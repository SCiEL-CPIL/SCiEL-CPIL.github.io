import numpy as np
import itertools
import time
import sys

D = 11

def generate_vectors(weight):
    """Generate all vectors of given weight in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), weight):
        for signs in itertools.product([1, -1], repeat=weight):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def greedy_mis_with_clique_coverage(conflicts, deadline, rng):
    """
    Greedy MIS with clique coverage heuristic.
    Prioritize vertices that appear in the most maximal-ish conflict cliques,
    then remove them and their neighbors.
    """
    N = len(conflicts)
    if N == 0:
        return []
    
    alive = np.ones(N, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    # Precompute neighbor lists as boolean masks for fast operations
    # conflicts is already boolean-like (0/1)
    
    selected_indices = []
    
    while np.any(alive):
        if time.time() > deadline:
            break
        
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        # Clique coverage heuristic: score = degree + (number of high-degree neighbors)
        # This approximates how many "cliques" a vertex participates in
        alive_degrees = degrees[alive_indices]
        
        # Count how many alive neighbors have degree >= median
        median_deg = np.median(alive_degrees) if len(alive_degrees) > 0 else 0
        high_deg_neighbors = (conflicts[alive_indices] @ (degrees >= median_deg).astype(np.int32))
        
        # Combined score: primary is degree, secondary is high-degree neighbor count
        # We want to REMOVE vertices that are in many conflicts, so pick high-degree first
        # But for MIS, we SELECT a vertex and remove its neighbors.
        # Strategy: select vertex with minimum "clique coverage" to preserve options
        # Actually, standard greedy MIS: select min-degree vertex.
        # Let's try: select vertex that minimizes (degree + high_deg_neighbors/2)
        
        scores = alive_degrees.astype(np.float64) + 0.5 * high_deg_neighbors
        min_score_idx = np.argmin(scores)
        chosen = alive_indices[min_score_idx]
        
        selected_indices.append(chosen)
        
        # Remove chosen and its neighbors
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        alive[to_remove] = False
        
        # Update degrees
        remaining_alive = np.where(alive)[0]
        if len(remaining_alive) > 0:
            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[to_remove] = True
            degrees[remaining_alive] -= conflicts[remaining_alive][:, to_remove].sum(axis=1)
    
    return selected_indices

def local_search_mis(conflicts, initial_solution, deadline, rng, max_iterations=500):
    """
    Local search: try to add vertices from the complement, or swap.
    """
    N = len(conflicts)
    if N == 0:
        return list(initial_solution)
    
    selected_set = set(initial_solution)
    selected_arr = np.array(sorted(selected_set)) if selected_set else np.array([], dtype=int)
    in_set = np.zeros(N, dtype=bool)
    for i in selected_set:
        in_set[i] = True
    
    best_size = len(selected_set)
    best_set = list(selected_set)
    
    # Find vertices not in set
    not_in_set = np.where(~in_set)[0]
    
    # Try adding vertices that don't conflict with any selected
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        
        if len(not_in_set) == 0:
            break
        
        # For each not-in-set vertex, check if it can be added
        # A vertex v can be added if it has no conflicts with any selected vertex
        # conflicts[v] & in_set should be all zeros
        
        # Vectorized: for each candidate, count conflicts with selected
        if len(selected_arr) > 0:
            conflict_counts = conflicts[not_in_set][:, selected_arr].sum(axis=1)
            addable = np.where(conflict_counts == 0)[0]
        else:
            addable = not_in_set
        
        if len(addable) == 0:
            # Try swapping: remove one selected, add one or more not-in-set
            # This is complex; for now, just stop
            break
        
        # Add all addable vertices at once (they don't conflict with selected,
        # but they might conflict with each other)
        # To be safe, add one at a time greedily
        rng.shuffle(addable)
        
        added_any = False
        for idx in addable:
            v = not_in_set[idx]
            # Check if v conflicts with any currently selected
            if len(selected_arr) > 0:
                if conflicts[v][:, selected_arr].any():
                    continue
            
            # Check if v conflicts with any other vertex we're about to add
            # For simplicity, just add if no conflict with current selected
            in_set[v] = True
            selected_set.add(v)
            selected_arr = np.array(sorted(selected_set))
            added_any = True
        
        if added_any:
            best_size = len(selected_set)
            best_set = list(selected_set)
            
            # Update not_in_set
            not_in_set = np.where(~in_set)[0]
        else:
            break
    
    return best_set

def construct():
    t0 = time.time()
    deadline = t0 + 145  # Leave 5s margin for safety
    
    rng = np.random.default_rng(42)
    
    # 1. Generate W4 vectors
    w4_vectors = generate_vectors(4)
    N4 = len(w4_vectors)
    
    # 2. Build conflict graph for W4
    # Conflict if dot product >= 3 (distance^2 = 4+4-2*dot < 4*4=16, i.e., dot > 2)
    # Wait, let me re-derive.
    # Norm of W4 vector = sqrt(4) = 2
    # Norm squared = 4
    # Distance^2 = 4 + 4 - 2*u.v = 8 - 2*u.v
    # We need Distance^2 >= max_norm^2 = 4
    # 8 - 2*u.v >= 4
    # 4 >= 2*u.v
    # u.v <= 2
    # So conflict if u.v >= 3
    
    dots4 = w4_vectors @ w4_vectors.T
    conflicts4 = (dots4 >= 3).astype(np.int8)
    np.fill_diagonal(conflicts4, 0)
    
    # 3. Solve MIS on W4
    best_set4 = []
    
    # Multiple greedy restarts with different heuristics
    for i in range(30):
        if time.time() > deadline - 20:
            break
        cur_set = greedy_mis_with_clique_coverage(conflicts4, deadline - 20, rng)
        if len(cur_set) > len(best_set4):
            best_set4 = cur_set
    
    # Local search polish
    if time.time() < deadline - 5 and len(best_set4) > 0:
        improved_set4 = local_search_mis(conflicts4, best_set4, deadline - 5, rng)
        if len(improved_set4) > len(best_set4):
            best_set4 = improved_set4
    
    selected_w4 = w4_vectors[best_set4] if best_set4 else np.empty((0, D), dtype=np.int8)
    
    # 4. Add W3 vectors
    w3_vectors = generate_vectors(3)
    N3 = len(w3_vectors)
    
    # W3 norm = sqrt(3), norm^2 = 3
    # Distance^2 = 3 + 3 - 2*u.v = 6 - 2*u.v
    # Need >= max_norm^2. If max norm is from W4 (norm^2=4), need 6-2*u.v >= 4 => u.v <= 1
    # If only W3, max norm^2 = 3, need 6-2*u.v >= 3 => u.v <= 1.5 => u.v <= 1
    # So W3-W3 conflict if u.v >= 2
    # W3-W4 conflict if u.v >= 2 (since 3+4-2*u.v >= 4 => 3 >= 2*u.v => u.v <= 1.5 => u.v <= 1)
    # Wait: 3 + 4 - 2*u.v >= 4 => 3 >= 2*u.v => u.v <= 1.5. Since integer, u.v <= 1.
    # So conflict if u.v >= 2
    
    if len(selected_w4) > 0:
        dots34 = w3_vectors @ selected_w4.T
        compatible3 = (dots34 < 2).all(axis=1)
    else:
        compatible3 = np.ones(N3, dtype=bool)
        
    w3_candidates = w3_vectors[compatible3]
    N3c = len(w3_candidates)
    
    if N3c > 0:
        dots33 = w3_candidates @ w3_candidates.T
        conflicts33 = (dots33 >= 2).astype(np.int8)
        np.fill_diagonal(conflicts33, 0)
        
        best_set3 = []
        for i in range(20):
            if time.time() > deadline - 10:
                break
            cur_set = greedy_mis_with_clique_coverage(conflicts33, deadline - 10, rng)
            if len(cur_set) > len(best_set3):
                best_set3 = cur_set
        
        if time.time() < deadline - 5 and len(best_set3) > 0:
            improved_set3 = local_search_mis(conflicts33, best_set3, deadline - 5, rng)
            if len(improved_set3) > len(best_set3):
                best_set3 = improved_set3
        
        selected_w3 = w3_candidates[best_set3] if best_set3 else np.empty((0, D), dtype=np.int8)
    else:
        selected_w3 = np.empty((0, D), dtype=np.int8)
        
    # 5. Add W2 vectors
    w2_vectors = generate_vectors(2)
    N2 = len(w2_vectors)
    
    # W2 norm^2 = 2
    # W2-W2: 2+2-2*u.v >= 4 => 4-2*u.v >= 4 => u.v <= 0. Conflict if u.v >= 1
    # W2-W3: 2+3-2*u.v >= 4 => 5-2*u.v >= 4 => u.v <= 0.5 => u.v <= 0. Conflict if u.v >= 1
    # W2-W4: 2+4-2*u.v >= 4 => 6-2*u.v >= 4 => u.v <= 1. Conflict if u.v >= 2
    
    if len(selected_w4) > 0:
        dots24 = w2_vectors @ selected_w4.T
        comp24 = (dots24 < 2).all(axis=1)
    else:
        comp24 = np.ones(N2, dtype=bool)
        
    if len(selected_w3) > 0:
        dots23 = w2_vectors @ selected_w3.T
        comp23 = (dots23 < 1).all(axis=1)  # u.v <= 0, so conflict if u.v >= 1
    else:
        comp23 = np.ones(N2, dtype=bool)
        
    compatible2 = comp24 & comp23
    w2_candidates = w2_vectors[compatible2]
    N2c = len(w2_candidates)
    
    if N2c > 0:
        dots22 = w2_candidates @ w2_candidates.T
        conflicts22 = (dots22 >= 1).astype(np.int8)
        np.fill_diagonal(conflicts22, 0)
        
        best_set2 = []
        for i in range(20):
            if time.time() > deadline - 5:
                break
            cur_set = greedy_mis_with_clique_coverage(conflicts22, deadline - 5, rng)
            if len(cur_set) > len(best_set2):
                best_set2 = cur_set
        
        if time.time() < deadline - 2 and len(best_set2) > 0:
            improved_set2 = local_search_mis(conflicts22, best_set2, deadline - 2, rng)
            if len(improved_set2) > len(best_set2):
                best_set2 = improved_set2
        
        selected_w2 = w2_candidates[best_set2] if best_set2 else np.empty((0, D), dtype=np.int8)
    else:
        selected_w2 = np.empty((0, D), dtype=np.int8)
        
    # 6. Add W1 vectors (weight 1, norm^2 = 1)
    w1_vectors = generate_vectors(1)
    N1 = len(w1_vectors)
    
    # W1-W1: 1+1-2*u.v >= 4 => 2-2*u.v >= 4 => -2*u.v >= 2 => u.v <= -1
    # u.v is -1, 0, or 1. So conflict if u.v > -1, i.e., u.v >= 0
    # W1-W2: 1+2-2*u.v >= 4 => 3-2*u.v >= 4 => -2*u.v >= 1 => u.v <= -0.5 => u.v <= -1
    # Conflict if u.v >= 0
    # W1-W3: 1+3-2*u.v >= 4 => 4-2*u.v >= 4 => u.v <= 0. Conflict if u.v >= 1
    # W1-W4: 1+4-2*u.v >= 4 => 5-2*u.v >= 4 => u.v <= 0.5 => u.v <= 0. Conflict if u.v >= 1
    
    if len(selected_w4) > 0:
        dots14 = w1_vectors @ selected_w4.T
        comp14 = (dots14 < 1).all(axis=1)
    else:
        comp14 = np.ones(N1, dtype=bool)
        
    if len(selected_w3) > 0:
        dots13 = w1_vectors @ selected_w3.T
        comp13 = (dots13 < 1).all(axis=1)
    else:
        comp13 = np.ones(N1, dtype=bool)
        
    if len(selected_w2) > 0:
        dots12 = w1_vectors @ selected_w2.T
        comp12 = (dots12 < 0).all(axis=1)  # u.v <= -1, so conflict if u.v >= 0
    else:
        comp12 = np.ones(N1, dtype=bool)
        
    compatible1 = comp14 & comp13 & comp12
    w1_candidates = w1_vectors[compatible1]
    N1c = len(w1_candidates)
    
    if N1c > 0:
        dots11 = w1_candidates @ w1_candidates.T
        conflicts11 = (dots11 >= 0).astype(np.int8)
        np.fill_diagonal(conflicts11, 0)
        
        best_set1 = []
        for i in range(20):
            if time.time() > deadline - 3:
                break
            cur_set = greedy_mis_with_clique_coverage(conflicts11, deadline - 3, rng)
            if len(cur_set) > len(best_set1):
                best_set1 = cur_set
        
        if time.time() < deadline - 1 and len(best_set1) > 0:
            improved_set1 = local_search_mis(conflicts11, best_set1, deadline - 1, rng)
            if len(improved_set1) > len(best_set1):
                best_set1 = improved_set1
        
        selected_w1 = w1_candidates[best_set1] if best_set1 else np.empty((0, D), dtype=np.int8)
    else:
        selected_w1 = np.empty((0, D), dtype=np.int8)
        
    # Combine results
    parts = [selected_w4, selected_w3, selected_w2, selected_w1]
    parts = [p for p in parts if len(p) > 0]
    
    if len(parts) == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    all_vectors = np.vstack(parts)
    
    # Final validation: check that min pairwise distance >= max norm
    # For safety, if validation fails, return a valid subset
    if len(all_vectors) > 1:
        norms_sq = (all_vectors ** 2).sum(axis=1)
        max_norm_sq = norms_sq.max()
        
        # Compute pairwise distances
        # dist^2 = ||u||^2 + ||v||^2 - 2*u.v
        dots = all_vectors @ all_vectors.T
        dists_sq = norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - 2 * dots
        np.fill_diagonal(dists_sq, np.inf)
        min_dist_sq = dists_sq.min()
        
        if min_dist_sq < max_norm_sq - 1e-9:
            # Validation failed. This shouldn't happen if our logic is correct.
            # Fallback: return just W4 vectors
            if len(selected_w4) > 0:
                return selected_w4.astype(np.int64)
            return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    return all_vectors.astype(np.int64)
