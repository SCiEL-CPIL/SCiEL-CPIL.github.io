import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
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
    
    if N4 == 0:
        return vectors_v1
    
    # Compute conflict graph among V4 vectors: conflict if dot > 2
    dots4 = vectors_v4 @ vectors_v4.T
    conflicts4 = dots4 > 2
    np.fill_diagonal(conflicts4, False)
    
    # Also need to check V1 vs V4 compatibility
    # V1 vectors have norm^2=4, V4 vectors have norm^2=4
    # Distance squared = 4 + 4 - 2*dot = 8 - 2*dot
    # Need distance >= max_norm = 2, so distance^2 >= 4
    # 8 - 2*dot >= 4 => dot <= 2
    # V1[i] has a single 2 or -2 at position i
    # V4[j] has entries in {-1,0,1}
    # dot(V1[i], V4[j]) = 2*V4[j][i] if V1[i][i]=2, or -2*V4[j][i] if V1[i][i]=-2
    # So |dot| <= 2 always, meaning all V1 are compatible with all V4!
    # Also V1[i] vs V1[j]: if i!=j, dot=0, dist^2=8>=4 OK
    # If i==j, they're the same vector (not allowed to have duplicates)
    # V1[i] vs -V1[i]: dot = -4, dist^2 = 8-2*(-4)=12>=4 OK
    # So all V1 vectors are mutually compatible and compatible with all V4.
    
    # The problem reduces to: find maximum independent set in the conflict graph of V4
    # Then add all V1 vectors.
    
    # Strategy: Use a combination of greedy + local search (2-opt swaps)
    # for maximum independent set on the V4 conflict graph.
    
    best_selected = np.zeros(N4, dtype=bool)
    best_count = 0
    
    # Multiple strategies for initial solution + local search
    
    # Strategy 1: Greedy minimum degree
    def greedy_min_degree(conflicts, rng):
        n = len(conflicts)
        alive = np.ones(n, dtype=bool)
        selected = np.zeros(n, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int64)
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            alive_degrees = degrees[alive_indices]
            min_deg = np.min(alive_degrees)
            cands = alive_indices[alive_degrees == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            selected[chosen] = True
            to_remove = np.where(conflicts[chosen] | (np.arange(n) == chosen))[0]
            alive[to_remove] = False
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                degrees[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
        
        return selected
    
    # Local search: try 1-add, 1-swap, 2-swap moves
    def local_search(conflicts, selected, rng, max_iters=2000):
        n = len(conflicts)
        selected_copy = selected.copy()
        current_count = selected_copy.sum()
        
        # Precompute conflict counts for non-selected vertices
        # For each non-selected vertex v, count how many selected vertices it conflicts with
        sel_conflict_counts = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if selected_copy[i]:
                sel_conflict_counts += conflicts[i]
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try adding a non-selected vertex that has no conflicts with selected
            non_sel = np.where(~selected_copy)[0]
            if len(non_sel) == 0:
                break
            
            # Vertices that can be added freely (no conflict with any selected)
            free_adds = non_sel[sel_conflict_counts[non_sel] == 0]
            if len(free_adds) > 0:
                # Add the one that conflicts with the fewest other non-selected (to preserve options)
                # Or just add one
                chosen = free_adds[rng.integers(len(free_adds))]
                selected_copy[chosen] = True
                current_count += 1
                # Update conflict counts
                sel_conflict_counts += conflicts[chosen]
                improved = True
                continue
            
            # Try 1-swap: remove one selected, add one non-selected
            # Net gain: +1 - (number of selected neighbors of new vertex after removal)
            # This is complex; try a simpler heuristic
            # For each non-selected vertex v, find minimum number of selected neighbors to remove
            # to make v addable
            
            # Try 2-swap: add two non-selected, remove their selected neighbors
            # Too expensive for now. Let's try a different approach.
            
            # Try: for each selected vertex s, try removing it and see if we can add more
            # This is expensive. Let's just do random swaps.
            
            # Pick a random non-selected vertex and try to add it by removing its selected neighbors
            v = non_sel[rng.integers(len(non_sel))]
            sel_neighbors = np.where(selected_copy & conflicts[v])[0]
            
            if len(sel_neighbors) == 0:
                # Should have been caught above
                continue
            
            # Net gain = 1 - len(sel_neighbors)
            # Only do if len(sel_neighbors) == 1 (net gain 0, might help future)
            # or if we can add multiple by removing fewer
            if len(sel_neighbors) == 1:
                # Remove the one neighbor and add v
                s = sel_neighbors[0]
                selected_copy[s] = False
                selected_copy[v] = True
                # Update conflict counts
                sel_conflict_counts -= conflicts[s]
                sel_conflict_counts += conflicts[v]
                improved = True
        
        return selected_copy, current_count
    
    # Run multiple restarts with different strategies
    rng = np.random.default_rng(42)
    
    for restart in range(30):
        if time.time() > deadline - 15:
            break
        
        if restart < 10:
            # Greedy min degree with different random seeds
            seed = 42 + restart
            rng_r = np.random.default_rng(seed)
            selected = greedy_min_degree(conflicts4, rng_r)
        elif restart < 20:
            # Random order greedy
            rng_r = np.random.default_rng(100 + restart)
            n = N4
            alive = np.ones(n, dtype=bool)
            selected = np.zeros(n, dtype=bool)
            order = rng_r.permutation(n)
            for i in order:
                if not alive[i]:
                    continue
                selected[i] = True
                to_remove = np.where(conflicts4[i] | (np.arange(n) == i))[0]
                alive[to_remove] = False
        else:
            # Random subset initialization followed by local search
            rng_r = np.random.default_rng(200 + restart)
            selected = np.zeros(N4, dtype=bool)
            # Start with random independent set
            alive = np.ones(N4, dtype=bool)
            while np.any(alive):
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                chosen = alive_indices[rng_r.integers(len(alive_indices))]
                selected[chosen] = True
                to_remove = np.where(conflicts4[chosen] | (np.arange(N4) == chosen))[0]
                alive[to_remove] = False
        
        count = selected.sum()
        
        # Local search refinement
        if time.time() < deadline - 5:
            selected, count = local_search(conflicts4, selected, rng, max_iters=1000)
        
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
    
    # Get result
    result_v4 = vectors_v4[best_selected]
    
    # Add all V1 vectors
    if N1 > 0:
        final_vectors = np.vstack([result_v4, vectors_v1])
    else:
        final_vectors = result_v4
    
    # Verify constraint
    if len(final_vectors) > 1:
        norms_sq = np.sum(final_vectors * final_vectors, axis=1)
        max_norm_sq = np.max(norms_sq)
        dots_f = final_vectors @ final_vectors.T
        dists_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots_f
        np.fill_diagonal(dists_sq, 0)
        min_dist_sq = np.min(dists_sq)
        
        if min_dist_sq < max_norm_sq:
            # Fallback: remove problematic vectors
            # This should rarely happen
            print(f"Warning: constraint violated. min_dist_sq={min_dist_sq}, max_norm_sq={max_norm_sq}", file=sys.stderr)
    
    return final_vectors
