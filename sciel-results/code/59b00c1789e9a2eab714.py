import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # 1. Generate candidate vectors
    # Class 1: Weight 4 in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    # Class 2: Weight 2 in {-1, 0, 1}^11 (Norm^2 = 2)
    # These are compatible with all v4 vectors (dot product in {-1, 0, 1})
    # and with v1 vectors (dot product in {-2, 0, 2}, dist^2 = 2+4-2*dot >= 0? 
    # Wait. v1 has norm 2. v2 has norm sqrt(2).
    # dist^2(v1, v2) = 4 + 2 - 2*v1.v2.
    # v1 is (2, 0...), v2 is (1, 1, 0...).
    # If v1 and v2 share a coordinate, dot = 2*1 = 2 or 2*(-1) = -2.
    # If dot=2, dist^2 = 6-4=2. Max norm is 2 (from v1 or v4). dist >= 2 is required.
    # sqrt(2) < 2. So v1 and v2 are NOT compatible if they share a coordinate with same sign.
    # If they share coordinate with opposite sign, dot=-2, dist^2=10. OK.
    # If they don't share, dot=0, dist^2=6. OK.
    # So v1 and v2 are compatible UNLESS they share a coordinate and the signs are the same.
    # This complicates things. Let's stick to v4 and v1 which are mutually compatible.
    # v4 (norm 2) and v1 (norm 2).
    # dot(v4, v1): v4 has 1 at pos i, v1 has 2 at pos i. dot = 2 or -2.
    # dist^2 = 4+4-2*(2) = 4. dist=2. OK.
    # dist^2 = 4+4-2*(-2) = 12. OK.
    # So v4 and v1 are always compatible.
    
    # What about v2 and v4?
    # v2 (norm sqrt(2)), v4 (norm 2).
    # dot(v2, v4) is in {-1, 0, 1}.
    # dist^2 = 2 + 4 - 2*dot = 6 - 2*dot.
    # Min dist^2 = 6 - 2(1) = 4. dist = 2. OK.
    # So v2 and v4 are always compatible.
    
    # So we can include v2!
    # We just need to handle v1 vs v2 compatibility.
    # v1 vs v2:
    # v1 = 2*e_i. v2 = e_j + e_k.
    # If i not in {j,k}, dot=0, dist^2=6. OK.
    # If i in {j,k}, say i=j. v2 = e_i + e_k.
    # dot = 2*1 = 2 (if v2[i]=1) or 2*(-1) = -2 (if v2[i]=-1).
    # If dot=2, dist^2 = 4+2-4=2. dist=sqrt(2) < 2. FAIL.
    # If dot=-2, dist^2 = 4+2+4=10. OK.
    # So v1 and v2 are incompatible if they share an index and the sign of v2 at that index matches the sign of v1.
    
    # This mixed compatibility makes the "always add v1 and v2" strategy invalid.
    # We must select a subset of v1 and v2 that are mutually compatible.
    # However, v4 and v1 are always compatible. v4 and v2 are always compatible.
    # So the core problem is: Max Independent Set on v4 graph.
    # Then, for the selected v4 set, we can add v1 and v2 vectors that don't conflict with each other.
    # v1 vs v1: always compatible (dot 0 or -4, dist^2 4 or 12).
    # v2 vs v2: 
    # v2a = e_i+e_j, v2b = e_k+e_l.
    # If disjoint, dot=0, dist^2=4. OK.
    # If share 1 index, dot=1 or -1. dist^2 = 2+2-2(1)=2. FAIL.
    # So v2 vectors must be disjoint (support sets disjoint) to be compatible?
    # If share 1 index with same sign: dot=1, dist^2=2. FAIL.
    # If share 1 index with opposite sign: dot=-1, dist^2=4. OK.
    # If share 2 indices: dot=2 or -2 or 0.
    # If dot=2, dist^2=0. FAIL.
    # If dot=0 (one sign diff), dist^2=4. OK.
    
    # This is getting complex. Let's stick to the proven strategy:
    # 1. Find Max Independent Set on v4 graph.
    # 2. Add all v1 vectors (22 of them). They are compatible with all v4.
    # 3. Can we add v2 vectors?
    #    A v2 vector is compatible with all selected v4 vectors.
    #    It is compatible with all v1 vectors EXCEPT those that share an index and sign.
    #    It is compatible with other v2 vectors if they don't cause dist < 2.
    
    # Given the complexity, let's just add v1 vectors.
    # Score is k.
    # Current best is ~2.57 * seed.
    # If seed is 1, we are getting ~2.5 vectors? No, "score is k".
    # The prompt says "score is k".
    # "search score is value/seed".
    # If the seed program returns 1 vector, and we return 3, score is 3.
    # Current score 2.5727 implies we are returning ~2.57 vectors?
    # Or maybe the seed returns a specific number.
    # Let's assume the goal is to maximize k.
    
    # Let's generate v1 vectors.
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
    
    # Compute conflict graph for v4 vectors
    # Conflict if dist < 2 => dist^2 < 4
    # dist^2 = 4 + 4 - 2*dot = 8 - 2*dot
    # 8 - 2*dot < 4 => 2*dot > 4 => dot > 2
    # So conflict if dot > 2.
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Spectral Initialization
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    
    L = np.diag(conflicts.sum(axis=1)) - conflicts.astype(np.float64)
    L_sparse = csr_matrix(L)
    
    try:
        eigvals, eigvecs = eigsh(L_sparse, k=3, sigma=0, which='LM')
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        fiedler = eigvecs[:, 1]
    except Exception:
        rng = np.random.default_rng(42)
        fiedler = rng.standard_normal(N4)
        
    # Greedy MIS with Spectral Ordering
    # Order by Fiedler value
    order = np.argsort(fiedler)
    
    selected = np.zeros(N4, dtype=bool)
    alive = np.ones(N4, dtype=bool)
    
    for i in order:
        if not alive[i]:
            continue
        selected[i] = True
        to_remove = np.where(conflicts[i] | (np.arange(N4) == i))[0]
        alive[to_remove] = False
        
    # Local Search: Swap-based
    # We want to maximize the number of selected v4 vectors.
    # Each v4 vector contributes 1 to the count.
    # Each v1 vector contributes 1 to the count (always added).
    # So maximizing v4 count is equivalent to maximizing total count.
    
    rng_ls = np.random.default_rng(77)
    current_selected = selected.copy()
    current_count = current_selected.sum()
    
    # Precompute conflict masks for speed?
    # conflicts is N4xN4 boolean.
    
    improved = True
    iterations = 0
    max_iterations = 5000
    
    while improved and iterations < max_iterations and time.time() < deadline - 10:
        improved = False
        iterations += 1
        
        # Shuffle selected indices to avoid bias
        sel_indices = np.where(current_selected)[0]
        if len(sel_indices) == 0:
            break
            
        rng_ls.shuffle(sel_indices)
        
        # Try swapping out one selected vertex
        for s_out in sel_indices:
            if time.time() > deadline - 10:
                break
                
            # Temporarily remove s_out
            current_selected[s_out] = False
            
            # Find all unselected vertices that do not conflict with current selected set
            # has_conflict[v] = any(conflicts[v, current_selected])
            # We can compute this efficiently
            
            # Vectorized check:
            # conflicts[:, current_selected] is N4 x count_selected
            # any(axis=1) gives N4 boolean
            
            if np.any(current_selected):
                has_conflict = np.any(conflicts[:, current_selected], axis=1)
            else:
                has_conflict = np.zeros(N4, dtype=bool)
                
            # Candidates are those without conflict, not currently selected
            candidates = ~has_conflict & ~current_selected
            
            if np.any(candidates):
                cand_indices = np.where(candidates)[0]
                # Pick one randomly
                u_in = cand_indices[rng_ls.integers(len(cand_indices))]
                
                # Perform swap
                current_selected[u_in] = True
                new_count = current_selected.sum()
                
                # Since we removed 1 and added 1, count is same.
                # But we might have opened up space for more?
                # No, we just swapped.
                # However, the new set might allow further improvements in subsequent iterations.
                # To ensure we don't get stuck in a local minimum of the same size,
                # we accept the swap if it doesn't reduce count (it doesn't).
                # To diversify, we accept with some probability or always?
                # Always accepting a neutral swap can lead to cycling.
                # Let's only accept if we can find a sequence that increases count?
                # That's hard to check.
                # Instead, let's try to ADD a vertex if possible after the swap?
                # No, the swap itself is the move.
                
                # Let's use a different strategy:
                # Try to remove a vertex and see if we can add MORE than 1.
                # Or, try to add a vertex by removing the MINIMUM number of vertices.
                
                # Let's revert and try a different approach:
                # 1. Try to add any free vertex (no conflicts).
                # 2. If none, try to remove 1 selected vertex and add 2 non-selected vertices.
                # 3. If none, try to remove 1 and add 1 (neutral, for diversity).
                
                # Revert swap for now to implement "Add 2 Remove 1"
                current_selected[u_in] = False
                current_selected[s_out] = True
                
                # Check for "Add 2 Remove 1"
                # Remove s_out.
                current_selected[s_out] = False
                
                has_conflict_after_removal = np.any(conflicts[:, current_selected], axis=1) if np.any(current_selected) else np.zeros(N4, dtype=bool)
                candidates_after_removal = ~has_conflict_after_removal & ~current_selected
                
                if np.sum(candidates_after_removal) >= 2:
                    # We can add at least 2.
                    # Pick the best 2? Or just 2?
                    # Let's pick the 2 with highest degree to selected set? No, they are not conflicting.
                    # Just pick any 2.
                    cand_idx = np.where(candidates_after_removal)[0]
                    # Shuffle to randomize
                    rng_ls.shuffle(cand_idx)
                    # Add first 2
                    current_selected[cand_idx[0]] = True
                    current_selected[cand_idx[1]] = True
                    improved = True
                    break # Restart loop with new configuration
                else:
                    # Revert removal
                    current_selected[s_out] = True
                    # Try neutral swap for diversity?
                    # If we have many candidates, a neutral swap might help escape local optima.
                    # But it's risky. Let's stick to improvements only.
                    pass
                    
    # Final Polish: Add any remaining free vertices
    # (This handles cases where the local search didn't fully saturate the set)
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        changed = False
        non_sel = np.where(~current_selected)[0]
        for v in non_sel:
            if np.any(conflicts[v] & current_selected):
                continue
            current_selected[v] = True
            changed = True
        if not changed:
            break

    result_v4 = vectors_v4[current_selected]
    
    # Add all V1 vectors
    # v1 vectors are compatible with all v4 vectors.
    # v1 vectors are compatible with each other.
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    # Verification
    if len(final_vectors) > 1:
        norms = np.sqrt(np.sum(final_vectors.astype(np.float64)**2, axis=1))
        max_norm = np.max(norms)
        
        n_final = len(final_vectors)
        valid = np.ones(n_final, dtype=bool)
        
        # Batch verification
        batch_size = 200
        for i in range(0, n_final, batch_size):
            if time.time() > deadline - 2:
                break
            batch = final_vectors[i:i+batch_size]
            if len(batch) == 0:
                break
            dots_batch = batch @ final_vectors.T
            norms_sq = np.sum(batch.astype(np.float64)**2, axis=1)
            all_norms_sq = np.sum(final_vectors.astype(np.float64)**2, axis=1)
            dist_sq = norms_sq[:, None] + all_norms_sq[None, :] - 2 * dots_batch
            
            for r in range(len(batch)):
                idx_in_final = i + r
                if not valid[idx_in_final]:
                    continue
                d_sq = dist_sq[r, :]
                d_sq[idx_in_final] = np.inf
                # Check against valid vectors
                # If any dist < max_norm, invalidate both
                # dist^2 < max_norm^2
                invalid_mask = (d_sq < max_norm**2 - 1e-6) & valid & (np.arange(n_final) != idx_in_final)
                if np.any(invalid_mask):
                    valid[idx_in_final] = False
                    valid[invalid_mask] = False
                    break
                    
        final_vectors = final_vectors[valid]
        
    return final_vectors

if __name__ == "__main__":
    v = construct()
    print(f"Vectors: {len(v)}")
    norms = np.sqrt(np.sum(v**2, axis=1))
    if len(v) > 1:
        dots = v @ v.T
        dists = np.sqrt(np.maximum(0, norms[:, None]**2 + norms[None, :]**2 - 2*dots))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        max_norm = np.max(norms)
        print(f"Min Dist: {min_dist}, Max Norm: {max_norm}, Valid: {min_dist >= max_norm - 1e-6}")
