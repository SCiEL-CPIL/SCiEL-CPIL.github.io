import numpy as np
import time
import itertools

D = 11


def construct():
    start_time = time.time()
    deadline = start_time + 105.0  # 15s margin before 120s limit

    # Generate all weight-2 vectors: v has exactly two non-zero entries, each in {1, -1}
    # There are C(11,2) * 4 = 55 * 4 = 220 such vectors
    all_vecs = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    all_vecs.append(v)
    all_vecs = np.array(all_vecs, dtype=np.int64)
    n_total = len(all_vecs)  # 220

    # Precompute the conflict graph: two vectors conflict if their distance < max_norm
    # max_norm = 2 (norm of weight-2 vector with entries +/-1: sqrt(1+1)=sqrt(2)≈1.414)
    # Wait, the task says "normalizing each vector to norm 2 certifies a kissing configuration"
    # So we normalize to norm 2. The norm of a weight-2 vector is sqrt(2).
    # After normalizing to norm 2, the inner product between two normalized vectors u,v is:
    # (u·v)/(|u||v|) * 2... wait, let me think again.
    # 
    # Actually, the condition is: minimum pairwise distance >= maximum norm.
    # For weight-2 vectors, all have norm sqrt(2). So max_norm = sqrt(2).
    # Distance between u and v: |u - v| >= sqrt(2)
    # |u-v|^2 >= 2
    # |u|^2 + |v|^2 - 2 u·v >= 2
    # 2 + 2 - 2 u·v >= 2
    # 4 - 2 u·v >= 2
    # u·v <= 1
    #
    # For weight-2 vectors, u·v can be: -2, -1, 0, 1, 2
    # u·v = 2 means same vector (conflict: not allowed, but we won't have duplicates)
    # u·v = 1 means they share exactly one coordinate with same sign
    # u·v = 0 means disjoint supports or... let's check: if supports overlap in one coord,
    #   u·v = si*si' where si, si' are the values. So u·v = ±1.
    # If supports are disjoint, u·v = 0.
    # If supports are the same (i=j pair), u·v = si*si' + sj*sj' which is ±2 or 0.
    #
    # So conflict condition: u·v > 1, i.e., u·v = 2 (same vector)
    # Wait, that means ALL weight-2 vectors are mutually non-conflicting except identical ones!
    # That can't be right because then we'd just return all 220.
    #
    # Let me re-read: "minimum pairwise distance is at least the maximum vector norm"
    # All weight-2 vectors have norm sqrt(2). Max norm = sqrt(2).
    # Min pairwise distance must be >= sqrt(2).
    # For two distinct weight-2 vectors u, v:
    # |u-v|^2 = |u|^2 + |v|^2 - 2u·v = 4 - 2u·v
    # We need |u-v|^2 >= 2, i.e., 4 - 2u·v >= 2, i.e., u·v <= 1.
    #
    # u·v = 2 only when u = v (identical). For distinct vectors, max u·v = 1.
    # So indeed all 220 weight-2 vectors satisfy the condition!
    #
    # But wait - the seed program returns these 220 vectors and scores 1.0.
    # The task says score is k, and search score is value/seed. Seed returns 220.
    # So seed value = 220. To improve, we need k > 220.
    #
    # We can mix in other vectors. Let's consider weight-1 vectors (unit vectors).
    # Norm = 1. If we include a weight-1 vector, max norm might change.
    # Actually, we should think about what mix of vectors gives the best packing.
    #
    # Key insight: we want to maximize the number of vectors such that for all pairs,
    # |u - v| >= max(|u|, |v|, ..., |all vectors|)
    # i.e., |u - v| >= R where R = max_i |v_i|
    #
    # If we use vectors of different norms, R is set by the largest norm vector.
    # 
    # Strategy: use vectors with norms in a small set, and ensure pairwise distances >= R.
    #
    # Let's try: include weight-2 vectors (norm sqrt(2)) and weight-3 vectors (norm sqrt(3)).
    # R = sqrt(3). Need |u-v| >= sqrt(3) for all pairs.
    # |u-v|^2 >= 3.
    # 
    # For two weight-3 vectors: |u-v|^2 = 6 - 2u·v >= 3 => u·v <= 1.5, so u·v <= 1.
    # For weight-2 and weight-3: |u-v|^2 = 2 + 3 - 2u·v = 5 - 2u·v >= 3 => u·v <= 1.
    # For two weight-2: |u-v|^2 = 4 - 2u·v >= 3 => u·v <= 0.5, so u·v <= 0.
    #
    # This is more restrictive. Let's think about what's optimal.
    #
    # Alternative: just use weight-2 vectors but also add their negatives? 
    # Wait, the seed already includes all sign combinations. The 220 vectors include both
    # v and -v for each pair. Let me verify: for pair (i,j), we have (1,1), (1,-1), (-1,1), (-1,-1).
    # Yes, all 4 sign combos. So we already have 220.
    #
    # Can we add weight-1 vectors? A weight-1 vector e_i has norm 1.
    # If max norm is sqrt(2) (from weight-2 vectors), we need |e_i - v| >= sqrt(2) for all weight-2 v.
    # |e_i - v|^2 = 1 + 2 - 2 e_i·v = 3 - 2e_i·v >= 2 => e_i·v <= 0.5 => e_i·v <= 0.
    # e_i·v is the i-th component of v. For v to have v_i = 0, v must not use coordinate i.
    # So e_i conflicts with any weight-2 vector that uses coordinate i with non-zero value.
    # e_i·v = v_i which is 0, 1, or -1. We need v_i = 0.
    # So e_i is compatible with weight-2 vectors that don't touch coordinate i.
    # Number of weight-2 vectors not touching coord i: C(10,2)*4 = 180.
    # But we need e_i to be compatible with ALL selected weight-2 vectors.
    # If we include all 220 weight-2 vectors, no weight-1 vector works.
    #
    # Trade-off: remove some weight-2 vectors to add weight-1 vectors.
    # Removing all weight-2 vectors touching coord i: 2*(10) = 20 vectors (10 pairs * 2 sign combos... 
    # wait, pairs involving i: (i,j) for j≠i, that's 10 pairs, each with 4 sign combos = 40 vectors.
    # Remove 40 to add 2 (e_i and -e_i). Net: -38. Bad.
    #
    # What about a different approach? Use vectors with larger norms but fewer of them,
    # mixed with many small-norm vectors?
    #
    # Actually, let me reconsider. The kissing number in dimension 11 is known to be 504.
    # The seed gives 220. We need a better construction.
    #
    # Let me try a greedy approach with a larger candidate pool.
    # Include weight-1, weight-2, and weight-3 vectors, and greedily select.
    
    # Build candidate pool
    candidates = []
    
    # Weight 1: C(11,1) * 2 = 22
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    
    # Weight 2: C(11,2) * 4 = 220
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Weight 3: C(11,3) * 8 = 165 * 8 = 1320
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates ** 2, axis=1).astype(np.int64)
    
    # Greedy: sort candidates by... we want to maximize count.
    # Start with all weight-2 (known good set of 220), then try to add others.
    # Actually, let's do a proper greedy:
    # 1. Sort candidates (e.g., by norm or by some heuristic)
    # 2. Greedily add each candidate if it's compatible with all already-added
    
    # For efficiency, let's use a conflict matrix approach with the selected set.
    
    def is_compatible(u, selected):
        """Check if u is compatible with all vectors in selected."""
        if len(selected) == 0:
            return True
        u_norm_sq = np.dot(u, u)
        # Max norm squared in selected
        max_ns = max(np.dot(v, v) for v in selected)
        R_sq = max(u_norm_sq, max_ns)
        # Check all pairs
        for v in selected:
            diff = u - v
            dist_sq = np.dot(diff, diff)
            if dist_sq < R_sq:
                return False
        return True
    
    # This is O(n_selected) per candidate, which could be slow.
    # Let's use a smarter approach: precompute pairwise distances.
    
    # Actually, with ~1562 candidates, the full distance matrix is 1562^2 ≈ 2.4M entries.
    # That's manageable.
    
    if time.time() > deadline:
        # Fallback: return the 220 weight-2 vectors
        return all_vecs
    
    # Compute pairwise distance matrix and norm matrix
    # dist_sq[i,j] = |cand_i - cand_j|^2
    # This is O(n^2 * D) = 1562^2 * 11 ≈ 26.6M ops. Should be fast with numpy.
    
    # dist_sq[i,j] = norms_sq[i] + norms_sq[j] - 2 * dot(cand_i, cand_j)
    dots = candidates @ candidates.T  # (n, n) matrix of inner products
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # For a set S, the constraint is: for all u,v in S, dist_sq[u,v] >= max(norms_sq[u], norms_sq[v])
    # Equivalently: dist_sq[u,v] >= max_norm_sq(S) where max_norm_sq(S) = max_{w in S} norms_sq[w]
    # Since we need dist >= R where R = max norm, dist_sq >= R^2 = max_norm_sq.
    
    # Greedy algorithm with conflict checking:
    # Sort candidates by some priority. Let's try: prefer weight-2 first (known good), then others.
    
    # Index order: 0-21 are weight-1, 22-241 are weight-2, 242-1561 are weight-3
    # Let's create a priority order
    
    # Strategy: start with all weight-2 vectors (indices 22 to 241), then try to add others.
    
    selected_indices = list(range(22, 242))  # All weight-2 vectors
    selected_set = set(selected_indices)
    max_ns_selected = norms_sq[22:242].max()  # Should be 2
    
    # Try to add weight-1 and weight-3 vectors
    other_indices = list(range(0, 22)) + list(range(242, n_cand))
    
    # For each candidate, check if it's compatible with all selected
    # dist_sq[cand, selected] >= max(norms_sq[cand], max_ns_selected)
    
    # To make this efficient, precompute the conflict with current selected set.
    # For candidate c: min over s in selected of dist_sq[c,s] >= max(norms_sq[c], max_ns_selected)
    
    # Let's just iterate and add compatible ones
    added = 0
    for c in other_indices:
        if time.time() > deadline:
            break
        R_sq = max(norms_sq[c], max_ns_selected)
        # Check against all selected
        # dist_sq[c, selected_indices] >= R_sq
        sel_arr = np.array(selected_indices, dtype=np.int64)
        if len(sel_arr) == 0:
            selected_indices.append(c)
            selected_set.add(c)
            added += 1
            if norms_sq[c] > max_ns_selected:
                max_ns_selected = norms_sq[c]
            continue
        
        # Vectorized check
        min_dist = np.min(dist_sq[c, sel_arr])
        if min_dist >= R_sq:
            selected_indices.append(c)
            selected_set.add(c)
            added += 1
            if norms_sq[c] > max_ns_selected:
                max_ns_selected = norms_sq[c]
    
    # Now try local search: for each pair of selected vectors, see if we can swap
    # Actually, let's try a different structural approach.
    # 
    # The issue is that mixing norms creates tighter constraints.
    # Let's try a pure weight-2 approach but with a better selection.
    # Actually, all 220 weight-2 vectors are mutually compatible, so we get 220.
    #
    # Let's try a completely different approach: use the structure of the problem.
    # 
    # Key insight: if we scale all vectors to have the same norm R, then we need
    # |u-v| >= R, which means u·v <= R^2/2 (since |u-v|^2 = 2R^2 - 2u·v >= R^2).
    # So we need u·v <= R^2/2.
    #
    # For integer vectors, if we use vectors of norm^2 = 2 (weight-2), R^2 = 2,
    # need u·v <= 1. Since distinct weight-2 vectors have u·v <= 1, all work.
    #
    # For norm^2 = 4 (e.g., (2,0,0,...) or (1,1,1,0,...) has norm^2=3, not 4),
    # (2,0,...) has norm^2=4. R^2=4, need u·v <= 2.
    # Two vectors (2,0,...) and (0,2,...): u·v=0, ok.
    # (2,0,...) and (1,1,0,...): u·v=2, ok (2<=2).
    # But (1,1,0,...) has norm^2=2, not 4. Mixed norms.
    #
    # Let me try a different strategy: greedy with a larger pool and better ordering.
    
    # Restart with a fresh greedy that considers all candidates
    # Sort by norm (prefer lower norms to keep R small)
    norm_order = np.argsort(norms_sq, kind='stable')
    
    selected = []
    selected_set = set()
    max_ns = 0
    
    for idx in norm_order:
        if time.time() > deadline:
            break
        R_sq = max(norms_sq[idx], max_ns)
        if len(selected) == 0:
            selected.append(idx)
            selected_set.add(idx)
            max_ns = norms_sq[idx]
            continue
        sel_arr = np.array(selected, dtype=np.int64)
        min_dist = np.min(dist_sq[idx, sel_arr])
        if min_dist >= R_sq:
            selected.append(idx)
            selected_set.add(idx)
            if norms_sq[idx] > max_ns:
                max_ns = norms_sq[idx]
    
    k_greedy = len(selected)
    
    # Compare with the weight-2 only set
    k_weight2 = 220
    
    if k_greedy > k_weight2:
        result_indices = selected
    else:
        result_indices = list(range(22, 242))
    
    # Try to improve with local search: try swapping one selected vector for one unselected
    if time.time() < deadline - 30:
        unselected = [i for i in range(n_cand) if i not in selected_set]
        # For each selected vector, try replacing it with an unselected one
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            sel_arr = np.array(result_indices, dtype=np.int64)
            max_ns_cur = norms_sq[sel_arr].max()
            
            for i, s in enumerate(result_indices):
                if time.time() > deadline - 15:
                    break
                # Try removing s and adding each unselected candidate
                remaining = [x for j, x in enumerate(result_indices) if j != i]
                rem_arr = np.array(remaining, dtype=np.int64) if remaining else np.array([], dtype=np.int64)
                max_ns_rem = norms_sq[rem_arr].max() if len(rem_arr) > 0 else 0
                
                for u in unselected[:50]:  # Limit to first 50 unselected for speed
                    R_sq = max(norms_sq[u], max_ns_rem)
                    if len(rem_arr) == 0:
                        result_indices[i] = u
                        selected_set.discard(s)
                        selected_set.add(u)
                        improved = True
                        break
                    min_d = np.min(dist_sq[u, rem_arr])
                    if min_d >= R_sq:
                        result_indices[i] = u
                        selected_set.discard(s)
                        selected_set.add(u)
                        improved = True
                        break
    
    result = candidates[np.array(result_indices, dtype=np.int64)]
    return result
