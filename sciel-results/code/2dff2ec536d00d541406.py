"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import time

D = 11
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # Generate candidate pool: all integer vectors in R^11 with norm^2 <= 4
    # (norm <= 2), which includes weight-2 (2*e_i) and weight-4 (e_i+e_j, e_i-e_j) type vectors
    # Actually, let's be more careful. We want vectors where min pairwise distance >= max norm.
    # If all vectors have norm 2, then we need pairwise distance >= 2, i.e., dot product <= 0.
    
    # Strategy: 
    # 1. Generate all integer vectors with norm^2 = 4 (these have norm 2)
    #    - Type A: two non-zero entries, each +/-1 (weight-2 in terms of non-zeros, but norm^2=2... wait)
    #    Actually norm^2 = sum of squares. For integer vectors:
    #    - 4*e_i: norm^2 = 16, too big
    #    - 2*e_i: norm^2 = 4, norm = 2
    #    - e_i + e_j: norm^2 = 2, norm = sqrt(2)
    #    - e_i - e_j: norm^2 = 2
    #    
    # Let me reconsider. The condition is: min pairwise distance >= max norm.
    # If we use vectors of different norms, the constraint is harder.
    # Best to use uniform norm vectors.
    #
    # Vectors with norm^2 = 4:
    # - (±2, 0, ..., 0): 22 vectors
    # - (±1, ±1, 0, ..., 0): C(11,2) * 4 = 55 * 4 = 220 vectors
    #
    # Total: 242 candidates with norm 2.
    #
    # Two vectors u, v with norm 2 have distance^2 = 4 + 4 - 2*u·v = 8 - 2*u·v
    # We need distance >= 2, so distance^2 >= 4, so 8 - 2*u·v >= 4, so u·v <= 2.
    #
    # For (±2,0,...,0) and (±1,±1,0,...,0): dot product can be 0, ±2, ±4
    #   - If they share a coordinate: dot = ±2*1 = ±2 or ±2*(-1) = ∓2. So |dot| = 2.
    #   - If they don't share: dot = 0.
    #   - dot = 2 is allowed (distance^2 = 8-4 = 4, distance = 2 >= 2). OK.
    #   - dot = -2: distance^2 = 12, fine.
    #   - dot = 4: would need 2*2=4, but that requires same sign on same coord, giving dot=2 not 4.
    #     Wait: (2,0,...) · (1,1,0,...) = 2*1 = 2. (2,0,...)·(-1,1,...) = -2. Max |dot|=2. OK.
    #
    # For (±1,±1,0,...) and (±1,±1,0,...):
    #   - Same two coords, same signs: dot = 1+1 = 2. OK (distance=2).
    #   - Same two coords, one sign differs: dot = 1-1 = 0. OK.
    #   - Same two coords, both signs differ: dot = -1-1 = -2. OK.
    #   - Share one coord: dot = ±1. OK.
    #   - Disjoint: dot = 0. OK.
    #   - So ANY pair of weight-2 norm-2 vectors has |dot| <= 2. All compatible!
    #
    # Wait, that means all 242 vectors are mutually compatible? Let me double-check.
    # (2,0,...,0) and (1,1,0,...,0): dot = 2. distance^2 = 8 - 2*2 = 4. distance = 2. >= 2. OK.
    # (2,0,...,0) and (-1,1,0,...,0): dot = -2. distance^2 = 8+4=12. OK.
    # (1,1,0,...) and (1,-1,0,...): dot = 0. OK.
    # (1,1,0,...) and (-1,-1,0,...): dot = -2. OK.
    # (1,1,0,...) and (1,1,0,...) same vector - not applicable since we pick distinct.
    # 
    # So the maximum kissing number using norm-2 integer vectors is at most 242?
    # But wait, we need to check if ALL 242 can coexist. Since all pairs have dot <= 2,
    # yes, all 242 are mutually compatible!
    
    # But wait - the current seed only gets 252 from E8⊕A3 construction which uses non-integer
    # vectors (half-integers). Let me re-examine.
    #
    # The seed uses E8 half-integer vectors scaled by 2, giving integer vectors in R^8 with
    # entries ±1, padded to R^11. These have norm^2 = 8, norm = sqrt(8) ≈ 2.83.
    # And E8 integer vectors with entries ±2, norm^2 = 8.
    # And A3 vectors with entries ±2, norm^2 = 8.
    # So all have norm sqrt(8), and pairwise distance >= sqrt(8).
    #
    # So the seed uses vectors of norm sqrt(8), not norm 2. The score is just k (number of vectors).
    # 
    # Can we do better than 252? The known kissing number in dimension 11 is not known exactly,
    # but lower bounds are around 504 or so. Let me think about what constructions exist.
    #
    # Actually, let me try a different approach: use vectors of norm^2 = 4 (norm 2) since
    # we showed all 242 are compatible. But 242 < 252, so that's worse.
    #
    # Let me try norm^2 = 8 vectors (like the seed). Integer vectors in R^11 with norm^2 = 8:
    # - Four entries of ±1: C(11,4) * 2^4 = 330 * 16 = 5280 vectors
    # - One entry ±2, two entries ±1: C(11,1)*C(10,2)*2^3 = 11*45*8 = 3960 vectors  
    # - Two entries ±2: C(11,2)*2^2 = 55*4 = 220 vectors
    # Total: 5280 + 3960 + 220 = 9460 candidates with norm^2 = 8.
    #
    # Two vectors u,v with norm^2=8 have distance^2 = 16 - 2*u·v.
    # Need distance >= sqrt(8), so distance^2 >= 8, so 16 - 2*u·v >= 8, so u·v <= 4.
    #
    # This is a much larger pool. Let's use MIS to find the largest compatible subset.
    
    # Generate all integer vectors in R^11 with norm^2 = 8
    candidates = []
    candidate_set = set()
    
    # Type 1: Four ±1 entries, rest 0
    for positions in itertools.combinations(range(11), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = [0] * 11
            for pos, s in zip(positions, signs):
                v[pos] = s
            key = tuple(v)
            if key not in candidate_set:
                candidate_set.add(key)
                candidates.append(v)
    
    if time.time() > deadline:
        return np.array(candidates, dtype=np.int64) if candidates else np.zeros((0, 11), dtype=np.int64)
    
    # Type 2: One ±2, two ±1 entries
    for pos2 in range(11):
        for s2 in [-2, 2]:
            for pos1, pos3 in itertools.combinations([p for p in range(11) if p != pos2], 2):
                for s1 in [-1, 1]:
                    for s3 in [-1, 1]:
                        v = [0] * 11
                        v[pos2] = s2
                        v[pos1] = s1
                        v[pos3] = s3
                        key = tuple(v)
                        if key not in candidate_set:
                            candidate_set.add(key)
                            candidates.append(v)
    
    if time.time() > deadline:
        return np.array(candidates, dtype=np.int64) if candidates else np.zeros((0, 11), dtype=np.int64)
    
    # Type 3: Two ±2 entries
    for pos1, pos2 in itertools.combinations(range(11), 2):
        for s1 in [-2, 2]:
            for s2 in [-2, 2]:
                v = [0] * 11
                v[pos1] = s1
                v[pos2] = s2
                key = tuple(v)
                if key not in candidate_set:
                    candidate_set.add(key)
                    candidates.append(v)
    
    if time.time() > deadline:
        return np.array(candidates, dtype=np.int64) if candidates else np.zeros((0, 11), dtype=np.int64)
    
    n = len(candidates)
    # Convert to numpy for fast dot products
    cand_arr = np.array(candidates, dtype=np.int64)
    
    # Build conflict graph: edge between i,j if dot(i,j) > 4
    # dot product matrix
    # This is 9460 x 9460 which is ~89M entries. Might be too large.
    # Let's check: 9460^2 * 8 bytes = ~715 MB. Too much.
    
    # Alternative: use a different norm. Let me try norm^2 = 4 first since it's smaller.
    # Actually, let me reconsider the problem.
    
    # The seed gets 252. Let me try to improve by adding more vectors.
    # 
    # Key insight: if we use vectors of different norms, we need min_dist >= max_norm.
    # If we mix norm-2 and norm-sqrt(8) vectors, max_norm = sqrt(8) ≈ 2.83.
    # A norm-2 vector and norm-sqrt(8) vector: dist^2 = 4 + 8 - 2*u·v = 12 - 2*u·v.
    # Need dist >= sqrt(8), so dist^2 >= 8, so 12 - 2*u·v >= 8, u·v <= 2.
    #
    # This gets complicated. Let me stick with uniform norm.
    
    # For norm^2 = 8, the pool is 9460. Let me try a greedy approach first,
    # then see if MIS can improve it.
    
    # Greedy: sort by degree (number of conflicts), pick lowest-degree first
    # But building the full conflict matrix is expensive.
    
    # Alternative: use a smarter generation. 
    # Actually, let me try a completely different approach.
    # 
    # Known construction: 2*E8 x 2*D3 gives 252 vectors.
    # Can we find a better lattice or code?
    #
    # Let me try: use all vectors with norm^2 = 4 (242 vectors, all compatible)
    # PLUS some vectors with norm^2 = 8 that are compatible with all of them.
    #
    # A norm^2=8 vector v is compatible with all norm^2=4 vectors if v·w <= 2 for all norm^2=4 vectors w.
    # Norm^2=4 vectors are: 2e_i (22 of them) and e_i±e_j (220 of them).
    # v·2e_i = 2*v_i, need |2*v_i| <= 2, so |v_i| <= 1.
    # v·(e_i+e_j) = v_i + v_j, need |v_i + v_j| <= 2 for all i≠j.
    # v·(e_i-e_j) = v_i - v_j, need |v_i - v_j| <= 2 for all i≠j.
    #
    # For integer v with norm^2=8 and |v_i| <= 1 for all i:
    # v has entries in {-1, 0, 1}, norm^2 = 8, so exactly 8 non-zero entries, each ±1.
    # |v_i + v_j| <= 2: since v_i, v_j ∈ {-1,0,1}, max |v_i+v_j| = 2. Always OK.
    # |v_i - v_j| <= 2: max |v_i - v_j| = 2. Always OK.
    #
    # So ANY integer vector with norm^2=8 and entries in {-1,0,1} is compatible with all norm^2=4 vectors!
    # Number of such vectors: C(11,8) * 2^8 = 165 * 256 = 42240. That's huge!
    #
    # But we need pairwise compatibility among these too.
    # Two such vectors u, v with norm^2=8: u·v <= 4.
    # u·v = sum of products, each term in {-1,0,1}. Max u·v = 8 (if u=v).
    # For distinct u,v, max u·v could be 6 or 7.
    #
    # So we need to find a large set of weight-8 ±1 vectors in R^11 with pairwise dot <= 4.
    #
    # This is equivalent to finding a large code with minimum distance.
    # u·v <= 4 means dist^2 = 16 - 2*u·v >= 16 - 8 = 8. So min distance >= sqrt(8).
    #
    # Let me use MIS on this subproblem. The pool of weight-8 ±1 vectors is 42240.
    # Still large. Let me try a greedy approach.
    
    # Actually, let me just do a greedy selection from the norm^2=4 pool plus compatible norm^2=8 vectors.
    
    # Start with all 242 norm^2=4 vectors
    base_vectors = []
    # 2e_i vectors
    for i in range(11):
        base_vectors.append((2, i))  # 2 in position i
        base_vectors.append((-2, i))
    # e_i ± e_j vectors
    for i, j in itertools.combinations(range(11), 2):
        base_vectors.append((1, i, j))    # e_i + e_j
        base_vectors.append((1, i, -j))   # e_i - e_j
        base_vectors.append((-1, i, j))   # -e_i + e_j
        base_vectors.append((-1, i, -j))  # -e_i - e_j
    
    # Convert to actual vectors
    base_vecs = []
    for spec in base_vectors:
        v = [0] * 11
        if len(spec) == 2:
            v[spec[1]] = spec[0]
        elif len(spec) == 3:
            v[spec[1]] = spec[0]
            v[spec[2]] = spec[2]
        base_vecs.append(v)
    
    base_arr = np.array(base_vecs, dtype=np.int64)
    n_base = len(base_vecs)
    
    # Now find norm^2=8 vectors compatible with all base vectors.
    # As shown, all weight-8 ±1 vectors work. Let's generate them and use greedy MIS.
    
    weight8_vectors = []
    for positions in itertools.combinations(range(11), 8):
        for signs in itertools.product([-1, 1], repeat=8):
            v = [0] * 11
            for pos, s in zip(positions, signs):
                v[pos] = s
            weight8_vectors.append(v)
    
    if time.time() > deadline:
        return np.array(base_vecs, dtype=np.int64)
    
    w8_arr = np.array(weight8_vectors, dtype=np.int64)
    n_w8 = len(w8_arr)
    
    # Greedy MIS on weight-8 vectors: sort by some heuristic, pick compatible ones
    # Conflict: u·v > 4
    # Use greedy: repeatedly pick vector with fewest conflicts among remaining
    
    # For efficiency, let's use a simpler greedy: random order, pick if compatible with all selected
    # But that might not be optimal. Let me try degree-based greedy.
    
    # Actually, with 42240 vectors, even O(n^2) is ~1.8B operations. Too slow in Python.
    # Let me use numpy for batch dot products.
    
    # Greedy with numpy: 
    # 1. Compute all pairwise dot products (42240 x 42240 = ~1.8B entries, 14GB). Too much.
    
    # Alternative: process in chunks or use a different strategy.
    
    # Let me try: randomly sample a subset and do greedy on that.
    # Or use a smarter approach.
    
    # Actually, let me try a completely different strategy.
    # Use the known 252-vector construction and try to add more vectors.
    
    # Reconstruct the seed's 252 vectors
    seed_vectors = []
    
    # Part 1: E8 half-integer vectors scaled by 2 (128 vectors)
    for signs in itertools.product([-1, 1], repeat=8):
        if signs.count(1) % 2 == 0:
            v = list(signs) + [0, 0, 0]
            seed_vectors.append(v)
    
    # Part 2: E8 integer vectors scaled by 2 (112 vectors)
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 11
                v[i] = 2 * si
                v[j] = 2 * sj
                seed_vectors.append(v)
    
    # Part 3: A3 vectors (12 vectors)
    for i, j in itertools.combinations(range(8, 11), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 11
                v[i] = 2 * si
                v[j] = 2 * sj
                seed_vectors.append(v)
    
    seed_arr = np.array(seed_vectors, dtype=np.int64)
    n_seed = len(seed_arr)
    
    # All seed vectors have norm^2 = 8.
    # Try to add more vectors with norm^2 = 8 that are compatible with all seed vectors.
    # v compatible with all seed vectors: v·s <= 4 for all s in seed.
    
    # Generate candidate vectors with norm^2 = 8 not already in seed
    all_norm8 = []
    # Type: four ±1
    for positions in itertools.combinations(range(11), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = [0] * 11
            for pos, s in zip(positions, signs):
                v[pos] = s
            all_norm8.append(tuple(v))
    
    # Type: one ±2, two ±1
    for pos2 in range(11):
        for s2 in [-2, 2]:
            for pos1, pos3 in itertools.combinations([p for p in range(11) if p != pos2], 2):
                for s1 in [-1, 1]:
                    for s3 in [-1, 1]:
                        v = [0] * 11
                        v[pos2] = s2
                        v[pos1] = s1
                        v[pos3] = s3
                        all_norm8.append(tuple(v))
    
    # Type: two ±2
    for pos1, pos2 in itertools.combinations(range(11), 2):
        for s1 in [-2, 2]:
            for s2 in [-2, 2]:
                v = [0] * 11
                v[pos1] = s1
                v[pos2] = s2
                all_norm8.append(tuple(v))
    
    if time.time() > deadline:
        return seed_arr
    
    # Remove duplicates and vectors already in seed
    seed_set = set(tuple(v) for v in seed_vectors)
    unique_norm8 = list(set(all_norm8))
    candidates_to_add = [v for v in unique_norm8 if v not in seed_set]
    
    cand_arr = np.array(candidates_to_add, dtype=np.int64)
    n_cand = len(cand_arr)
    
    # Check which candidates are compatible with all seed vectors
    # dot(cand, seed) <= 4 for all seed
    # Compute dot products: n_cand x n_seed
    dots = cand_arr @ seed_arr.T
    compatible_mask = np.all(dots <= 4, axis=1)
    compatible_cands = cand_arr[compatible_mask]
    n_compat = len(compatible_cands)
    
    if time.time() > deadline:
        return seed_arr
    
    # Now find MIS among compatible_cands, where conflict is dot > 4
    # Greedy approach: repeatedly pick vector with minimum degree in remaining conflict graph
    
    if n_compat == 0:
        return seed_arr
    
    # For large n_compat, use a simpler greedy
    # Sort by number of conflicts (approximate by random sampling)
    # Or just do random greedy with multiple restarts
    
    best_selection = []
    best_size = 0
    
    # Try greedy with different orderings
    for trial in range(5):
        if time.time() > deadline:
            break
        
        if trial == 0:
            order = np.arange(n_compat)
        else:
            order = np.random.permutation(n_compat)
        
        selected = []
        remaining = set(order.tolist())
        
        # Greedy: pick vector, remove all conflicting vectors
        while remaining and time.time() < deadline:
            # Pick first remaining vector (or one with fewest conflicts)
            # For speed, just pick the first one
            idx = min(remaining)
            selected.append(idx)
            remaining.discard(idx)
            
            # Remove all vectors that conflict with idx
            v_idx = compatible_cands[idx]
            dots_with_rest = compatible_cands[list(remaining)] @ v_idx
            conflict_mask = dots_with_rest > 4
            for i, r in enumerate(list(remaining)):
                if conflict_mask[i]:
                    remaining.discard(r)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selection = selected
    
    # Combine seed vectors with best selection
    final_vectors = seed_vectors + [compatible_cands[i] for i in best_selection]
    final_arr = np.array(final_vectors, dtype=np.int64)
    
    # Verify the result
    # Check all vectors have norm^2 = 8
    norms = np.sum(final_arr ** 2, axis=1)
    if not np.all(norms == 8):
        # Fallback to seed
        return seed_arr
    
    # Check pairwise distances
    # dist^2(i,j) = 16 - 2*dot(i,j) >= 8, i.e., dot(i,j) <= 4
    # For large arrays, this is expensive. Spot check.
    if len(final_arr) > 100:
        # Sample some pairs
        np.random.seed(42)
        for _ in range(1000):
            i, j = np.random.randint(0, len(final_arr), 2)
            if i != j:
                d = np.dot(final_arr[i], final_arr[j])
                if d > 4:
                    # Invalid, fall back
                    return seed_arr
    
    return final_arr

# EVOLVE-BLOCK-END
