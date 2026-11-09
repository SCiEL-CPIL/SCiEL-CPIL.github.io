import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Strategy: Use known high-symmetry lattice structures.
    # The kissing number problem for integer vectors is closely related to root systems.
    # In dimension 11, we can combine lower-dimensional optimal configurations.
    # E8 has kissing number 240. A3 has kissing number 12.
    # E8 x A3 is a lattice in R^11.
    # However, we need integer vectors. E8 is not a subset of Z^8 directly in the standard basis,
    # but we can use the D8 lattice or A8 lattice which are integer.
    # Actually, the problem asks for integer vectors in R^11.
    # Let's try to construct a large set using the A_n lattice structure or D_n.
    
    # A very effective strategy for integer vectors is to use vectors with small norms.
    # If we restrict to vectors with squared norm 2 (i.e., permutations of (±1, ±1, 0, ...)),
    # the distance between any two such vectors v, u is ||v-u||^2 = ||v||^2 + ||u||^2 - 2<v,u> = 4 - 2<v,u>.
    # We need ||v-u||^2 >= max_norm^2.
    # If all vectors have norm^2 = 2, then max_norm^2 = 2.
    # Condition: 4 - 2<v,u> >= 2 => 2 <v,u> <= 2 => <v,u> <= 1.
    # For vectors of type (±1, ±1, 0...), <v,u> can be -2, 0, 2.
    # <v,u> = 2 implies v=u. <v,u> = -2 implies v=-u.
    # So if we pick a set of vectors such that no two are identical and no two are negatives,
    # and we ensure <v,u> is not 2 (impossible for distinct) or -2 (impossible if we avoid negatives),
    # wait. If v and u are distinct, <v,u> is in {-2, 0, 2}? No.
    # Let v = (1, 1, 0..), u = (1, -1, 0..). <v,u> = 1-1=0. Dist^2 = 4. 4 >= 2. OK.
    # Let v = (1, 1, 0..), u = (-1, 1, 0..). <v,u> = -1+1=0. Dist^2 = 4. OK.
    # Let v = (1, 1, 0..), u = (1, 1, 0..). Same vector. Dist^2 = 0. Fail.
    # Let v = (1, 1, 0..), u = (-1, -1, 0..). <v,u> = -2. Dist^2 = 0. Fail.
    
    # So for norm^2=2 vectors, we just need to avoid duplicates and antipodes.
    # How many such vectors?
    # Choose 2 positions out of 11: C(11,2) = 55.
    # Signs: 4 options per pair.
    # Total: 55 * 4 = 220.
    # This matches the baseline score roughly (score 1.27 * 220? No, score is k).
    # Baseline score 1.2727 implies k ~ 220? 
    # Wait, the task says "score is k". 
    # If baseline is 1.2727, and seed is 1.0, then seed k is K_seed.
    # Current k = 1.2727 * K_seed.
    # If K_seed = 220 (from the fallback code in current program: 11*10/2 * 4 = 220),
    # Then current k = 1.2727 * 220 = 280.
    # So we need to find more than 280 vectors.
    
    # Let's try mixing norms.
    # If we include vectors with norm^2 = 4 (e.g., (2,0,..) or (1,1,1,1,0..)),
    # The max norm becomes 4.
    # Condition: ||v-u||^2 >= 4.
    # If v has norm^2=2 and u has norm^2=4:
    # ||v-u||^2 = 2 + 4 - 2<v,u> = 6 - 2<v,u>.
    # We need 6 - 2<v,u> >= 4 => 2 <v,u> <= 2 => <v,u> <= 1.
    # If v has norm^2=2 and u has norm^2=2:
    # ||v-u||^2 = 4 - 2<v,u> >= 4 => <v,u> <= 0.
    # This is stricter than before. Previously <v,u> <= 1 was allowed if max norm was 2.
    # But if max norm is 4, the constraint is tighter.
    
    # Therefore, mixing norms often reduces the total count unless the higher norm vectors
    # are very sparse and orthogonal to the lower norm ones.
    
    # Alternative: Use a lattice with higher kissing number.
    # The Leech lattice is in 24D. 
    # In 11D, the best known kissing number is around 1200-1300 for real vectors.
    # For integer vectors, it is lower.
    
    # Let's try a greedy algorithm but with a much larger and smarter candidate pool.
    # Candidate Pool:
    # 1. All vectors with norm^2 = 2. (220 vectors)
    # 2. All vectors with norm^2 = 4 of type (2,0,..) or (1,1,1,1,..).
    #    If we add these, max_norm^2 becomes 4.
    #    This forces all pairs to have dist^2 >= 4.
    #    For norm^2=2 pairs, this requires <v,u> <= 0.
    #    In the set of 220 vectors, how many pairs have <v,u> > 0?
    #    <v,u> = 1 if they share 1 index with same sign.
    #    <v,u> = -1 if they share 1 index with opposite sign.
    #    <v,u> = 0 if disjoint or share 2 with opposite signs (impossible for norm 2 vectors to share 2 indices and be distinct? 
    #    If v=(1,1,0..), u=(1,-1,0..), they share indices 1,2. <v,u>=0.
    #    If v=(1,1,0..), u=(1,0,1..), they share index 1. <v,u>=1.
    #    
    #    So, if we want to include norm^2=4 vectors, we must prune the norm^2=2 set to keep only
    #    vectors that are "orthogonal" (dot <= 0) to each other? No, dot <= 0 is required for dist^2 >= 4.
    #    This would drastically reduce the number of norm^2=2 vectors we can keep.
    
    # So, sticking to norm^2=2 is likely better for count, UNLESS we can find a different norm.
    
    # What if we use norm^2 = 3? (1,1,1,0..)
    # Max norm^2 = 3.
    # Condition: ||v-u||^2 >= 3.
    # For v,u with norm^2=3:
    # ||v-u||^2 = 6 - 2<v,u> >= 3 => 2<v,u> <= 3 => <v,u> <= 1.5 => <v,u> <= 1.
    # <v,u> for (1,1,1,0..) vectors:
    # - 3 (same)
    # - 1 (share 2 indices same sign)
    # - -1 (share 2 indices opposite sign)
    # - -3 (opposite)
    # - 0 (disjoint)
    # - 1 (share 1 index same sign? No, if they share 1 index, <v,u>=1. If share 2, <v,u>=2 or 0 or -2?
    #   v=(1,1,1,0), u=(1,1,-1,0). <v,u> = 1+1-1 = 1.
    #   v=(1,1,1,0), u=(1,-1,1,0). <v,u> = 1-1+1 = 1.
    #   v=(1,1,1,0), u=(1,1,0,1). <v,u> = 1+1+0 = 2.
    #   So <v,u> can be 2. If <v,u>=2, dist^2 = 6-4=2 < 3. Invalid.
    #   So we must avoid pairs with <v,u>=2.
    
    # How many vectors of norm^2=3?
    # Choose 3 positions: C(11,3) = 165.
    # Signs: 2^3 = 8.
    # Total: 165 * 8 = 1320.
    # This is much larger than 220.
    # We need to select a subset where <v,u> <= 1 for all pairs.
    # This is a maximum independent set problem on a graph where edges connect vectors with <v,u> >= 2.
    # Or simply a greedy packing.
    
    # Let's generate all norm^2=3 vectors and use a greedy algorithm to pick a large subset.
    
    candidates = []
    
    # Generate all vectors with exactly 3 non-zero components, each +/-1
    # Norm squared = 3.
    # Max norm squared = 3.
    # Constraint: ||v-u||^2 >= 3.
    # 6 - 2<v,u> >= 3  => <v,u> <= 1.5 => <v,u> <= 1.
    # So we cannot have <v,u> = 2 or 3.
    # <v,u>=3 means v=u.
    # <v,u>=2 means they share 2 indices with same sign and 1 with opposite?
    # v=(1,1,1,0), u=(1,1,-1,0) -> <v,u>=1.
    # v=(1,1,1,0), u=(1,1,0,1) -> <v,u>=2. (Share indices 1,2 with same sign, 3rd index different pos).
    # So we must avoid pairs that share 2 indices with the same signs.
    
    # Number of such vectors is 1320.
    # Greedy selection:
    # Randomize order, pick v, remove all u with <v,u> >= 2.
    
    # To speed up, we can pre-compute conflicts or use a fast check.
    # 1320 is small enough for O(N^2) checks in Python if optimized?
    # 1320^2 = 1.7M checks. Each check is a dot product of sparse vectors.
    # Should be fast enough in < 1 second.
    
    # Let's generate them.
    indices = range(D)
    for i in indices:
        for j in range(i+1, D):
            for k in range(j+1, D):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        for s3 in [-1, 1]:
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            candidates.append(v)
                            
    # Also add norm^2=2 vectors?
    # If we add norm^2=2 vectors, max norm becomes sqrt(3) ~ 1.73? No, norm^2=3 is max.
    # Wait, if we mix norm^2=2 and norm^2=3, max norm^2 is 3.
    # Constraint for v(norm^2=2), u(norm^2=3):
    # ||v-u||^2 = 2 + 3 - 2<v,u> = 5 - 2<v,u> >= 3 => 2<v,u> <= 2 => <v,u> <= 1.
    # For v(norm^2=2), u(norm^2=2):
    # ||v-u||^2 = 4 - 2<v,u> >= 3 => 2<v,u> <= 1 => <v,u> <= 0.5 => <v,u> <= 0.
    # So norm^2=2 vectors must be orthogonal to each other.
    # This is a very strict constraint for the norm^2=2 set.
    # It's probably better to stick to just norm^2=3 vectors, or just norm^2=2.
    # Norm^2=3 pool is 1320. Even with pruning, we might get > 280.
    
    # Let's try just norm^2=3 vectors first.
    
    # Greedy Algorithm
    # To improve, we can try multiple random orderings.
    
    best_selection = []
    
    # Pre-compute dot products? No, just compute on the fly.
    # 1320 vectors.
    
    # Optimization: Represent vectors as tuples for hashing and fast comparison?
    # Or just use numpy arrays.
    
    # Let's use a list of tuples for the candidate pool to make it easier to shuffle and iterate.
    cand_tuples = [tuple(c) for c in candidates]
    
    # Shuffle
    np.random.shuffle(cand_tuples)
    
    selected = []
    selected_tuples = set()
    
    # For speed, we can keep the selected vectors in a list and check against them.
    # 1320 * len(selected) checks. If len(selected) grows to 500, that's 660k checks.
    # Each check is a dot product.
    
    for v_t in cand_tuples:
        if time.time() > deadline - 10:
            break
            
        v = np.array(v_t, dtype=np.int64)
        
        valid = True
        for u_t in selected:
            u = np.array(u_t, dtype=np.int64)
            dot = np.dot(v, u)
            if dot > 1: # <v,u> must be <= 1
                valid = False
                break
                
        if valid:
            selected.append(v_t)
            
    if len(selected) > len(best_selection):
        best_selection = selected
        
    # Convert back to array
    if best_selection:
        result = np.array(best_selection, dtype=np.int64)
        # Verify validity
        # Check min dist >= max norm
        if len(result) > 0:
            norms_sq = np.sum(result**2, axis=1)
            max_norm_sq = np.max(norms_sq)
            
            # Check pairwise distances
            # This is O(k^2 * D). k ~ 500, D=11. 500^2 * 11 = 2.7M ops. Fast.
            valid = True
            for i in range(len(result)):
                for j in range(i+1, len(result)):
                    diff = result[i] - result[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < max_norm_sq:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return result
            else:
                # If invalid, fall back to greedy with stricter check or smaller set
                pass

    # Fallback: Use the norm^2=2 set which is simpler
    # Just return the 220 vectors of norm^2=2, ensuring no antipodes.
    # Actually, the previous logic for norm^2=2:
    # Max norm^2 = 2.
    # Dist^2 >= 2.
    # 4 - 2<v,u> >= 2 => <v,u> <= 1.
    # For distinct v,u, <v,u> is in {-2, 0, 2}? No.
    # v=(1,1,0), u=(1,1,0) -> <v,u>=2. Dist=0.
    # v=(1,1,0), u=(1,-1,0) -> <v,u>=0. Dist=2.
    # v=(1,1,0), u=(-1,1,0) -> <v,u>=0. Dist=2.
    # v=(1,1,0), u=(1,0,1) -> <v,u>=1. Dist=2.
    # So <v,u> <= 1 is always true for distinct vectors of norm^2=2, EXCEPT when they are the same?
    # No, if they are the same, <v,u>=2.
    # So we just need to ensure no duplicates.
    # And we should avoid antipodes?
    # v=(1,1,0), u=(-1,-1,0). <v,u> = -2. Dist^2 = 4 - 2(-2) = 8 >= 2. OK.
    # So antipodes are fine.
    # So we can take ALL 220 vectors.
    
    fallback_vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    fallback_vectors.append(v)
                    
    return np.array(fallback_vectors, dtype=np.int64)
