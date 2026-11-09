# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates 
    such that their maximum norm is smaller than their minimum pairwise distance.
    
    Strategy:
    1. Initialize with a small valid set.
    2. Generate candidate integer vectors in order of increasing norm squared.
    3. Greedily add a candidate if it satisfies the constraint:
       min_pairwise_dist_sq >= max_norm_sq
       where min_pairwise_dist_sq is the minimum squared distance between the new point and all existing points.
    4. Continue until time limit is reached.
    
    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    start_time = time.time()
    deadline = start_time + 625  # 5s margin
    
    d = 11
    points_list = []
    
    # Helper to check if a point can be added
    def can_add(p, current_points, max_norm_sq):
        if len(current_points) == 0:
            return True
        # Calculate squared distances to all existing points
        # We need min_dist_sq >= max_norm_sq
        for q in current_points:
            diff = p - q
            dist_sq = np.dot(diff, diff)
            if dist_sq < max_norm_sq:
                return False
        return True

    # Helper to get max norm squared of current points
    def get_max_norm_sq(current_points):
        if not current_points:
            return 0
        return max(np.dot(p, p) for p in current_points)

    # Generate candidate points
    # We search for points with small norms first.
    # Norm squared = sum of squares of coordinates.
    # We iterate through possible norm squared values N = 1, 2, 3, ...
    # For each N, generate all integer vectors with sum of squares = N.
    
    # To be efficient, we can pre-generate candidates up to a certain norm.
    # Given the 600s limit, we can try to go up to a reasonable norm.
    # The current seed uses norm 2 (vectors with two 1s).
    # Let's try to expand.
    
    # Optimization: 
    # The constraint is min_dist_sq >= max_norm_sq.
    # If we add points with norm N, the max_norm_sq becomes N (assuming we add in increasing norm order).
    # Then any new point must be at distance >= N from all existing points.
    
    # Let's use a generator for integer vectors with bounded L2 norm.
    # We'll iterate by max coordinate value or by norm.
    # Iterating by norm is better for packing.
    
    # Pre-compute candidates for norms 1 to some limit
    # But generating all vectors for a given norm is expensive if norm is large.
    # Let's start with small norms.
    
    # Initial set: Empty
    # We will add points one by one.
    
    # To speed up, we can vectorize the distance check against the current set.
    # However, the current set grows.
    # Let's keep points as a numpy array for fast distance calculation.
    
    current_points = np.array([], dtype=np.int64).reshape(0, d)
    current_max_norm_sq = 0
    
    # We will generate candidates in order of increasing norm squared.
    # We need a way to generate integer vectors with sum of squares = N.
    
    def generate_vectors_with_norm_sq(N, d):
        """
        Generator for integer vectors in R^d with sum of squares = N.
        """
        if N < 0:
            return
        if d == 0:
            if N == 0:
                yield np.array([], dtype=np.int64)
            return
        if N == 0:
            yield np.zeros(d, dtype=np.int64)
            return
        
        # Recursive generation
        # v[0]^2 + v[1]^2 + ... + v[d-1]^2 = N
        # v[0]^2 <= N
        # max_v0 = int(sqrt(N))
        
        for v0_sq in range(N + 1):
            v0_sq_val = v0_sq * v0_sq
            if v0_sq_val > N:
                break
            remaining_N = N - v0_sq_val
            remaining_d = d - 1
            
            # Possible values for v0
            # v0^2 = v0_sq_val
            # v0 can be sqrt(v0_sq_val) or -sqrt(v0_sq_val)
            # If v0_sq_val is a perfect square, we have +/- (or just 0)
            # Wait, v0_sq is the value of the square, not the coordinate.
            # Let's iterate over possible coordinate values c0 such that c0^2 <= N.
            pass
            
        # Better approach: iterate over c0 from -sqrt(N) to sqrt(N)
        max_c0 = int(np.sqrt(N))
        for c0 in range(-max_c0, max_c0 + 1):
            c0_sq = c0 * c0
            if c0_sq > N:
                continue
            remaining_N = N - c0_sq
            # Generate rest of the vector
            for rest in generate_vectors_with_norm_sq(remaining_N, remaining_d):
                vec = np.zeros(d, dtype=np.int64)
                vec[0] = c0
                vec[1:] = rest
                yield vec

    # This recursive generator might be slow for large N.
    # Let's stick to a simpler, iterative search for small norms first, 
    # and see if we can improve on the 220 points.
    
    # The seed solution uses norm 2.
    # Points with norm 1: e_i and -e_i. (22 points)
    # If we start with e1, max_norm_sq = 1.
    # Next point must be dist >= 1 from e1.
    # e2 is dist 1 from e1? No, dist^2 = 2. 2 >= 1. OK.
    # So we can pack many norm 1 points.
    
    # Let's try to construct a set greedily.
    # We will iterate through candidate points in a specific order.
    # Order: by norm squared, then by lexicographical order.
    
    # To avoid generating millions of candidates, we can limit the search space.
    # Let's try to add all valid points with norm <= 3 or 4.
    
    # Actually, the constraint is strict.
    # If max_norm_sq = M, all points must be at least sqrt(M) apart.
    
    # Let's try a different structural approach:
    # Use a lattice. The seed uses the A11 lattice or a subset of it?
    # The seed uses vectors with two non-zero entries.
    # This is related to the root system of type A or D?
    # Vectors with two 1s are roots of A11? No, A11 roots are e_i - e_j.
    # These are e_i + e_j, e_i - e_j, etc.
    
    # Let's just do a brute force greedy search over a bounded box.
    # Box: coordinates in {-2, -1, 0, 1, 2}.
    # Total points: 5^11 = 48,828,125. Too many to iterate fully in 600s if checking all pairs.
    # But we can filter by norm.
    
    # We will generate candidates by increasing norm.
    # Norm 1: 22 points (e_i, -e_i)
    # Norm 2: 220 points (e_i +/- e_j)
    # Norm 3: e_i +/- e_j +/- e_k? No, sum of squares = 3.
    #   (1,1,1,0...) -> 11*10*9/6 * 8 signs? 
    #   Combinations of 3 indices: C(11,3) = 165.
    #   Signs: 2^3 = 8.
    #   Total: 165 * 8 = 1320 points.
    
    # Let's try to add points with Norm 1, then Norm 2, then Norm 3.
    
    # We need to check compatibility.
    
    # Start with empty set.
    # Add all possible Norm 1 points?
    # If we add e1, max_norm_sq = 1.
    # Can we add -e1? dist^2 = 4 >= 1. Yes.
    # Can we add e2? dist^2(e1, e2) = 2 >= 1. Yes.
    # So we can add all 22 norm-1 points.
    # Now max_norm_sq is still 1.
    # Min dist among these 22 points is 1 (e.g., e1 and -e1 have dist^2=4? No, e1-e1=0? No, distinct. e1 and -e1 dist^2 = 1+1=2? No, (1-(-1))^2 = 4. e1 and e2 dist^2 = 1+1=2.
    # Wait, e1 = (1,0...), -e1 = (-1,0...). Dist^2 = (1-(-1))^2 = 4.
    # e1 and e2 = (0,1...0). Dist^2 = 1^2 + 1^2 = 2.
    # So min dist sq is 2.
    # Max norm sq is 1.
    # Condition: min_dist_sq >= max_norm_sq. 2 >= 1. True.
    
    # Now try to add Norm 2 points.
    # Example: p = e1 + e2. Norm^2 = 2.
    # Check against existing points.
    # Dist to e1: (1-1)^2 + (1-0)^2 = 1.
    # 1 < max_norm_sq (which is 1)? No, 1 >= 1 is True.
    # Wait, the condition is min_dist_sq >= max_norm_sq.
    # Current max_norm_sq is 1.
    # Dist sq to e1 is 1.
    # So 1 >= 1 holds.
    # Dist to e2: (1-0)^2 + (1-1)^2 = 1. 1 >= 1 holds.
    # Dist to -e1: (1-(-1))^2 + (1-0)^2 = 4+1=5. 5 >= 1.
    # So e1+e2 can be added?
    
    # If we add e1+e2, max_norm_sq becomes 2.
    # Now we must check all NEW pairs involving e1+e2 against the NEW max_norm_sq (2).
    # But we also need to ensure the condition holds for the WHOLE set.
    # The condition is: min_pairwise_dist_sq(ALL POINTS) >= max_norm_sq(ALL POINTS).
    # If we add a point with norm 2, max_norm_sq becomes 2.
    # We must ensure ALL pairwise distances are >= 2.
    # Previously, min dist was 2 (between e1 and e2).
    # Now we have e1+e2.
    # Dist(e1, e1+e2) = 1.
    # 1 < 2. So the condition FAILS.
    
    # Therefore, we CANNOT add norm 2 points if we have norm 1 points that are close to them.
    # Specifically, e1 and e1+e2 are distance 1 apart.
    # If max_norm_sq becomes 2, we need dist >= 2.
    # So e1 and e1+e2 cannot coexist if e1+e2 is in the set.
    
    # This implies we might have to choose between "small norm, many points" or "larger norm, fewer points but maybe better packing?"
    # The seed solution uses ONLY norm 2 points.
    # Let's check the seed solution's validity.
    # Points: all vectors with two 1s (and signs).
    # Max norm sq = 2.
    # Min dist sq:
    # Consider e1+e2 and e1-e2.
    # Diff: (1-1, 1-(-1), ...) = (0, 2, ...).
    # Dist sq = 4.
    # Consider e1+e2 and e1+e3.
    # Diff: (0, 1, -1, ...).
    # Dist sq = 1+1 = 2.
    # So min dist sq is 2.
    # Condition: 2 >= 2. Valid.
    # Number of points: 220.
    
    # Can we do better?
    # If we use only norm 1 points, max k = 22.
    # If we use only norm 2 points, max k = 220.
    # Can we mix?
    # If we include any norm 1 point, say e1.
    # And any norm 2 point, say e1+e2.
    # Dist = 1.
    # Max norm = 2.
    # 1 >= 2 is False.
    # So we cannot mix norm 1 and norm 2 points if they are "adjacent".
    # Are ALL norm 1 and norm 2 points adjacent?
    # e1 and e2+e3. Dist^2 = 1 + 1 + 1 = 3.
    # 3 >= 2. OK.
    # So e1 and e2+e3 can coexist.
    # But e1 and e1+e2 cannot.
    
    # So a mixed set is possible if we avoid conflicts.
    # This becomes a graph coloring / independent set problem.
    # Vertices: all integer vectors with norm <= N_max.
    # Edge between u, v if dist_sq(u,v) < max_norm_sq(u,v).
    # Wait, max_norm_sq is global.
    # Let M be the max norm in the set.
    # We need a set S such that for all u,v in S, dist_sq(u,v) >= M.
    # And max_norm_sq(S) = M.
    
    # To maximize |S|, we should probably keep M small.
    # If M=1, S can only contain norm 1 points. Max 22.
    # If M=2, S can contain norm 1 and norm 2 points.
    # But any pair (u,v) in S must have dist >= 2.
    # If u has norm 1 and v has norm 2:
    # dist_sq(u,v) >= 2.
    # u=(1,0..), v=(1,1,0..). Dist=1. Conflict.
    # u=(1,0..), v=(0,1,1..). Dist=3. OK.
    
    # So we can have norm 1 points and norm 2 points, provided they don't conflict.
    # The seed solution has 220 points with M=2.
    # Can we add norm 1 points to the seed solution?
    # Seed points are all v with two 1s.
    # Take e1.
    # Check dist to all seed points.
    # Seed point e1+e2. Dist(e1, e1+e2) = 1.
    # 1 < 2 (M). Conflict.
    # So e1 cannot be added to the seed set.
    
    # What if we remove conflicting points from the seed set to add e1?
    # Remove all points containing e1?
    # Points containing e1: e1+e2, e1-e2, ..., e1+e11, e1-e11. (20 points).
    # If we remove these 20, can we add e1?
    # New set: 220 - 20 = 200 points. Plus e1. Total 201.
    # Worse than 220.
    
    # So the pure norm 2 set seems locally optimal for M=2.
    
    # What about M=3?
    # Points with norm 3: e_i+e_j+e_k.
    # Number of such points: C(11,3) * 8 = 165 * 8 = 1320.
    # If we use only norm 3 points.
    # Max norm = 3.
    # Min dist between two norm 3 points?
    # (1,1,1,0..) and (1,1,-1,0..).
    # Diff: (0,0,2,0..). Dist^2 = 4.
    # 4 >= 3. OK.
    # (1,1,1,0..) and (1,1,0,1..).
    # Diff: (0,0,1,-1..). Dist^2 = 1+1=2.
    # 2 < 3. Conflict.
    # So we can't just take all norm 3 points.
    
    # This suggests that for higher norms, the density of valid points drops or requires complex selection.
    
    # Given the 600s limit, a greedy search might find a better set than 220 if it explores mixed norms or different norms.
    # However, 220 is quite high.
    # The kissing number in 11D is known to be 554 (for real vectors).
    # For integer vectors, it's a different problem.
    
    # Let's implement the greedy search to see if it beats 220.
    # We will generate candidates in order of norm.
    # We will try to add them if they fit.
    
    # To make it fast, we will use numpy for distance checks.
    
    # Candidates:
    # Norm 1: 22
    # Norm 2: 220
    # Norm 3: 1320
    # Norm 4: 
    #   (2,0..) -> 22 points.
    #   (1,1,1,1..) -> C(11,4)*16 = 330*16 = 5280 points.
    #   (1,1,0..) with one 2? No, 1^2+1^2+2^2=6.
    #   Sum of squares = 4.
    #   Partitions of 4:
    #   4 (2,0..)
    #   2+2 (1,1,0.. no, 1^2+1^2=2. 2^2+0=4. 1^2+1^2+1^2+1^2=4)
    #   So (2,0..) and (1,1,1,1..).
    
    # We'll generate candidates up to Norm 4 or 5.
    
    # Algorithm:
    # 1. Generate all candidates with norm <= N_max.
    # 2. Sort by norm, then by index.
    # 3. Iterate. If point fits (min dist to current set >= current_max_norm), add it.
    #    Note: if we add a point with larger norm, current_max_norm increases.
    #    This might invalidate previously added points?
    #    No, previously added points had dist >= OLD_max_norm.
    #    If NEW_max_norm > OLD_max_norm, we need dist >= NEW_max_norm.
    #    So we might need to REMOVE points that are now too close.
    #    This makes greedy "add only" suboptimal.
    
    # Better Greedy:
    # Fix a target Max Norm M.
    # Find the largest subset of points with norm <= M such that min dist >= M.
    # Try M=1, 2, 3, 4...
    # For each M, solve the packing problem.
    
    # For M=2, we know the answer is 220 (all norm 2 points).
    # For M=1, answer is 22.
    # For M=3, let's try to find a large set.
    
    # Given time constraints, I will implement a greedy search that tries to build a set for a fixed M,
    # starting with M=2 (the known good one) and trying to extend or find better for M=3.
    
    # Actually, the simplest structural change is to just run a greedy search over a larger candidate pool
    # and see what it finds.
    
    # Let's generate candidates for Norm 1, 2, 3.
    # Total candidates: 22 + 220 + 1320 = 1562.
    # We can check all pairs of candidates? 1562^2 / 2 = ~1.2M pairs.
    # This is fast enough in Python with numpy?
    # 1.2M iterations in a Python loop might take ~1-2 seconds.
    # So we can afford to check compatibility.
    
    # Strategy:
    # 1. Generate candidates for Norm 1, 2, 3.
    # 2. Try to construct a set with Max Norm = 2.
    #    - Start with all Norm 2 points.
    #    - Try to add Norm 1 points if they fit.
    # 3. Try to construct a set with Max Norm = 3.
    #    - Start with empty set.
    #    - Greedily add points from candidates sorted by norm, then random/lex order.
    #    - When adding a point, check against all existing points.
    #    - If max norm of set becomes 3, check dist >= 3.
    
    # Let's implement this.
    
    def generate_candidates(d, max_norm):
        cands = []
        # Norm 1
        for i in range(d):
            for s in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = s
                cands.append(p)
        
        # Norm 2
        for i, j in itertools.combinations(range(d), 2):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    cands.append(p)
                    
        # Norm 3
        for i, j, k in itertools.combinations(range(d), 3):
            for si in [1, -1]:
                for sj in [1, -1]:
                    for sk in [1, -1]:
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = si
                        p[j] = sj
                        p[k] = sk
                        cands.append(p)
                        
        return cands

    candidates = generate_candidates(d, 3)
    # Sort by norm squared
    candidates.sort(key=lambda p: np.dot(p, p))
    
    best_points = np.array([], dtype=np.int64).reshape(0, d)
    best_count = 0
    
    # Try M=2
    # We know all norm 2 points form a valid set.
    # Let's verify and count.
    norm2_points = [p for p in candidates if np.dot(p, p) == 2]
    # Check if all norm 2 points are valid together.
    # We already know they are.
    # Can we add norm 1 points?
    # We'll try to add norm 1 points to the norm 2 set.
    
    set_M2 = np.array(norm2_points, dtype=np.int64)
    # Check validity of set_M2
    # It should be valid.
    
    # Try to add norm 1 points
    norm1_points = [p for p in candidates if np.dot(p, p) == 1]
    
    # For each norm 1 point, check if it fits in set_M2
    # If it fits, we add it. But adding it might force removal of norm 2 points?
    # No, we are just checking if we can ADD it.
    # If we add it, max_norm becomes 2 (still).
    # We need dist(p, q) >= 2 for all q in set_M2.
    
    added_norm1 = []
    for p in norm1_points:
        # Check dist to all in set_M2
        # dist_sq = ||p - q||^2
        # We need min dist_sq >= 2
        diffs = p - set_M2
        dists_sq = np.sum(diffs**2, axis=1)
        if np.min(dists_sq) >= 2:
            added_norm1.append(p)
            
    # If we can add some norm 1 points, the new set is set_M2 + added_norm1.
    # But wait, do the added norm 1 points conflict with EACH OTHER?
    # Dist between e1 and e2 is 2. 2 >= 2. OK.
    # Dist between e1 and -e1 is 4. OK.
    # So any subset of norm 1 points is valid internally.
    # And we checked they are valid against set_M2.
    # So we can add ALL compatible norm 1 points.
    
    if added_norm1:
        new_set_M2 = np.vstack([set_M2, np.array(added_norm1)])
        if len(new_set_M2) > best_count:
            best_points = new_set_M2
            best_count = len(new_set_M2)
    else:
        if len(set_M2) > best_count:
            best_points = set_M2
            best_count = len(set_M2)

    # Try M=3 with a greedy approach
    # We will try to build a set with max norm 3.
    # Start with empty set.
    # Candidates sorted by norm.
    # We prioritize adding points that allow more future points?
    # Simple greedy: add if possible.
    
    set_M3 = np.array([], dtype=np.int64).reshape(0, d)
    current_max_norm_M3 = 0
    
    # To speed up, we can restrict candidates to those that might fit.
    # But let's just try all.
    
    # Optimization: if time is running out, stop.
    
    for p in candidates:
        if time.time() > deadline - 5:
            break
            
        p_norm_sq = np.dot(p, p)
        
        # If p_norm_sq > 3, skip (we are targeting M=3)
        if p_norm_sq > 3:
            continue
            
        # Check if p can be added
        if len(set_M3) == 0:
            set_M3 = np.array([p], dtype=np.int64)
            current_max_norm_M3 = p_norm_sq
            continue
            
        # Check dist to all existing points
        diffs = p - set_M3
        dists_sq = np.sum(diffs**2, axis=1)
        min_dist = np.min(dists_sq)
        
        # The new max norm will be max(current_max_norm_M3, p_norm_sq)
        new_max_norm = max(current_max_norm_M3, p_norm_sq)
        
        if min_dist >= new_max_norm:
            set_M3 = np.vstack([set_M3, p])
            current_max_norm_M3 = new_max_norm
            
    if len(set_M3) > best_count:
        best_points = set_M3
        best_count = len(set_M3)

    # Final check and return
    if len(best_points) == 0:
        # Fallback
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    # Verify best_points
    norms_sq = np.sum(best_points**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    min_dist_sq = float('inf')
    for i in range(len(best_points)):
        for j in range(i+1, len(best_points)):
            diff = best_points[i] - best_points[j]
            dist_sq = int(np.sum(diff**2))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                
    if min_dist_sq < max_norm_sq:
        # Invalid, fallback to seed solution
        points_list = []
        for i, j in itertools.combinations(range(d), 2):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points_list.append(p)
        best_points = np.array(points_list, dtype=np.int64)
        
    return np.around(best_points).astype(np.int64)

# EVOLVE-BLOCK-END
