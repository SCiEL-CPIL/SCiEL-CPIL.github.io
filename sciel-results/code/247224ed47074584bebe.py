import numpy as np
import itertools
import time
import sys

def generate_candidates(d=11, max_norm_sq=4):
    """
    Generate a diverse set of candidate integer vectors in Z^d.
    We include vectors with squared norms 1, 2, 3, and 4.
    - Norm 1: e_i (11 vectors)
    - Norm 2: e_i +/- e_j (220 vectors)
    - Norm 3: e_i +/- e_j +/- e_k (660 vectors)
    - Norm 4: 2e_i (22 vectors), e_i+e_j+e_k+e_l (330 vectors), e_i+2e_j (110 vectors)
    
    Total candidates ~ 1350. This is small enough for graph-based selection.
    """
    candidates = []
    
    # Norm 1: (1, 0, ..., 0) permutations
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 1
        candidates.append(v)
        # Also -1? The set should be symmetric or we pick one. 
        # If we pick both v and -v, dist is 2*norm. 
        # Condition: min_dist >= max_norm.
        # If max_norm is 4, dist(v, -v) = 2*1 = 2 < 4. So we can't mix v and -v if max_norm is large.
        # We will let the selection algorithm handle this.
        v_neg = -v
        candidates.append(v_neg)

    # Norm 2: (1, 1, 0...), (1, -1, 0...)
    for i, j in itertools.combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)

    # Norm 3: (1, 1, 1...), (1, 1, -1...), etc.
    # Combinations of 3 indices
    for i, j, k in itertools.combinations(range(d), 3):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)

    # Norm 4 types:
    # Type A: 2*e_i (22 vectors: +/- 2e_i)
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 2
        candidates.append(v)
        v[i] = -2
        candidates.append(v)
        
    # Type B: e_i + e_j + e_k + e_l (330 vectors)
    for i, j, k, l in itertools.combinations(range(d), 4):
        # To keep candidate set manageable, we might skip full norm 4 generation if it's too large
        # C(11,4) * 2^4 = 330 * 16 = 5280 vectors. 
        # 5280 nodes is fine for graph construction, but greedy selection might be slow if not optimized.
        # Let's include them but maybe limit signs? No, let's include all.
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    for s4 in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        v[k] = s3
                        v[l] = s4
                        candidates.append(v)
                        
    # Type C: 2e_i + e_j (110 vectors: 11 choices for i, 10 for j, 2 signs for 2e_i, 2 for e_j = 440? No.
    # 2e_i + e_j: i!=j. 11*10 = 110 pairs. Signs: +/-2, +/-1. 4 sign combos. 440 vectors.
    for i in range(d):
        for j in range(d):
            if i == j: continue
            for s2 in [2, -2]:
                for s1 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s2
                    v[j] = s1
                    candidates.append(v)

    # Convert to array
    return np.array(candidates, dtype=np.int64)

def solve_max_independent_set_greedy(candidates, time_limit=500):
    """
    Greedy approach to find a large subset of candidates satisfying the condition.
    Condition: min_dist_sq >= max_norm_sq.
    
    Strategy:
    1. Sort candidates by norm squared descending.
    2. Iterate through candidates.
    3. For each candidate, check if adding it to the current set violates the condition.
       The condition is global: min_dist of the NEW set >= max_norm of the NEW set.
       Since we process by descending norm, max_norm is fixed by the first element added (or current max).
       Actually, if we add a vector with norm N, the max_norm of the set becomes max(current_max, N).
       If we process descending, max_norm is non-increasing? No, we start with high norms.
       
       Let's refine:
       The constraint is: for all pairs (u, v) in S, ||u-v||^2 >= M, where M = max_{w in S} ||w||^2.
       
       Greedy:
       Start with empty set S.
       Current M = 0.
       Current min_dist = infinity.
       
       Iterate candidates sorted by norm descending.
       For candidate c:
           new_M = max(M, ||c||^2)
           # Check if c is far enough from all existing points in S
           # dist(c, u)^2 >= new_M for all u in S?
           # AND min_dist(S) >= new_M?
           # If min_dist(S) < new_M, then adding c might not fix it, but if c has smaller norm, 
           # M doesn't change. If c has larger norm, M increases, which might invalidate existing pairs.
           
       This dynamic is tricky. A simpler robust heuristic:
       1. Generate all candidates.
       2. For a fixed target max_norm M (e.g., 1, 2, 3, 4), find the max independent set in the graph
          where nodes are candidates with norm <= M, and edges connect pairs with dist^2 < M.
          Wait, the condition is min_dist >= M. So edges exist if dist^2 < M.
          We want a set with NO edges.
          
       Let's try this fixed-M approach. It's structurally different from the "homogeneous norm" assumption 
       because for M=4, we can include vectors of norm 1, 2, 3, 4 as long as they are mutually far enough.
       
       Candidates with norm > M are excluded? No, if we include a vector with norm > M, then M increases.
       So we should iterate over possible M values (1, 2, 3, 4, 5, 6...) and for each M, 
       consider only candidates with norm <= M? No, if we have a vector with norm 5, M becomes 5.
       
       So:
       For M in [1, 2, 3, 4, 5, 6, 8, 9, 10, 12...]:
           Filter candidates to those with norm_sq <= M.
           Build conflict graph: edge (i, j) if dist_sq(i, j) < M.
           Find Max Independent Set (MIS) in this graph.
           The size of MIS is a valid k for this M.
           Track the best k.
           
       This is a solid structural change: Graph-based optimization per norm threshold.
    """
    
    # Precompute norms
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Unique norm values to consider as M
    # We only care about M values that are actual norms or slightly above.
    # But M is the max norm in the set. So M must be equal to the norm of some vector in the set.
    unique_norms = np.unique(norms_sq)
    
    best_k = 0
    best_set = np.array([])
    
    start_time = time.time()
    
    for M in unique_norms:
        if time.time() - start_time > time_limit:
            break
            
        # Candidates must have norm <= M
        mask = norms_sq <= M
        sub_candidates = candidates[mask]
        sub_norms = norms_sq[mask]
        
        if len(sub_candidates) < 2:
            continue
            
        # If the max norm in sub_candidates is less than M, then the actual M for the set 
        # will be lower. We should only consider M if there is at least one vector with norm == M.
        if not np.any(sub_norms == M):
            continue
            
        n_sub = len(sub_candidates)
        
        # Build adjacency matrix or list of conflicts
        # dist^2(i, j) < M
        # We can compute pairwise distances.
        # n_sub is at most ~1300. 1300^2 = 1.69M ops. Fast.
        
        # Vectorized distance calculation
        # sub_candidates shape (n, 11)
        # diff = sub_candidates[:, np.newaxis, :] - sub_candidates[np.newaxis, :, :]
        # dist_sq = np.sum(diff**2, axis=2)
        
        # This creates a 1300x1300x11 array, which is ~18MB. Acceptable.
        diff = sub_candidates[:, np.newaxis, :] - sub_candidates[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Conflict matrix: True if dist_sq < M
        # Diagonal is 0, which is < M, so we need to ignore self.
        conflict = dist_sq < M
        np.fill_diagonal(conflict, False)
        
        # Greedy MIS heuristic:
        # Order nodes by degree (number of conflicts) ascending? 
        # Or descending? 
        # Low degree nodes are "less constrained", easier to add.
        # Standard greedy for MIS: pick node with min degree, add to set, remove neighbors.
        
        degree = np.sum(conflict, axis=1)
        
        selected = np.zeros(n_sub, dtype=bool)
        remaining = np.ones(n_sub, dtype=bool)
        
        # To speed up, we can use a simple greedy loop.
        # Since n_sub is small, we can afford O(n^2) or O(n^3) per M.
        
        while np.any(remaining):
            # Find unselected node with min degree among remaining
            # degree[~remaining] is not directly usable if we don't update degrees dynamically.
            # For a simple heuristic, we just pick the first available with min initial degree?
            # Or recompute? Recomputing is slow.
            # Let's use a static ordering based on initial degree, or just random?
            # Static min-degree greedy is a good baseline.
            
            # Get indices of remaining nodes
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0: break
            
            # Sort by initial degree? Or current degree?
            # Let's try sorting by initial degree ascending.
            # This is a static approximation.
            # To make it better, we can recompute degrees for remaining nodes occasionally.
            
            # For simplicity and speed, let's just pick the node with min initial degree from remaining.
            # We can pre-sort indices by degree.
            # But we need to skip selected ones.
            
            # Efficient way:
            # Iterate through sorted indices, pick first that is remaining.
            
            # Pre-sort once outside the while loop for this M?
            # Yes.
            
            # Wait, the while loop structure above is inefficient if we just check all.
            # Let's restructure.
            
            # Actually, for n=1300, a simple O(n^2) greedy is fine.
            # Pick node u with min degree in current induced subgraph.
            # Add u to set.
            # Remove u and its neighbors from remaining.
            
            # To find min degree dynamically:
            # We can maintain a list of active nodes.
            
            active = list(np.where(remaining)[0])
            
            if not active:
                break
            
            # Find node with min conflicts to other active nodes
            # This is O(|active|^2) per step. Total O(n^3) worst case. 1300^3 is too slow.
            
            # Optimization: Use a static greedy order.
            # Sort all nodes by degree.
            # Iterate through sorted list.
            # If node is not selected and not neighbor of any selected node, select it.
            
            # This is a standard greedy MIS. It's O(n^2) total if implemented carefully.
            
            # Let's implement this static greedy.
            
            # We need the conflict matrix for this M.
            # sorted_indices = np.argsort(degree)
            
            # selected_mask = np.zeros(n_sub, dtype=bool)
            # neighbor_mask = np.zeros(n_sub, dtype=bool) # neighbors of selected
            
            # for idx in sorted_indices:
            #     if selected_mask[idx] or neighbor_mask[idx]:
            #         continue
            #     selected_mask[idx] = True
            #     neighbor_mask |= conflict[idx]
            
            # This is very fast.
            
            sorted_indices = np.argsort(degree)
            selected_mask = np.zeros(n_sub, dtype=bool)
            neighbor_mask = np.zeros(n_sub, dtype=bool)
            
            for idx in sorted_indices:
                if selected_mask[idx] or neighbor_mask[idx]:
                    continue
                selected_mask[idx] = True
                neighbor_mask |= conflict[idx]
                
            k_curr = np.sum(selected_mask)
            
            if k_curr > best_k:
                best_k = k_curr
                best_set = sub_candidates[selected_mask]
                
    return best_set

def validate_configuration(points):
    """
    Validate the configuration according to the task contract.
    Returns True if valid, False otherwise.
    """
    if points.shape[1] != 11:
        return False
    k = points.shape[0]
    if k < 2:
        return False
    
    # Check no zero vector
    if np.any(np.all(points == 0, axis=1)):
        return False
        
    # Calculate max squared norm
    norms_sq = np.sum(points**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Calculate min squared distance
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            d_sq = np.sum(diff**2)
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                
    # Condition: min_dist_sq >= max_norm_sq
    return min_dist_sq >= max_norm_sq

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    # Generate diverse candidates
    candidates = generate_candidates(11)
    
    # Solve using graph-based greedy
    best_points = solve_max_independent_set_greedy(candidates, time_limit=500)
    
    # Ensure dtype and rounding
    if len(best_points) == 0:
        # Fallback to D11 norm 2 set if greedy failed
        def generate_norm2_vectors(d=11):
            vecs = []
            for idxs in itertools.combinations(range(d), 2):
                i, j = idxs
                for si in [1, -1]:
                    for sj in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        vecs.append(v)
            return np.array(vecs)
        best_points = generate_norm2_vectors(11)
        
    best_points = np.around(best_points).astype(np.int64)
    
    # Verify validity
    if validate_configuration(best_points):
        return best_points
    else:
        # Fallback: Return a known valid small configuration
        fallback = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.int64)
        return fallback

if __name__ == "__main__":
    res = kissing_number11()
    print(f"Found {res.shape[0]} points")
