import numpy as np
import time
import random
from itertools import product

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit
    
    # Strategy: Geometric Greedy Construction with Dynamic Norm Expansion
    # 1. Start with an empty set.
    # 2. We want to maximize the number of points k.
    # 3. Constraint: min_{i,j} ||p_i - p_j||^2 >= max_i ||p_i||^2.
    #    Let R be the current max squared norm in the set.
    #    When adding a new point p, we must ensure:
    #    a) ||p||^2 <= R (if we don't want to increase R) OR if we do increase R, 
    #       the new R' = max(R, ||p||^2) must satisfy dist(p, q) >= R' for all q in set.
    #    
    #    Actually, the constraint is global. 
    #    If we add a point with norm > R, the new radius is R_new = ||p||^2.
    #    Then ALL pairs in the set (including old ones) must have dist >= R_new.
    #    This is a very strong constraint. If we start with small norms, we can't 
    #    easily add a large norm point later because it would invalidate the 
    #    distances between the existing small norm points.
    #    
    #    Therefore, a better strategy is:
    #    Fix a target radius R (squared norm bound).
    #    Find the maximum size set S such that all p in S have ||p||^2 <= R 
    #    and all pairwise distances >= R.
    #    
    #    The previous program tried R=1, 2, 3.
    #    Let's try larger R values? 
    #    If R is large, we can use vectors with larger norms.
    #    However, the density of integer vectors decreases as norm increases? No, 
    #    the number of integer vectors with norm^2 <= R grows roughly as R^(d/2 - 1) * R.
    #    But the distance constraint dist >= R becomes harder to satisfy as R grows?
    #    Actually, if R grows, the "exclusion zone" around each point grows (radius sqrt(R)).
    #    The volume of the exclusion zone is ~ R^(d/2).
    #    The volume of the ball of radius sqrt(R) containing all points is ~ R^(d/2).
    #    So the packing fraction might stay constant or decrease.
    #    
    #    Known kissing numbers:
    #    d=2: 6 (R=1: 6 points on unit circle? No, integer points. 
    #       (1,0), (-1,0), (0,1), (0,-1) -> dist 2, norm 1. Valid. k=4.
    #       Can we get 6? (1,0), (-1,0), (0,1), (0,-1) is 4.
    #       Add (1,1)? norm 2. R becomes 2.
    #       Dist((1,0), (1,1)) = 1 < 2. Invalid.
    #       So for integers, k might be limited.
    #       
    #    Let's analyze the current best: 110.
    #    The current program found 110 points.
    #    It used vectors of norm 1, 2, 3.
    #    If R=3, we need dist >= 3.
    #    
    #    Let's try a different approach: 
    #    Use a randomized greedy algorithm that attempts to build a set for a fixed R,
    #    but with a smarter candidate ordering and possibly multiple restarts.
    #    The previous program used MIS which is exact but potentially slow or limited 
    #    by the candidate pool size.
    #    
    #    New Strategy: 
    #    1. Generate a large pool of candidate vectors with norm^2 <= R_max.
    #    2. For a given R, filter pool to norm^2 <= R.
    #    3. Use a Greedy Algorithm:
    #       - Compute all pairwise distances.
    #       - Repeatedly pick the point that has the "most room" or "least conflict" 
    #         with the already selected set, or use a heuristic like "pick point with 
    #         max min-distance to existing set".
    #       - Actually, a standard greedy for packing is:
    #         a. Pick a random point (or one with high centrality).
    #         b. Add it to set.
    #         c. Remove all points that conflict with it (dist < R).
    #         d. Repeat.
    #       - This is the same as the greedy MIS step in the previous program.
    #       - The previous program's greedy might have been suboptimal.
    #       
    #    Let's try to improve the greedy heuristic.
    #    Instead of picking min degree (which is fast), maybe pick the point that 
    #    maximizes the minimum distance to the *current* set? 
    #    This is O(N*K) per step, which might be slow.
    #    
    #    Alternative: Use a different candidate pool.
    #    The previous pool was norm 1, 2, 3.
    #    What if we include norm 4, 5, ...?
    #    If R=4, we need dist >= 4.
    #    Vectors of norm 4: e.g., (2,0,...), (1,1,0,...) with 4 ones? No, norm^2=4.
    #    (2,0) has norm 4.
    #    (1,1,0) has norm 2.
    #    
    #    Let's check if a larger R allows a larger k.
    #    The kissing number in R^11 is 558 (known value).
    #    But we are restricted to INTEGER points.
    #    The integer kissing number is much smaller.
    #    
    #    Let's try to optimize the search for R=3 first, but with a better algorithm.
    #    The previous program used a recursive MIS solver. 
    #    Maybe the pool was too small?
    #    Pool size for R=3:
    #    Norm 1: 22
    #    Norm 2: 220
    #    Norm 3: 1320
    #    Total: 1562.
    #    
    #    Let's try to include Norm 4 vectors in the pool for R=4?
    #    If R=4, we need dist >= 4.
    #    Points with norm 4: 
    #    (2,0,...) -> 11 * 2 = 22 vectors.
    #    (1,1,1,1,0...) -> 4 ones. C(11,4) * 2^4 = 330 * 16 = 5280 vectors.
    #    (1,1,0...) is norm 2.
    #    
    #    Total pool for R=4 would be 1562 + 5302 = ~6864.
    #    Building a conflict graph for 6864 nodes: 6864^2 / 2 = 23M pairs.
    #    This is feasible in Python if optimized with NumPy, but the MIS search 
    #    on 6864 nodes is likely to time out or be intractable with simple recursion.
    #    
    #    However, a GREEDY algorithm is O(N^2) or O(N^3) and might finish.
    #    Greedy packing often gets close to optimal for random graphs or geometric graphs.
    #    
    #    Let's implement a robust Greedy Algorithm for multiple R values.
    #    We will use a "Farthest Point" heuristic or "Min Conflict" heuristic.
    #    
    #    Algorithm:
    #    1. Generate candidates up to norm^2 = R_max (e.g., 5 or 6).
    #    2. For each R in [1, 2, 3, 4, 5]:
    #       a. Filter candidates with norm^2 <= R.
    #       b. Compute distance matrix (or on the fly).
    #       c. Run Greedy:
    #          - Start with empty set.
    #          - While candidates remain:
    #             - Pick the candidate that maximizes the minimum distance to the current set?
    #               (This is the "Max Min" strategy, good for spreading points).
    #             - If no such point exists (all conflict), stop.
    #             - Add to set.
    #             - Remove all points that conflict with the new point (dist < R).
    #          - Record size.
    #    3. Return the best set found.
    #    
    #    To speed up "Max Min", we can approximate or use a simpler heuristic.
    #    Actually, for geometric packing, "pick random" or "pick first" often works okay, 
    #    but "Max Min" is better.
    #    With N=1500, "Max Min" is O(N^2) per pick, total O(N^3) ~ 3.3B ops. Too slow.
    #    
    #    Let's stick to "Min Degree" greedy (fast) but try multiple random permutations 
    #    or random subsets to escape local optima.
    #    Or, use the "Min Degree" greedy but on a subgraph?
    #    
    #    Let's try the "Min Degree" greedy on the full pool for R=3, 4, 5.
    #    And also try a "Randomized Greedy" where we shuffle the candidates 
    #    before applying the greedy rule.
    
    best_set = []
    best_size = 0
    
    # Generate candidates up to norm_sq = 6
    # This covers R=1..6
    candidates = []
    # Norm 1
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 1
        candidates.append(v)
        candidates.append(-v)
        
    # Norm 2
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
                    
    # Norm 3
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        for s3 in [1, -1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            candidates.append(v)
                            
    # Norm 4
    # Type 1: (2, 0, ...) -> 11 * 2 = 22
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 2
        candidates.append(v)
        candidates.append(-v)
    # Type 2: (1, 1, 1, 1, 0...) -> C(11,4) * 2^4 = 330 * 16 = 5280
    for i in range(d):
        for j in range(i+1, d):
            for k in range(j+1, d):
                for l in range(k+1, d):
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
                                    
    # Norm 5
    # Type 1: (2, 1, 0...) -> 11*10*2*2 = 440
    for i in range(d):
        for j in range(i+1, d):
            for s1 in [2, -2]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
    # Type 2: (1, 1, 1, 1, 1, 0...) -> C(11,5) * 2^5 = 462 * 32 = 14784
    # This is getting large. Total candidates ~ 1562 + 5302 + 15224 = ~22000.
    # Building distance matrix for 22000 is 22000^2 = 484M entries. 
    # 484M * 8 bytes = 3.8 GB. Too much memory.
    # We must compute distances on the fly or in blocks.
    
    # Let's limit the pool to Norm <= 4 for memory safety, or use on-the-fly checks.
    # For R=4, pool is ~6864. 6864^2 = 47M. 47M * 8 = 376 MB. This is borderline but okay.
    # For R=5, pool is ~22000. Too big for full matrix.
    
    # Let's stick to R=1,2,3,4 with full matrix, and maybe R=5 with on-the-fly?
    # Or just optimize R=3 and 4.
    
    unique_candidates = []
    seen = set()
    for v in candidates:
        if np.all(v == 0):
            continue
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(v)
            
    n_cand = len(unique_candidates)
    cand_array = np.array(unique_candidates, dtype=np.int64)
    norms_sq = np.sum(cand_array**2, axis=1)
    
    # We will test R = 1, 2, 3, 4
    # For each R, we run a randomized greedy search multiple times.
    
    best_size = 0
    best_set_indices = []
    
    for R in [1, 2, 3, 4]:
        if time.time() > deadline:
            break
            
        valid_indices = np.where(norms_sq <= R)[0]
        if len(valid_indices) < 2:
            continue
            
        sub_cands = cand_array[valid_indices]
        n_sub = len(sub_cands)
        
        # Compute distance matrix for this subgraph
        # If n_sub is too large, we might skip or use approximation.
        if n_sub > 8000:
            # Skip R=5 if we went that far, but we stopped at 4.
            # n_sub for R=4 is ~6864.
            pass
            
        dots = sub_cands @ sub_cands.T
        dist_sq = norms_sq[valid_indices, None] + norms_sq[valid_indices, None].T - 2 * dots
        # dist_sq is n_sub x n_sub
        
        # Conflict if dist_sq < R
        # We want Max Independent Set.
        # Greedy: 
        # 1. Compute degree (number of conflicts) for each node.
        #    deg[i] = count of j != i such that dist_sq[i,j] < R.
        #    Note: dist_sq[i,i] = 0 < R. So deg includes self? 
        #    Usually we exclude self.
        #    conflict_mask = dist_sq < R
        #    conflict_mask[range(n), range(n)] = False
        #    deg = conflict_mask.sum(axis=1)
        
        conflict_mask = dist_sq < R
        np.fill_diagonal(conflict_mask, False)
        deg = conflict_mask.sum(axis=1)
        
        # Run multiple greedy trials
        # Trial 1: Min Degree Greedy (Deterministic)
        # Trial 2..10: Randomized Min Degree (Pick min degree from top 10%?)
        # Or just shuffle and pick first?
        
        # Let's do 20 trials.
        for trial in range(20):
            if time.time() > deadline:
                break
                
            available = np.ones(n_sub, dtype=bool)
            selected = []
            
            # Heuristic: 
            # If trial == 0: Pick min degree.
            # Else: Pick random from nodes with degree < threshold?
            # Or: Pick node that minimizes degree, but with some randomness.
            
            while available.any():
                # Get indices of available nodes
                avail_idx = np.where(available)[0]
                
                if len(avail_idx) == 0:
                    break
                    
                # Compute degrees for available nodes
                # deg[avail_idx]
                # But degrees change as we remove nodes?
                # In greedy MIS, we typically don't update degrees dynamically 
                # (that's the "Simple Greedy"). 
                # A better greedy is "Dynamic Greedy" where we update degrees.
                # Updating degrees is O(N) per removal, total O(N^2).
                # Let's use Dynamic Greedy.
                
                # To make it dynamic, we maintain a `current_deg` array.
                # When we pick node u, we remove u and its neighbors.
                # For each neighbor v of u, current_deg[v] decreases by 1 (if v is still available).
                # Also, for any node w, if u was a neighbor of w, w's degree decreases.
                
                # This is complex to implement efficiently in NumPy without loops.
                # Let's stick to the static degree greedy first, it's fast.
                # To add randomness, we can pick the node with the minimum degree 
                # among a random subset of available nodes?
                
                # Strategy: 
                # 1. Get degrees of available nodes.
                # 2. Find min degree value.
                # 3. Get all nodes with min degree.
                # 4. If trial == 0: pick first.
                #    Else: pick random from these.
                
                avail_degrees = deg[avail_idx]
                min_deg = np.min(avail_degrees)
                min_nodes = avail_idx[avail_degrees == min_deg]
                
                if trial == 0:
                    pick_idx = min_nodes[0]
                else:
                    pick_idx = min_nodes[random.randint(0, len(min_nodes) - 1)]
                    
                selected.append(pick_idx)
                
                # Remove pick_idx and its neighbors from available
                # Neighbors are those where conflict_mask[pick_idx] is True
                to_remove = conflict_mask[pick_idx] | (np.arange(n_sub) == pick_idx)
                available[to_remove] = False
                
            current_size = len(selected)
            
            if current_size > best_size:
                best_size = current_size
                best_set_indices = selected[:]
                
    # Map indices back to vectors
    if best_size >= 2:
        # Note: best_set_indices are indices into sub_cands of the LAST R that was processed?
        # NO! We need to track which R and which sub_cands.
        # My loop above overwrites best_set_indices with indices from the current sub_cands.
        # I need to store the actual vectors.
        
        # Let's refactor: Store the actual vectors in best_set.
        
        # Re-run the logic to store vectors directly.
        
        best_set_final = []
        best_size_final = 0
        
        for R in [1, 2, 3, 4]:
            if time.time() > deadline:
                break
                
            valid_indices = np.where(norms_sq <= R)[0]
            if len(valid_indices) < 2:
                continue
                
            sub_cands = cand_array[valid_indices]
            n_sub = len(sub_cands)
            
            dots = sub_cands @ sub_cands.T
            dist_sq = norms_sq[valid_indices, None] + norms_sq[valid_indices, None].T - 2 * dots
            
            conflict_mask = dist_sq < R
            np.fill_diagonal(conflict_mask, False)
            deg = conflict_mask.sum(axis=1)
            
            for trial in range(20):
                if time.time() > deadline:
                    break
                    
                available = np.ones(n_sub, dtype=bool)
                selected_vecs = []
                
                while available.any():
                    avail_idx = np.where(available)[0]
                    if len(avail_idx) == 0:
                        break
                        
                    avail_degrees = deg[avail_idx]
                    min_deg = np.min(avail_degrees)
                    min_nodes = avail_idx[avail_degrees == min_deg]
                    
                    if trial == 0:
                        pick_idx = min_nodes[0]
                    else:
                        pick_idx = min_nodes[random.randint(0, len(min_nodes) - 1)]
                        
                    selected_vecs.append(sub_cands[pick_idx])
                    
                    to_remove = conflict_mask[pick_idx] | (np.arange(n_sub) == pick_idx)
                    available[to_remove] = False
                    
                if len(selected_vecs) > best_size_final:
                    best_size_final = len(selected_vecs)
                    best_set_final = selected_vecs[:]
                    
        if best_size_final >= 2:
            result = np.array(best_set_final, dtype=np.int64)
            # Verify
            max_norm = np.max(np.sum(result**2, axis=1))
            valid = True
            for i in range(len(result)):
                for j in range(i+1, len(result)):
                    d2 = np.dot(result[i] - result[j], result[i] - result[j])
                    if d2 < max_norm:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return result
            else:
                return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        else:
            return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    else:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
