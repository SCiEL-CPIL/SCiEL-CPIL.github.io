import numpy as np
import time
import random
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Generate candidate vectors
    # Type 1: Norm^2 = 2 (Weight 2, +/-1)
    # From E8 x D3 structure to ensure good packing, but we can also use all weight 2 vectors?
    # All weight 2 vectors in R^11: C(11,2) * 4 = 55 * 4 = 220.
    # Wait, E8 minimal vectors are weight 4 in 8D. 
    # In R^11, "minimal" integer vectors are weight 1 (norm^2=1), weight 2 (norm^2=2), etc.
    # If we allow norm^2=1, max norm^2=1. Distance >= 1.
    # Distance between (1,0,..) and (-1,0,..) is 2. Distance^2=4 >= 1.
    # Distance between (1,0,..) and (0,1,..) is sqrt(2). Distance^2=2 >= 1.
    # So all 22 unit vectors (e_i, -e_i) are valid?
    # Check: u=(1,0,..), v=(0,1,..). ||u-v||^2 = 1+1-0 = 2 >= 1. Yes.
    # u=(1,0,..), v=(-1,0,..). ||u-v||^2 = 1+1-2(-1) = 4 >= 1. Yes.
    # So we can take all 22 vectors of norm^2=1.
    # Can we add more?
    # If we add a vector w of norm^2=2, max norm^2 becomes 2.
    # Then distance between any pair must be >= 2.
    # Distance between u=(1,0,..) and v=(0,1,..) is sqrt(2). 2 >= 2. OK.
    # Distance between u=(1,0,..) and w=(1,1,0,..) (norm^2=2).
    # ||u-w||^2 = 1+2 - 2(1) = 1. 1 < 2. Fail.
    # So we cannot mix norm^2=1 and norm^2=2 vectors if they share a support coordinate with same sign.
    # Specifically, if u is a unit vector, and w has a 1 in the same position, distance^2 = 1.
    # So w must be 0 in all positions where u is non-zero?
    # If u=e_i, w must have w_i=0.
    # If we select all 22 unit vectors, any w of norm^2>=1 must have 0 in all positions? Impossible.
    # So we can either use ONLY norm^2=1 vectors (k=22) or use higher norms.
    
    # Strategy: Use norm^2=2 vectors.
    # Pool: All weight 2 vectors in R^11.
    # Count: C(11,2) * 2^2 = 55 * 4 = 220.
    # Constraint: For selected set S of norm^2=2 vectors,
    # ||u-v||^2 >= 2.
    # 2+2 - 2 u.v >= 2 => 2 u.v <= 2 => u.v <= 1.
    # Since u,v are weight 2 with +/-1, u.v is integer in {-2, -1, 0, 1, 2}.
    # Conflict if u.v == 2. (Same sign, same support).
    # u.v == 2 implies u == v. But we pick distinct vectors.
    # So u.v cannot be 2 for distinct vectors?
    # Wait. u=(1,1,0..), v=(1,1,0..). u.v=2.
    # Are there other pairs with u.v=2?
    # If u and v have same support and same signs, they are the same vector.
    # So for distinct weight 2 vectors, u.v <= 1.
    # Let's verify.
    # u=(1,1,0..), v=(1,-1,0..). u.v = 1-1=0.
    # u=(1,1,0..), v=(1,0,1..). u.v = 1.
    # u=(1,1,0..), v=(0,1,1..). u.v = 1.
    # Max dot product for distinct weight 2 vectors is 1.
    # So ALL 220 weight 2 vectors form a valid kissing configuration!
    # Score = 220.
    
    # Can we do better with norm^2=3?
    # Pool: Weight 3 vectors. C(11,3)*2^3 = 165*8 = 1320.
    # Constraint: ||u-v||^2 >= 3.
    # 3+3 - 2 u.v >= 3 => 3 u.v <= 3 => u.v <= 1.
    # Conflict if u.v >= 2.
    # u,v weight 3. Max u.v=3 (same vector).
    # Distinct vectors can have u.v=2 or 3?
    # If they share 2 coordinates with same sign, and 1 different, u.v = 1+1-1 = 1? No.
    # u=(1,1,1,0..), v=(1,1,-1,0..). u.v = 1+1-1 = 1.
    # u=(1,1,1,0..), v=(1,1,1,0..). u.v=3.
    # u=(1,1,1,0..), v=(1,1,0,1..). u.v = 1+1+0+0 = 2.
    # So conflict if u.v >= 2.
    # We want Max Independent Set on graph of 1320 nodes where edge exists if u.v >= 2.
    
    # Can we mix norm^2=2 and norm^2=3?
    # Max norm^2 = 3.
    # Dist(u, v) for u(norm2), v(norm3):
    # 2+3 - 2 u.v >= 3 => 2 u.v <= 2 => u.v <= 1.
    # Dist(u, w) for u(norm2), w(norm2):
    # u.v <= 1 (always true for distinct).
    # Dist(v, w) for v(norm3), w(norm3):
    # v.w <= 1.
    
    # So we can mix them if:
    # 1. Norm2 vectors are distinct (always ok).
    # 2. Norm3 vectors have pairwise dot <= 1.
    # 3. Any Norm2 vector has dot <= 1 with any selected Norm3 vector.
    
    # Let's start with the 220 Norm2 vectors. They are all mutually compatible.
    # Can we add Norm3 vectors?
    # A Norm3 vector v must have dot <= 1 with ALL 220 Norm2 vectors.
    # Let v have support S (size 3) and signs.
    # For any Norm2 vector u with support T (size 2):
    # If T is subset of S, u.v is sum of 2 products.
    # If u has same signs as v on T, u.v = 2. Conflict!
    # So v cannot have any Norm2 vector in its support with same signs.
    # Since we include ALL Norm2 vectors, for any pair of coordinates in v's support, there is a Norm2 vector with those signs.
    # Thus, v conflicts with that Norm2 vector.
    # So we cannot add ANY Norm3 vector if we keep all 220 Norm2 vectors.
    
    # So we must drop some Norm2 vectors to add Norm3 vectors.
    # This is a weighted MIS problem or just MIS on a combined graph.
    
    # Let's build the graph for Norm3 vectors only first, find a large clique (independent set in conflict graph),
    # then see if we can add Norm2 vectors.
    # Or just run SA on the combined pool of 220 + 1320 = 1540 vectors.
    
    # Vector Pool
    vectors = []
    # Add Norm2 vectors
    for support in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            v[support[0]] = signs[0]
            v[support[1]] = signs[1]
            vectors.append(v)
            
    # Add Norm3 vectors
    for support in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(support, signs):
                v[pos] = s
            vectors.append(v)
            
    n = len(vectors)
    vec_arr = np.array(vectors, dtype=np.int64)
    norms_sq = np.sum(vec_arr ** 2, axis=1)
    # Norms are 2 or 3.
    # Max norm in any valid set will be 3 if any norm3 vector is present, else 2.
    # To simplify, let's assume we aim for a set where max norm is 3.
    # Then constraint is: dist^2 >= 3.
    # For u,v in set:
    # if norm(u)=2, norm(v)=2: dist^2 = 4 - 2u.v >= 3 => u.v <= 0.5 => u.v <= 0.
    #   Wait, earlier I said u.v <= 1 for norm2-norm2.
    #   If max norm is 3, then dist >= sqrt(3). dist^2 >= 3.
    #   If both are norm2: 4 - 2u.v >= 3 => 2u.v <= 1 => u.v <= 0.5 => u.v <= 0.
    #   So if we mix, Norm2 vectors must be ORTHOGONAL.
    # if norm(u)=2, norm(v)=3: 5 - 2u.v >= 3 => 2u.v <= 2 => u.v <= 1.
    # if norm(u)=3, norm(v)=3: 6 - 2u.v >= 3 => 2u.v <= 3 => u.v <= 1.5 => u.v <= 1.
    
    # This mixed constraint is complex.
    # Alternative: Just use Norm3 vectors.
    # Constraint: u.v <= 1 for all pairs.
    # This is a standard MIS problem.
    
    # Let's try to find a large independent set in the Norm3 conflict graph.
    # Conflict if u.v >= 2.
    
    # Precompute dot products for Norm3 vectors only?
    # 1320 vectors. 1320x1320 matrix is fine.
    
    norm3_indices = range(220, n)
    norm3_vecs = vec_arr[norm3_indices]
    n3 = len(norm3_vecs)
    
    dot3 = norm3_vecs @ norm3_vecs.T
    conflict3 = dot3 >= 2
    
    # Simulated Annealing for MIS on conflict3
    # Initial solution: Greedy
    selected = set()
    remaining = list(range(n3))
    random.shuffle(remaining)
    for i in remaining:
        if all(not conflict3[i, j] for j in selected):
            selected.add(i)
            
    best_set = set(selected)
    best_size = len(selected)
    
    # SA
    T = 1.0
    cool_rate = 0.999
    min_T = 0.01
    rng = random.Random(42)
    
    while T > min_T:
        if time.time() > deadline - 5.0:
            break
            
        # Pick a random vertex to remove and a random vertex to add?
        # Or swap?
        # Let's try 1-move: add one, remove one?
        # Or just try to add a random vertex, and if conflict, remove one of the conflicting ones.
        
        if selected:
            u = rng.choice(list(selected))
            # Try to replace u with some v not in selected
            # Find v that conflicts with u? No, v must be compatible with (selected - {u})
            # and compatible with u? No, u is removed.
            
            # Simpler: Random walk.
            # Pick random u in selected. Remove it.
            # Pick random v not in selected. Add it if compatible with new selected.
            
            candidate_v = rng.randint(0, n3 - 1)
            if candidate_v in selected:
                continue
                
            # Check if candidate_v is compatible with selected - {u}
            temp_selected = set(selected)
            temp_selected.remove(u)
            
            if all(not conflict3[candidate_v, w] for w in temp_selected):
                new_selected = temp_selected | {candidate_v}
                # Size is same. We need to increase size?
                # SA usually minimizes energy. Here we want to maximize size.
                # If size is same, we accept with prob exp((0)/T) = 1?
                # This is just random walk.
                # We should try to INCREASE size.
                
                # Better move:
                # 1. Try to add a vertex that is compatible with all selected.
                # 2. If none, try to swap: remove u, add v, if v is compatible with rest.
                #    This keeps size same.
                # 3. To increase size, we need to find a vertex compatible with all.
                
                # Let's stick to a greedy + local search that tries to add.
                pass
            
        # This SA loop is getting messy. Let's use a simpler Local Search.
        
        # Restart local search from best_set
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try to add any vertex
            for v in range(n3):
                if v in selected:
                    continue
                if all(not conflict3[v, u] for u in selected):
                    selected.add(v)
                    improved = True
                    break
                    
            if not improved:
                # Try 1-swap to enable future additions?
                # Or just break.
                break
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            
        T *= cool_rate
        
    # Map back to original indices
    final_indices = [220 + i for i in best_set]
    
    # Check if we can add any Norm2 vectors
    # If we have Norm3 vectors, max norm is 3.
    # Norm2 vector u must have u.v <= 1 for all v in best_set (Norm3).
    # And if we add multiple Norm2 vectors, they must be orthogonal (u.w <= 0).
    
    # Let's try to add Norm2 vectors greedily
    norm2_indices = range(220)
    norm2_vecs = vec_arr[norm2_indices]
    n2 = len(norm2_vecs)
    
    # Precompute dot products between Norm2 and Norm3
    # dot23[i, j] = norm2_vecs[i] . norm3_vecs[j]
    # n2=220, n3=1320. 220*1320 = 290k. Fine.
    dot23 = norm2_vecs @ norm3_vecs.T
    
    # For each Norm2 vector, check compatibility with current best_set (Norm3)
    compatible_norm2 = []
    for i in range(n2):
        # Check against all selected Norm3 vectors
        ok = True
        for j in best_set:
            if dot23[i, j] > 1:
                ok = False
                break
        if ok:
            compatible_norm2.append(i)
            
    # Now select a set of compatible Norm2 vectors that are mutually orthogonal
    # Conflict between Norm2 i, k: dot22[i,k] != 0
    # dot22 = norm2_vecs @ norm2_vecs.T
    
    # Greedy selection of orthogonal Norm2 vectors
    selected_norm2 = []
    for i in compatible_norm2:
        # Check against selected_norm2
        if all(norm2_vecs[i] @ norm2_vecs[k] == 0 for k in selected_norm2):
            selected_norm2.append(i)
            
    # Final set
    final_indices = final_indices + [i for i in selected_norm2]
    
    result_vectors = vec_arr[final_indices]
    
    # Final Validation
    if len(result_vectors) > 0:
        res_arr = result_vectors.astype(np.float64)
        dists = np.linalg.norm(res_arr[:, np.newaxis, :] - res_arr[np.newaxis, :, :], axis=2)
        norms = np.linalg.norm(res_arr, axis=1)
        max_norm = np.max(norms)
        # Exclude diagonal
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        # Due to floating point, allow small epsilon
        if min_dist + 1e-9 < max_norm:
            # Fallback to just Norm2 vectors if mixed failed validation
            # Though logic above should be safe.
            # If it fails, return empty or just norm2?
            # Let's return the Norm2 vectors which are always valid (k=220)
            # Actually, if min_dist < max_norm, it's invalid.
            # The Norm2 vectors alone have min_dist sqrt(2), max_norm sqrt(2). Valid.
            # Let's return the first 220 vectors (all Norm2) as a safe fallback.
            return vec_arr[:220].astype(np.int64)
            
    return result_vectors.astype(np.int64)
