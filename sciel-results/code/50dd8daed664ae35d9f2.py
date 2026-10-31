D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def valid_solution(vecs):
        if len(vecs) == 0:
            return True
        if len(vecs) < 2:
            return all(np.dot(v, v) > 0 for v in vecs)
        norms_sq = np.array([np.dot(v, v) for v in vecs])
        max_norm = np.max(norms_sq)
        # Compute pairwise distances squared
        vecs_arr = np.array(vecs)
        # Use broadcasting for pairwise squared distances
        # dist^2 = ||v||^2 + ||w||^2 - 2 v.w
        dots = vecs_arr @ vecs_arr.T
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        # Check all off-diagonal entries >= max_norm
        np.fill_diagonal(dist_sq, 1e18)
        return np.min(dist_sq) >= max_norm - 1e-9

    # Generate E8 minimal vectors (240 in 8D)
    def gen_e8():
        vecs = []
        # Type 1: (±1, ±1, 0, ..., 0)
        for i, j in combinations(range(8), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0]*8
                    v[i] = si
                    v[j] = sj
                    vecs.append(v)
        # Type 2: (±1/2, ±1/2, ..., ±1/2) with even number of minus signs
        # But we want integer vectors. The E8 lattice points include half-integers.
        # The task requires integer vectors.
        # The minimal vectors of E8 in the standard integer basis are the 240 type 1 vectors.
        # The other 120 are half-integer. Since we need integer vectors, we only use the 240.
        return np.array(vecs, dtype=np.int64)

    e8_vecs = gen_e8()
    
    # We embed E8 in R^11 by padding with 3 zeros.
    e8_11 = np.zeros((240, 11), dtype=np.int64)
    e8_11[:, :8] = e8_vecs
    
    # Now we want to add vectors in the last 3 dimensions (and potentially mix).
    # Let's look at vectors of the form (e8_vec, z_vec) where z_vec in Z^3.
    # The norm squared is ||e8||^2 + ||z||^2 = 8 + ||z||^2.
    # The distance squared between (e8_i, z_i) and (e8_j, z_j) is ||e8_i-e8_j||^2 + ||z_i-z_j||^2.
    # We need this >= max(8+||z_i||^2, 8+||z_j||^2).
    
    # Let's restrict ourselves to a set of "center" vectors in Z^3 and add them to all E8 vectors?
    # No, that would create too many conflicts.
    # Instead, let's consider the set of all integer vectors in R^11 with small norm.
    # The seed uses weight-2 vectors (norm^2=2).
    # The E8 vectors have norm^2=8.
    # If we mix weight-2 and norm^2=8 vectors, the distance must be >= 8.
    # A weight-2 vector is like (1,1,0...).
    # An E8 vector is like (1,1,0...).
    # Distance between (1,1,0...) and (1,-1,0...) is sqrt(4) = 2 < sqrt(8). Conflict.
    # So we cannot simply combine seed vectors with E8 vectors.
    
    # Let's focus on finding a large independent set in the graph of "good" vectors.
    # Candidates:
    # 1. All 240 E8 vectors (embedded).
    # 2. Integer vectors in Z^3 (padded with 0 in first 8 dims) with norm^2 <= 8.
    #    Let's call these Z3_vecs.
    #    Distance between E8 (a,0) and Z3 (0,b) is sqrt(8 + ||b||^2).
    #    We need sqrt(8+||b||^2) >= max(8, 8+||b||^2)?? No.
    #    Norm of E8 is sqrt(8). Norm of Z3 is sqrt(||b||^2).
    #    If we use E8 and Z3 vectors, max norm is max(sqrt(8), sqrt(||b||^2)).
    #    If ||b||^2 <= 8, max norm is sqrt(8).
    #    Distance is sqrt(8 + ||b||^2).
    #    Is sqrt(8+||b||^2) >= sqrt(8)? Yes, since ||b||^2 >= 0.
    #    So ANY E8 vector is compatible with ANY Z3 vector (with norm^2 <= 8).
    #    
    #    Now, what about two Z3 vectors b1, b2?
    #    Distance is ||b1 - b2||.
    #    We need ||b1 - b2||^2 >= max(||b1||^2, ||b2||^2).
    #    Let's find the maximum independent set of Z3 vectors with norm^2 <= 8 under this constraint.
    #
    #    What about two E8 vectors?
    #    Distance ||a1 - a2||.
    #    We need ||a1 - a2||^2 >= 8.
    #    In E8, minimal distance is sqrt(2). So ||a1-a2||^2 = 2 or 4.
    #    So E8 vectors CONFLICT with each other if we require distance >= sqrt(8).
    #    Wait. The condition is min pairwise distance >= max norm.
    #    If we take a set S of E8 vectors, their norms are all sqrt(8).
    #    So we need pairwise distance >= sqrt(8).
    #    But in E8, the minimum distance between distinct minimal vectors is sqrt(2).
    #    So we can take at most 1 E8 vector!
    #    
    #    This changes everything. The "kissing number" logic usually applies to spheres centered at the origin, but here we are choosing the centers themselves?
    #    "Construct nonzero integer vectors ... whose minimum pairwise distance is at least the maximum vector norm."
    #    Yes, the vectors are the centers.
    #    So if I pick two E8 vectors, their distance is ~1.41, but their norm is 2.82. 1.41 < 2.82. Invalid.
    #    
    #    So we cannot use many E8 vectors together.
    #    We can use many weight-2 vectors (norm sqrt(2)).
    #    Distance between (1,1,0..) and (1,-1,0..) is sqrt(4)=2.
    #    2 >= sqrt(2). Valid.
    #    Distance between (1,1,0..) and (-1,1,0..) is sqrt(4)=2. Valid.
    #    Distance between (1,1,0..) and (0,1,1..) is sqrt(1+0+1)=sqrt(2).
    #    sqrt(2) >= sqrt(2). Valid.
    #    
    #    So the seed solution (weight-2 vectors) is valid because their mutual distances are >= 2 (mostly) or sqrt(2) (some).
    #    Wait, if dist is sqrt(2) and norm is sqrt(2), it's valid.
    #    
    #    Can we add E8 vectors to the seed solution?
    #    Norm of seed vector: sqrt(2).
    #    Norm of E8 vector: sqrt(8).
    #    Max norm: sqrt(8).
    #    We need all pairwise distances >= sqrt(8).
    #    Seed vectors have mutual distances as small as sqrt(2).
    #    So we cannot mix them.
    #    
    #    We must choose a "scale".
    #    Option A: Use only weight-2 vectors (norm^2=2). Need dist^2 >= 2.
    #    Option B: Use only weight-3 vectors? (norm^2=3). Need dist^2 >= 3.
    #    Option C: Use E8 vectors (norm^2=8). Need dist^2 >= 8. (Max 1 vector).
    #    Option D: Mix different weights, but max norm determines the threshold.
    
    # The seed found 246 vectors with norm^2=2.
    # Let's check if we can find more vectors with norm^2=2.
    # Total weight-2 vectors: C(11,2)*4 = 55*4 = 220.
    # Wait. The seed code:
    # for i, j in combinations(range(D), 2):
    #   for si, sj in (1,-1)x(1,-1):
    #     v[i]=si, v[j]=sj
    # This generates 110 * 4 = 440 vectors.
    # But the score was 1.0, and the seed program returned them all.
    # Are they all valid?
    # Let's check validity of the seed output.
    # If v=(1,1,0..) and w=(1,-1,0..), dist^2 = 0 + 4 + 0 = 4. 4 >= 2. OK.
    # If v=(1,1,0..) and w=(0,1,1..), dist^2 = 1 + 0 + 1 = 2. 2 >= 2. OK.
    # If v=(1,1,0..) and w=(1,0,1..), dist^2 = 0 + 1 + 1 = 2. 2 >= 2. OK.
    # If v=(1,1,0..) and w=(1,1,0..), dist=0. Not distinct.
    # So all 440 weight-2 vectors are mutually at distance >= sqrt(2).
    # Why did the seed score 1.0? Maybe the score is just k/seed_k.
    # If seed returns 440, and I return 440, score is 1.0.
    # Can I get more than 440?
    
    # What if I include weight-1 vectors (unit vectors)?
    # Norm^2 = 1.
    # If I mix weight-1 and weight-2, max norm^2 = 2.
    # Need dist^2 >= 2.
    # Unit vector e1 = (1,0,0..).
    # Weight-2 vector v = (1,1,0..). Dist^2 = 0 + 1 = 1 < 2. Conflict.
    # v = (0,1,1..). Dist^2 = 1 + 0 + 0 = 1 < 2. Conflict.
    # v = (2,0,0..)? Not weight 2.
    # So unit vectors conflict with many weight-2 vectors.
    # Specifically, e_i conflicts with any v that has v_i != 0 and dist^2 < 2.
    # If v has v_i=1, dist^2 = ||v||^2 - 1 = 2 - 1 = 1. Conflict.
    # If v has v_i=-1, dist^2 = 2 - 1 = 1. Conflict.
    # If v has v_i=0, dist^2 = 2. OK.
    # So e_i is compatible with v iff v_i = 0.
    # 
    # This looks like a packing problem.
    # The seed solution of 440 might be optimal for weight-2.
    # Let's try to add more vectors of higher weight, but we must drop conflicting weight-2 vectors.
    # Or, use a different set entirely.
    
    # Let's try weight-3 vectors.
    # Norm^2 = 3.
    # Number of weight-3 vectors: C(11,3)*2^3 = 165 * 8 = 1320.
    # We need pairwise dist^2 >= 3.
    # 
    # Let's try to find a large independent set in the weight-3 graph.
    # Or mix weight-2 and weight-3?
    # If max norm^2 = 3.
    # Need dist^2 >= 3.
    # Weight-2 vectors have dist^2 between 2 and 4 (with each other).
    # Some pairs have dist^2=2 < 3. So we must remove those from the weight-2 set.
    # Weight-2 and Weight-3:
    # v (wt 2), w (wt 3).
    # Dist^2 = ||v||^2 + ||w||^2 - 2 v.w = 5 - 2 v.w.
    # We need 5 - 2 v.w >= 3 => 2 v.w <= 2 => v.w <= 1.
    # v.w can be -3, -2, -1, 0, 1, 2, 3?
    # Max v.w is 2 (if they share 2 coordinates with same sign).
    # If v.w = 2, dist^2 = 1 < 3. Conflict.
    # If v.w = 1, dist^2 = 3 >= 3. OK.
    # If v.w = 0, dist^2 = 5 >= 3. OK.
    
    # This is getting complex. Let's stick to a single weight class first.
    # The seed uses weight 2. Score 1.0 means 440 vectors.
    # Let's generate all weight-3 vectors and find MIS.
    # 1320 nodes.
    
    # Let's also consider weight 1.
    # 22 unit vectors.
    # Norm^2=1. Dist^2 >= 1.
    # All unit vectors are dist 2 apart (orthogonal) or 0 (same).
    # So all 22 unit vectors form a valid set. Score 22/440 = 0.05. Bad.
    
    # Let's try to improve on weight 2.
    # Is 440 maximal?
    # The graph of weight-2 vectors in R^11.
    # Nodes: 440.
    # Edges: dist^2 < 2.
    # Dist^2 is integer. So dist^2 = 1?
    # Can two weight-2 vectors have dist^2 = 1?
    # v=(1,1,0..), w=(1,0,1..). Dist^2 = 0 + 1 + 1 = 2.
    # v=(1,1,0..), w=(0,1,0..)? w is weight 1.
    # Two weight-2 vectors:
    # Same support: (1,1) vs (1,-1) -> dist^2=4. (1,1) vs (-1,1) -> dist^2=4.
    # Overlap 1: (1,1,0) vs (1,0,1) -> dist^2=2.
    # Disjoint: (1,1,0) vs (0,0,1,1) -> dist^2=4.
    # So minimum dist^2 is 2.
    # Condition: dist^2 >= max_norm^2 = 2.
    # So ALL weight-2 vectors are valid!
    # The seed program returns all of them.
    # So 440 is the count for weight 2.
    
    # Can we get > 440?
    # We need vectors with norm^2 > 2, but we must filter out conflicts.
    # Or, we can use a mix, but it's hard.
    
    # Let's try weight 3 vectors (norm^2=3).
    # We need dist^2 >= 3.
    # Min dist^2 for weight 3:
    # v=(1,1,1), w=(1,1,-1). Dist^2 = 0+0+4 = 4.
    # v=(1,1,1), w=(1,0,1). w is weight 2.
    # v=(1,1,1), w=(0,1,1). Dist^2 = 1+0+0 = 1? No.
    # v=(1,1,1) (wt 3), w=(1,1,1) (same).
    # v=(1,1,1), w=(1,1,-1) (wt 3). Dist^2 = 4.
    # v=(1,1,1), w=(1,0,1) (wt 2). Dist^2 = 0+1+0 = 1.
    # If we only use weight 3 vectors:
    # v, w both wt 3.
    # Same support: (1,1,1) vs (1,1,-1) -> dist^2=4.
    # Overlap 2: (1,1,1) vs (1,1,0,1)? No, support size 3.
    # v=(1,1,1,0), w=(1,1,-1,0). Dist^2=4.
    # v=(1,1,1,0), w=(1,0,1,1). Overlap {1,3}.
    # v-w = (0,1,0,-1). Dist^2 = 2.
    # So dist^2 can be 2.
    # We need dist^2 >= 3.
    # So we must exclude pairs with dist^2=2.
    # This is an MIS problem on 1320 nodes.
    
    # Let's implement a greedy MIS for weight-3 vectors.
    
    def gen_weight_vectors(weight):
        vecs = []
        for comb in combinations(range(D), weight):
            for signs in product((1, -1), repeat=weight):
                v = [0]*D
                for idx, s in zip(comb, signs):
                    v[idx] = s
                vecs.append(v)
        return np.array(vecs, dtype=np.int64)

    from itertools import product

    # Try weight 3
    wt3_vecs = gen_weight_vectors(3)
    n3 = len(wt3_vecs)
    
    # Build conflict graph for weight 3 (dist^2 < 3)
    # dist^2 can be 1, 2, 4...
    # Conflicts if dist^2 < 3, i.e., 1 or 2.
    # Since wt3 vectors have integer coords, dist^2 is integer.
    # dist^2 = ||v||^2 + ||w||^2 - 2v.w = 6 - 2v.w.
    # v.w can be -3..3.
    # dist^2 = 6 - 2v.w.
    # If v.w=3, dist^2=0 (same vector).
    # If v.w=2, dist^2=2. Conflict.
    # If v.w=1, dist^2=4. OK.
    # So conflict iff v.w >= 2.
    
    # Compute dot products
    dots3 = wt3_vecs @ wt3_vecs.T
    # Conflict if dots3[i][j] >= 2 and i != j
    # Note: dots3[i][i] = 3.
    
    # Greedy MIS
    # Sort by degree (number of conflicts)
    # Degree i = sum(dots3[i] >= 2) - 1 (exclude self)
    
    degrees3 = np.sum(dots3 >= 2, axis=1) - 1
    order = np.argsort(degrees3)
    
    selected_idx = []
    removed = np.zeros(n3, dtype=bool)
    
    for i in order:
        if removed[i]:
            continue
        selected_idx.append(i)
        # Remove neighbors
        neighbors = np.where(dots3[i] >= 2)[0]
        removed[neighbors] = True
        if elapsed() > deadline - 10:
            break

    wt3_selected = wt3_vecs[selected_idx]
    
    # Check validity
    if not valid_solution(wt3_selected):
        # Fallback to seed
        vecs = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vecs.append(v)
        return np.array(vecs, dtype=np.int64)
    
    # Can we do better?
    # Maybe weight 4?
    # Norm^2 = 4.
    # Need dist^2 >= 4.
    # v.w conflict if dist^2 < 4.
    # dist^2 = 8 - 2v.w.
    # Conflict if 8 - 2v.w < 4 => 2v.w > 4 => v.w > 2.
    # So v.w >= 3 is conflict.
    # 
    # Let's try weight 4.
    wt4_vecs = gen_weight_vectors(4)
    n4 = len(wt4_vecs)
    
    dots4 = wt4_vecs @ wt4_vecs.T
    # Conflict if dots4[i][j] >= 3 (and i!=j)
    # Self dot is 4.
    
    degrees4 = np.sum(dots4 >= 3, axis=1) - 1
    order4 = np.argsort(degrees4)
    
    selected_idx4 = []
    removed4 = np.zeros(n4, dtype=bool)
    
    for i in order4:
        if removed4[i]:
            continue
        selected_idx4.append(i)
        neighbors = np.where(dots4[i] >= 3)[0]
        removed4[neighbors] = True
        if elapsed() > deadline - 5:
            break

    wt4_selected = wt4_vecs[selected_idx4]
    
    # Compare sizes
    best_vecs = wt3_selected
    if len(wt4_selected) > len(best_vecs) and valid_solution(wt4_selected):
        best_vecs = wt4_selected
        
    # Also keep seed as fallback if something goes wrong or is larger (unlikely)
    seed_vecs = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                seed_vecs.append(v)
    seed_vecs = np.array(seed_vecs, dtype=np.int64)
    
    if len(seed_vecs) > len(best_vecs):
        best_vecs = seed_vecs
        
    return best_vecs

# EVOLVE-BLOCK-END
