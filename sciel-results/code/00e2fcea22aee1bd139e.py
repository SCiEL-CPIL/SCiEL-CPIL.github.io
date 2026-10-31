"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate a large candidate pool of integer vectors.
    # We include vectors with small integer coefficients to increase the chance
    # of finding dense kissing configurations.
    # The condition is: min_pairwise_distance >= max_norm.
    # Let R = max_norm. Then for all i != j, ||v_i - v_j|| >= R.
    # This is equivalent to: ||v_i - v_j||^2 >= R^2.
    # Note that ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i^T v_j.
    # This constraint is not just about angles; it depends on magnitudes too.
    # However, if we restrict to vectors of equal norm, it simplifies to angle constraints.
    # But integer vectors rarely have equal norms.
    # Let's stick to the general definition: 
    # A set S is valid if min_{u,v in S, u!=v} ||u-v|| >= max_{w in S} ||w||.
    
    # Strategy:
    # 1. Generate candidates.
    # 2. Compute all pairwise distances? No, too slow for large N.
    # 3. Use a greedy approach:
    #    - Sort candidates by some heuristic (e.g., norm, or random).
    #    - Iterate through candidates.
    #    - For each candidate, check if it conflicts with any already selected vector.
    #    - Conflict condition: ||v_new - v_sel||^2 < max_norm_current^2?
    #    Wait, the condition is global: min_dist >= max_norm.
    #    If we add a new vector, the max_norm might increase.
    #    This makes the constraint dynamic.
    
    # Alternative Interpretation/Strategy:
    # Often "kissing number" problems imply equal radius spheres, which implies equal norm vectors if centered at origin.
    # The prompt says: "Normalizing each vector to norm 2 certifies a kissing configuration".
    # This implies the final vectors should ideally be normalized.
    # But the output is integer vectors.
    # "minimum pairwise distance is at least the maximum vector norm".
    # Let M = max(||v_i||).
    # We need ||v_i - v_j|| >= M for all i != j.
    
    # If we pick vectors all with the same norm R, then M=R.
    # Condition becomes ||v_i - v_j|| >= R.
    # ||v_i - v_j||^2 = 2R^2 - 2 v_i^T v_j >= R^2
    # R^2 >= 2 v_i^T v_j
    # v_i^T v_j <= R^2 / 2.
    # This is a standard angular constraint.
    
    # Let's try to find a large set of integer vectors with the SAME norm.
    # In D=11, finding many integer vectors of the same norm is hard.
    # Example: norm^2 = 2 (vectors with two 1s).
    # Example: norm^2 = 3 (vectors with three 1s).
    
    # Let's generate all vectors with norm^2 <= K for some small K.
    # And group them by norm^2.
    # For each group, we need to select an independent set where v_i^T v_j <= norm^2 / 2.
    
    # Actually, the prompt allows different norms.
    # But mixing norms complicates the "min_dist >= max_norm" check.
    # If we have a vector with very large norm, it forces all other vectors to be far away.
    # It is usually better to stick to vectors of similar, small norms.
    
    # Let's focus on vectors with norm^2 = 2 (weight 2, entries +/-1) and norm^2 = 3 (weight 3, entries +/-1).
    # And maybe norm^2 = 4 (weight 4 or one 2).
    
    # Let's generate candidates with small weights.
    # Weight 1: 11 vectors (and signs? 22). Norm^2=1.
    # Weight 2: C(11,2)*4 = 55*4 = 220 vectors. Norm^2=2.
    # Weight 3: C(11,3)*8 = 165*8 = 1320 vectors. Norm^2=3.
    # Weight 4: C(11,4)*16 = 330*16 = 5280 vectors. Norm^2=4.
    
    # Total candidates ~ 7000.
    # Pairwise distance matrix for 7000 vectors is 49M entries.
    # 49M * 8 bytes (float64) = 392 MB. This is feasible.
    # Or we can do it in chunks or use int32 if possible, but distances are squared ints.
    # Max dist^2 for weight 4 vectors:
    # v1=(1,1,1,1,0...), v2=(1,1,1,0,1,0...). Diff=(0,0,0,1,-1,0...). Norm^2=2.
    # Max diff can be larger.
    # If v1 has 1s at i,j and v2 has 1s at k,l (disjoint), diff has four non-zeros. Norm^2=4.
    # If signs differ, e.g., v1=(1,1...), v2=(-1,-1...), diff=(2,2...), Norm^2=8.
    # So squared distances are small integers. We can use int32 or even int16.
    
    # Algorithm:
    # 1. Generate all integer vectors with L1 norm <= 4 (or similar).
    # 2. Compute the squared distance matrix D^2.
    # 3. We want to select a subset S such that min_{i!=j in S} D^2_{ij} >= max_{k in S} ||v_k||^2.
    #    Let R^2 = max_{k in S} ||v_k||^2.
    #    Then we need D^2_{ij} >= R^2 for all i!=j in S.
    
    # This is a "maximum clique" problem on a graph where edges exist if D^2_{ij} >= R^2?
    # But R^2 depends on the set S.
    # If we fix R^2, we look for the largest clique in the graph G_{R^2} where edge (i,j) exists if D^2_{ij} >= R^2 AND max(norm_i^2, norm_j^2) <= R^2?
    # No, the condition is global.
    # If we pick a set S, let M = max_{v in S} ||v||^2.
    # Then all pairs in S must have D^2 >= M.
    # This implies that for any pair (i,j) in S, D^2_{ij} >= max(||v_i||^2, ||v_j||^2).
    
    # So, we can define a compatibility graph G where edge (i,j) exists if D^2_{ij} >= max(||v_i||^2, ||v_j||^2).
    # Then ANY clique in this graph is a valid kissing configuration!
    # Why?
    # If S is a clique, then for all i!=j in S, D^2_{ij} >= max(||v_i||^2, ||v_j||^2).
    # Let M = max_{k in S} ||v_k||^2.
    # For any i,j in S, max(||v_i||^2, ||v_j||^2) <= M.
    # So D^2_{ij} >= max(...) is NOT necessarily >= M.
    # Wait.
    # If D^2_{ij} >= max(||v_i||^2, ||v_j||^2), does it imply D^2_{ij} >= M?
    # No. Example: v1 norm^2=1, v2 norm^2=1, v3 norm^2=100.
    # Edge (1,2) requires D^2_{12} >= 1.
    # Edge (1,3) requires D^2_{13} >= 100.
    # Edge (2,3) requires D^2_{23} >= 100.
    # If S={1,2,3} is a clique, then D^2_{13} >= 100 and D^2_{23} >= 100.
    # And D^2_{12} >= 1.
    # The global max norm is 100.
    # The min pairwise distance is min(D^2_{12}, D^2_{13}, D^2_{23}).
    # We need min(D^2) >= 100.
    # But D^2_{12} might be 1. So the condition fails.
    
    # So a clique in that graph is NOT sufficient.
    # We need: min_{i!=j in S} D^2_{ij} >= max_{k in S} ||v_k||^2.
    # Let M(S) = max_{k in S} ||v_k||^2.
    # We need all pairwise distances in S to be at least M(S).
    
    # Greedy Approach:
    # 1. Sort candidates by norm descending.
    # 2. Initialize S = empty, M = 0.
    # 3. Iterate through candidates v:
    #    - If ||v||^2 > M, then M becomes ||v||^2 (if we add it).
    #    - But adding v might force M to increase, which might invalidate previously added vectors if their distances to new vectors are small.
    #    - Actually, if we add a vector with larger norm, the threshold for ALL pairs increases.
    #    - This is tricky.
    
    # Better Greedy:
    # 1. Sort candidates by norm ASCENDING.
    # 2. Maintain a set S.
    # 3. Current Max Norm M = max_{v in S} ||v||^2.
    # 4. For each candidate v (in ascending norm order):
    #    - If ||v||^2 <= M:
    #       - Check if D^2_{v,u} >= M for all u in S.
    #       - If yes, add v to S.
    #    - If ||v||^2 > M:
    #       - If we add v, new M' = ||v||^2.
    #       - We need to check if D^2_{v,u} >= M' for all u in S.
    #       - AND we need to check if all pairs in S still satisfy D^2 >= M'?
    #       - No, the pairs in S were valid for M. If M' > M, they might now be invalid.
    #       - So we can only add v if D^2_{v,u} >= ||v||^2 for all u in S, AND all existing pairs in S have D^2 >= ||v||^2.
    #       - This is expensive to check for all pairs in S.
    
    # Simplified Strategy:
    # Restrict to vectors of EQUAL norm.
    # If all vectors have norm^2 = R^2, then M = R^2.
    # Condition: D^2_{ij} >= R^2 for all i!=j.
    # This is a standard independent set / clique problem on a graph where edge (i,j) exists if D^2_{ij} < R^2 (conflict graph).
    # We want the maximum independent set in the conflict graph.
    # Or maximum clique in the compatibility graph (edge if D^2 >= R^2).
    
    # Let's generate candidates for a specific R^2, e.g., R^2=2.
    # Vectors with exactly two non-zero entries of value +/-1.
    # Count: C(11,2) * 4 = 220.
    # This is the seed. Score 1.0.
    
    # Let's try R^2=3.
    # Vectors with exactly three non-zero entries of value +/-1.
    # Count: C(11,3) * 8 = 1320.
    # Let's build the conflict graph for these 1320 vectors.
    # Edge (i,j) if D^2_{ij} < 3.
    # D^2_{ij} = ||v_i - v_j||^2.
    # Since ||v_i||^2 = ||v_j||^2 = 3, D^2_{ij} = 6 - 2 v_i^T v_j.
    # We need D^2_{ij} >= 3 => 6 - 2 v_i^T v_j >= 3 => 2 v_i^T v_j <= 3 => v_i^T v_j <= 1.5 => v_i^T v_j <= 1.
    # So conflict if v_i^T v_j >= 2.
    # v_i^T v_j is the dot product.
    # For weight 3 vectors, dot product can be -3, -1, 1, 3 (if supports overlap in 3, 1, 1, 3 positions with matching signs).
    # Actually, if supports are disjoint, dot=0.
    # If 1 overlap: +/-1.
    # If 2 overlaps: +/-2.
    # If 3 overlaps: +/-3.
    # So conflict if dot >= 2.
    # This means if they share 2 or 3 positions with signs such that dot >= 2.
    # Specifically:
    # - 3 overlaps: dot is 3 or -3. Conflict if 3. (Same sign on all 3).
    # - 2 overlaps: dot is 2 or -2 or 0 (if one sign diff). Conflict if 2. (Same sign on 2).
    # - 1 overlap: dot 1 or -1. No conflict.
    # - 0 overlap: dot 0. No conflict.
    
    # So we want a set of weight-3 vectors such that no two vectors have dot product >= 2.
    # This is a combinatorial design problem.
    
    # Let's use a greedy algorithm on this conflict graph.
    # 1. Generate all 1320 vectors.
    # 2. Compute dot products? Or just check conflicts on the fly.
    # 3. Sort vectors randomly or by some heuristic.
    # 4. Greedy selection:
    #    - Maintain a list of selected vectors.
    #    - For each candidate, check dot product with all selected.
    #    - If max dot product < 2, add it.
    
    # To speed this up, we can vectorize the dot product check.
    # Let S be the matrix of selected vectors (k x 11).
    # Let v be the new candidate (11,).
    # Dots = S @ v (k,).
    # If max(Dots) < 2, add v.
    
    # We can do this for multiple R^2 values and pick the best.
    
    candidates = []
    
    # Helper to generate vectors with specific weight and signs
    def gen_vectors(weight, D):
        vecs = []
        for indices in itertools.combinations(range(D), weight):
            for signs in itertools.product([1, -1], repeat=weight):
                v = [0] * D
                for idx, s in zip(indices, signs):
                    v[idx] = s
                vecs.append(v)
        return np.array(vecs, dtype=np.int8)
    
    best_set = []
    best_score = 0
    
    # Try weight 2 (norm^2=2)
    # Try weight 3 (norm^2=3)
    # Try weight 4 (norm^2=4)
    
    for weight in [2, 3, 4]:
        if time.time() > deadline:
            break
            
        print(f"Trying weight {weight}")
        vecs = gen_vectors(weight, D)
        N = len(vecs)
        
        # Randomize order to avoid bias
        np.random.shuffle(vecs)
        
        selected = []
        selected_matrix = np.empty((0, D), dtype=np.int8)
        
        for i in range(N):
            if time.time() > deadline:
                break
            v = vecs[i]
            if len(selected_matrix) == 0:
                selected_matrix = v.reshape(1, -1)
                continue
                
            # Check dot products
            dots = selected_matrix @ v
            # Condition for validity: D^2 >= R^2
            # R^2 = weight (since entries are +/-1)
            # D^2 = 2*R^2 - 2*dot
            # 2*weight - 2*dot >= weight
            # weight >= 2*dot
            # dot <= weight / 2
            max_allowed_dot = weight // 2
            
            if np.max(dots) <= max_allowed_dot:
                selected_matrix = np.vstack([selected_matrix, v.reshape(1, -1)])
        
        k = len(selected_matrix)
        if k > best_score:
            best_score = k
            best_set = selected_matrix.copy()
            
    if len(best_set) == 0:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return best_set.astype(np.int64)

# EVOLVE-BLOCK-END
