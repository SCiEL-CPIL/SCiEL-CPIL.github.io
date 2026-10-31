"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    # The seed program builds all weight-2 vectors (norm 2): 440 vectors.
    # Each pair of weight-2 vectors has squared distance in {0, 2, 4, 8}.
    # We want a subset where minimum pairwise squared distance >= max squared norm = 4.
    # So we need: no two vectors share both coordinates (distance 0 impossible since distinct),
    # and no two vectors have distance^2 < 4, i.e., distance^2 in {0, 2}.
    # distance^2 = 2 means they share exactly one coordinate with the same sign.
    # So the constraint is: for any pair, they cannot share a coordinate with the same sign.
    
    # Actually let's think more carefully.
    # A weight-2 vector has two nonzero entries at positions i, j with values ±1, ±1.
    # Vector v = (si at i, sj at j), vector w = (ti at k, tj at l)
    # If {i,j} == {k,l}: same support. Then if si==ti and sj==tj, distance=0 (same vector).
    #   If si==ti, sj!=tj: distance^2 = 4 (the two differing entries). Wait: 
    #   v - w at position i: si - ti = 0, at position j: sj - tj = ±2. So distance^2 = 4. OK.
    #   If si!=ti, sj==tj: distance^2 = 4. OK.
    #   If si!=ti, sj!=tj: distance^2 = 8. OK.
    # If supports overlap in exactly one coordinate, say i==k, j!=l:
    #   v-w at i: si-ti, at j: sj, at l: -tj. distance^2 = (si-ti)^2 + sj^2 + tj^2
    #   = (si-ti)^2 + 1 + 1 = (si-ti)^2 + 2.
    #   If si==ti: distance^2 = 2. BAD.
    #   If si!=ti: distance^2 = 4. OK.
    # If supports are disjoint: distance^2 = 4. OK.
    
    # So the constraint is: no two vectors can share a coordinate with the same sign.
    # This is equivalent to: assign each vector to its pair (i,j) with signs (si,sj).
    # For each coordinate i, among all vectors using coordinate i, the signs must be all distinct? No:
    # We can have multiple vectors using coordinate i with sign +1, as long as they don't also
    # share the other coordinate with the same sign.
    
    # Actually, the constraint "no two vectors share a coordinate with the same sign" means:
    # For each coordinate i and sign s, at most one vector in our set uses (i, s).
    # Wait, that's too strong. Let me re-examine.
    # v uses (i, si) and (j, sj). w uses (k, tk) and (l, tl).
    # They conflict if i==k and si==tk (share coord i with same sign), OR j==l and sj==tl.
    # So: for each coordinate i and sign s, at most one vector can use position i with sign s.
    # That gives at most 11*2 = 22 vectors. That's worse than 440!
    
    # Wait, I think I'm overcomplicating. Let me re-read the constraint.
    # The task says: "minimum pairwise distance is at least the maximum vector norm."
    # All vectors have norm sqrt(2), so max norm = sqrt(2).
    # So min pairwise distance >= sqrt(2), i.e., min squared distance >= 2.
    # But distance^2 = 2 is allowed! So the constraint is min squared distance >= 2.
    # Wait, but the seed has 440 vectors. Let me check if the seed is valid.
    # In the seed, take v = (1 at 0, 1 at 1) and w = (1 at 0, 1 at 2).
    # They share coordinate 0 with sign +1. distance^2 = (1-1)^2 + (1-0)^2 + (0-1)^2 = 0 + 1 + 1 = 2.
    # So distance = sqrt(2) = max norm. This is valid!
    # So the constraint is min squared distance >= 2, which allows distance^2 = 2.
    
    # So the actual constraint is: min squared distance >= 2.
    # distance^2 = 0: same vector (not possible in a set of distinct vectors).
    # distance^2 = 2: share one coordinate with same sign. This IS allowed.
    # So ALL 440 weight-2 vectors are mutually compatible! The seed is already optimal for weight-2.
    
    # Now, the knowledge says: "Adding all 22 norm-4 weight-1 vectors to the weight-4 independent set increases the count by 22"
    # Wait, "weight-4 independent set"? That doesn't match. Let me re-read.
    # "Adding all 22 norm-4 weight-1 vectors to the weight-4 independent set increases the count by 22, as they are compatible with all weight-4 vectors and each other"
    # Hmm, "norm-4 weight-1 vectors" - a weight-1 vector with norm 4? That would be (4, 0, ..., 0).
    # And "weight-4 vectors" - vectors with 4 nonzero entries?
    
    # Actually I think the terminology is: "weight" = number of nonzero coordinates.
    # A weight-1 vector with norm 4: (±4, 0, ..., 0) - 22 such vectors.
    # A weight-4 vector: 4 nonzero coordinates, each ±1, so norm = sqrt(4) = 2.
    # Wait, but then the max norm would be 4, and we'd need min distance >= 4, i.e., min squared distance >= 16.
    
    # Let me reconsider. The task says "Normalizing each vector to norm 2 certifies a kissing configuration."
    # So all vectors should have the same norm after normalization. The constraint is:
    # min pairwise distance >= max vector norm.
    
    # If we mix vectors of different norms, the max norm matters.
    # If we have weight-2 vectors (norm sqrt(2)) and weight-1 vectors with value 4 (norm 4),
    # then max norm = 4, and we need min distance >= 4, i.e., min squared distance >= 16.
    # Distance between (4,0,...,0) and (1,1,0,...,0) = sqrt(9+1) = sqrt(10) < 4. Not compatible.
    
    # So maybe the "weight-4 vectors" in the knowledge refer to something else.
    # Let me think about what gives a good kissing number in dimension 11.
    
    # The kissing number in dimension 11 is known to be 282.
    # The seed gives 440, which seems too high. Let me double-check.
    
    # Actually wait - the seed has 440 vectors, and the score is 1.0. The seed score is k/seed_k.
    # If seed_k = 440, then score = k/440. The current program scores 1.0, meaning it returns 440 vectors.
    
    # But is 440 valid? All pairs have distance^2 >= 2, and max norm^2 = 2. So yes, valid.
    # The kissing number bound is for equal-radius spheres tangent to a central sphere.
    # The kissing number in R^11 is 282 (lower bound), upper bound is 639.
    # But 440 weight-2 vectors... let me verify the minimum distance more carefully.
    
    # Two weight-2 vectors v, w.
    # Case 1: Same support {i,j}. 
    #   v = (si, sj), w = (ti, tj) at positions (i,j).
    #   distance^2 = (si-ti)^2 + (sj-tj)^2.
    #   If si!=ti and sj!=tj: distance^2 = 4. OK.
    #   If si==ti and sj!=tj: distance^2 = 4. OK. Wait, no: (si-ti)^2 = 0, (sj-tj)^2 = 4. Total 4.
    #   Actually all sign combinations give distance^2 in {0, 4, 8}. 
    #   si==ti, sj==tj: distance^2 = 0. But these are the same vector! Not in the set twice.
    #   So distinct vectors with same support have distance^2 >= 4.
    # Case 2: Overlap in one coordinate.
    #   distance^2 = (si-ti)^2 + 1 + 1 = (si-ti)^2 + 2.
    #   If si==ti: distance^2 = 2. 
    #   If si!=ti: distance^2 = 4.
    # Case 3: Disjoint supports.
    #   distance^2 = 1+1+1+1 = 4.
    
    # So minimum distance^2 = 2 (when sharing one coordinate with same sign).
    # Max norm^2 = 2. So 2 >= 2. Valid!
    
    # So 440 is valid. Can we do better?
    
    # Let's add weight-1 vectors. A weight-1 vector e_i * a has norm |a|.
    # If we add (a, 0, ..., 0), its norm is |a|.
    # Distance to a weight-2 vector (1,1,0,...,0) at positions (0,1): 
    #   distance^2 = (a-1)^2 + 1.
    # We need distance^2 >= max_norm^2.
    # If max norm is still sqrt(2) (i.e., |a| <= sqrt(2)), then a must be integer, so a = 1 or 0.
    # a=1: norm = 1 < sqrt(2). Distance^2 to (1,1,0,...) = 0+1 = 1 < 2. NOT VALID.
    # So we can't add weight-1 vectors with small norm.
    
    # What if we use weight-1 vectors with a=2? Norm = 2 > sqrt(2).
    # Then max norm = 2, need min distance^2 >= 4.
    # But weight-2 vectors have distance^2 = 2 from each other. So we'd need to remove them.
    # That's a trade-off.
    
    # Alternative: use only weight-2 vectors but find a larger set?
    # We already have all 440. Can't add more weight-2 vectors.
    
    # What about weight-3 vectors? (1,1,1,0,...,0) has norm sqrt(3).
    # If we mix weight-2 and weight-3, max norm = sqrt(3), need min distance^2 >= 3.
    # Distance between two weight-3 vectors sharing 2 coordinates: distance^2 = 2. < 3. Bad.
    # So we can't have many weight-3 vectors.
    
    # Let me try a different approach: use a lattice-based construction.
    # The Leech lattice or similar might give better packing.
    # But for now, let me try to combine weight-2 vectors with some other vectors.
    
    # Actually, let me reconsider. The score is k, and the seed gives 440.
    # To beat the seed, I need k > 440.
    
    # Let me think about adding vectors that are compatible with all 440 weight-2 vectors.
    # A vector v is compatible with all weight-2 vectors if distance(v, w)^2 >= 2 for all w.
    # Actually, we need distance(v, w)^2 >= max_norm^2. If v has norm > sqrt(2), max_norm changes.
    
    # If v has norm exactly sqrt(2) (weight-2), it's already in the set.
    # If v has norm > sqrt(2), say norm = r > sqrt(2), then we need all pairwise distances >= r.
    # For a weight-2 vector w, distance(v, w)^2 >= r^2.
    
    # This is getting complex. Let me try a greedy approach:
    # Start with all 440 weight-2 vectors. Then try to add other vectors one by one,
    # checking compatibility.
    
    # But first, let me try a simpler improvement: can we use vectors with entries in {-1, 0, 1}
    # but with weight > 2?
    
    # Actually, let me just try to find a good set by search.
    
    # Strategy: 
    # 1. Start with all 440 weight-2 vectors.
    # 2. Try adding weight-1 vectors with value 1: (1,0,...,0). 
    #    Norm = 1. But max norm becomes sqrt(2) (from weight-2).
    #    Distance from (1,0,...,0) to (1,1,0,...,0) = sqrt(1) = 1 < sqrt(2). NOT COMPATIBLE.
    #    So can't add.
    
    # 3. Try a mixed approach: drop some weight-2 vectors to add more of other types.
    
    # Let me try: use weight-1 vectors with value 1, and weight-2 vectors, but with a constraint.
    # Actually this is a maximum independent set problem in a graph where edges connect incompatible pairs.
    
    # Given time constraints, let me try a heuristic: 
    # Generate all candidate vectors (weight 1, 2, 3 with entries in {-1,0,1} or small integers),
    # build a compatibility graph, and find a large clique.
    
    # But 11-dimensional with weight up to 3: 
    # weight 1: 22 vectors
    # weight 2: 440 vectors  
    # weight 3: C(11,3)*8 = 165*8 = 1320 vectors
    # Total: ~1782 vectors. Clique in 1782-node graph is hard.
    
    # Let me try a simpler approach: greedy with randomization.
    
    # Actually, let me just return the seed result for now and focus on correctness.
    # The key insight from the knowledge is about adding norm-4 weight-1 vectors.
    # Let me try: use weight-1 vectors with value 2 (norm 2), and weight-2 vectors with a constraint.
    
    # If max norm = 2, need min distance^2 >= 4.
    # Weight-2 vectors: min pairwise distance^2 = 2. So we can't have all 440.
    # We need a subset of weight-2 vectors with min distance^2 >= 4.
    # This means: no two vectors share a coordinate with the same sign.
    # As I analyzed: for each (coord, sign), at most one vector. 11*2 = 22 slots.
    # But each vector uses 2 slots. So at most 11 vectors from weight-2? No, that's not right either.
    
    # Actually, the constraint "no two vectors share a coordinate with the same sign" means:
    # For each coordinate i and sign s, at most one vector uses (i,s).
    # Total slots: 22. Each vector uses 2 slots. So at most 11 weight-2 vectors.
    # Plus 22 weight-1 vectors (one per (i, s)).
    # Total: 33. Much worse than 440.
    
    # So the weight-2 set of 440 is hard to beat with simple constructions.
    
    # Let me just return the seed for now.
    
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    
    if time.time() > deadline - 5:
        return np.array(vectors, dtype=np.int64)
    
    # Try to add more vectors
    # Generate candidate weight-1 vectors with value 1
    candidates = []
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            candidates.append(v)
    
    # Check if any candidate is compatible with all existing vectors
    # distance^2 >= 2 (since max norm^2 = 2)
    existing = np.array(vectors, dtype=np.int64)
    norms_sq = np.sum(existing * existing, axis=1)
    max_norm_sq = np.max(norms_sq)  # = 2
    
    for cand in candidates:
        cand_arr = np.array(cand, dtype=np.int64)
        dists_sq = np.sum((existing - cand_arr) ** 2, axis=1)
        if np.min(dists_sq) >= max_norm_sq:
            vectors.append(cand)
    
    result = np.array(vectors, dtype=np.int64)
    return result
# EVOLVE-BLOCK-END
