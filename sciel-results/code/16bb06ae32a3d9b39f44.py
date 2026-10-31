"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Strategy:
    # Use vectors with norm squared = 4 (Norm = 2).
    # The constraint is min pairwise dist >= max norm.
    # If all vectors have norm 2, we need dist^2 >= 4.
    # dist^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v.
    # 8 - 2 u.v >= 4 => u.v <= 2.
    
    # Type A: Weight 1 vectors in {-2, 0, 2}^11.
    # Norm^2 = 4.
    # Count: 11 positions * 2 signs = 22.
    # Mutual compatibility:
    # Same pos, diff sign: u.v = -4 <= 2.
    # Diff pos: u.v = 0 <= 2.
    # All 22 are compatible.
    
    # Type B: Weight 2 vectors in {-1, 0, 1}^11.
    # Norm^2 = 2.
    # Wait, if we mix Norm 2 and Norm sqrt(2), max norm is 2.
    # Constraint: dist^2 >= 4.
    # For u (Norm 2) and v (Norm sqrt(2)):
    # dist^2 = 4 + 2 - 2 u.v = 6 - 2 u.v.
    # 6 - 2 u.v >= 4 => 2 u.v <= 2 => u.v <= 1.
    
    # Let's check Type B vs Type B (both Norm sqrt(2)):
    # dist^2 = 2 + 2 - 2 u.v = 4 - 2 u.v.
    # 4 - 2 u.v >= 4 => -2 u.v >= 0 => u.v <= 0.
    
    # So Type B vectors must have dot product <= 0 with each other.
    # And Type B vs Type A must have dot product <= 1.
    
    # Let's re-evaluate Type B mutual compatibility.
    # u = (1, 1, 0...), v = (1, -1, 0...). u.v = 0. OK.
    # u = (1, 1, 0...), v = (1, 0, 1...). u.v = 1. FAIL (1 > 0).
    # So Type B vectors are NOT mutually compatible if they share a coordinate with same sign.
    
    # Therefore, we cannot just take ALL Type B vectors.
    # We need an independent set for Type B with constraint u.v <= 0.
    # And we need to check Type B vs Type A.
    
    # Let's simplify.
    # Just take Type A (22 vectors).
    # And try to find a large set of Type B vectors compatible with each other and Type A.
    
    # Actually, let's look at Type C: Weight 4 in {-1, 0, 1}^11.
    # We established that W4 vs W4 requires u.v <= 2.
    # W4 vs W1 (Type A) requires u.v <= 2. (Always true for W4, max dot 2).
    # W4 vs W2 (Type B) requires u.v <= 1.
    
    # Given the complexity of mixing, let's stick to the safest high-yield set.
    # The inspiration 2 got ~193 W4 vectors.
    # If we take those 193 W4 vectors, we can add ALL 22 W1 vectors.
    # Total ~215.
    
    # Can we do better?
    # What if we take NO W4, but a large set of W2?
    # W2 vs W2 requires u.v <= 0.
    # This is equivalent to finding a set of edges in K11 such that no two edges share a vertex with the same "sign orientation"?
    # Actually, u.v <= 0 for W2 vectors means they cannot share a coordinate with the same sign.
    # This is like a matching in a signed graph.
    # Max number of W2 vectors with u.v <= 0?
    # We can take all vectors with support {i, j} and signs (+, -) or (-, +)?
    # u=(1, -1, 0..), v=(1, 0, -1..). u.v = 1. Fail.
    
    # Let's just use the W4 + W1 strategy, which is proven to work and yields ~215.
    # We will implement the fast MIS for W4.
    
    # Generate W4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix for W4
    # Conflict if u.v > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Greedy Min-Degree MIS
    alive = np.ones(N4, dtype=bool)
    selected_indices = []
    degrees = conflicts.sum(axis=1).astype(np.int32)
    rng = np.random.default_rng(42)
    
    while np.any(alive) and time.time() < deadline - 10:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]
        
        selected_indices.append(chosen)
        
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        alive[to_remove] = False
        
        rem = np.where(alive)[0]
        if len(rem) > 0:
            degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
            
    result_v4 = vectors_v4[selected_indices]
    
    # Combine with W1
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
