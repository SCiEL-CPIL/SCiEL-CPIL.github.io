"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np


def _e8_minimal_vectors():
    """Return the 240 minimal vectors of E8 lattice (norm^2=2)."""
    vecs = []
    # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0)
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 8
                    v[i] = si
                    v[j] = sj
                    vecs.append(v)
    # Type 2: (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even number of -1s
    # To keep integers, we scale by 2: (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1) with even number of -1s
    # But wait, the norm of (1,1,...,1) is sqrt(8), not sqrt(2).
    # The minimal vectors of E8 are actually:
    # 1) (±1, ±1, 0, ..., 0) permutations: 8*7*2*2 = 224? No.
    #    Pairs: C(8,2)=28, signs: 4. 28*4=112.
    # 2) (±1/2, ..., ±1/2) with even number of minus signs.
    #    There are 2^7 = 128 such vectors.
    #    Total 112 + 128 = 240.
    # To represent as integers, we can scale everything by 2.
    # Then Type 1 becomes (±2, ±2, 0, ..., 0) with norm^2 = 8.
    # Type 2 becomes (±1, ±1, ..., ±1) with even number of -1s, norm^2 = 8.
    # This is a valid lattice (2*E8) but the kissing number is the same.
    # However, the task asks for "nonzero integer vectors". Scaling is fine.
    # But wait, the distance condition: min pairwise distance >= max norm.
    # If we scale by 2, norms become 2*sqrt(2) for type 1 and 2*sqrt(2) for type 2?
    # No, (±2, ±2, 0...) has norm sqrt(4+4)=sqrt(8).
    # (±1, ..., ±1) has norm sqrt(8).
    # So all norms are equal.
    # Distance between two vectors u, v: ||u-v||.
    # In E8, the minimal distance is sqrt(2).
    # Scaled by 2, minimal distance is 2*sqrt(2) = sqrt(8).
    # So min distance = max norm. This holds.
    
    # Let's generate the scaled E8 vectors.
    # Type 1: (±2, ±2, 0, ..., 0)
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 8
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vecs.append(v)
    
    # Type 2: (±1, ±1, ..., ±1) with even number of -1s.
    # Generate all 2^8 sign combinations, keep those with even number of -1s.
    for mask in range(256):
        # Count bits
        if bin(mask).count('1') % 2 == 0:
            v = []
            for i in range(8):
                if mask & (1 << i):
                    v.append(-1)
                else:
                    v.append(1)
            vecs.append(v)
    
    return np.array(vecs, dtype=np.int64)


def _d3_minimal_vectors():
    """Return the 12 minimal vectors of D3 lattice (norm^2=2).
    D3 in 3D: vectors with integer coords, sum of squares = 2.
    These are (±1, ±1, 0) permutations.
    C(3,2)=3 pairs, 4 sign combos. 3*4=12.
    """
    vecs = []
    for i in range(3):
        for j in range(i + 1, 3):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 3
                    v[i] = si
                    v[j] = sj
                    vecs.append(v)
    return np.array(vecs, dtype=np.int64)


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Get E8 minimal vectors (240)
    e8_vecs = _e8_minimal_vectors()
    
    # Get D3 minimal vectors (12)
    d3_vecs = _d3_minimal_vectors()
    
    # Construct the combined set in R^11
    # Option 1: E8 x 0^3 -> 240 vectors
    # Option 2: 0^8 x D3 -> 12 vectors
    # Option 3: (E8 + D3) direct sum? No, that's not a lattice product in the same way.
    # The knowledge says: "Using the direct sum lattice E8 x D3 instead of E8 x 0^3 adds 12 valid minimal vectors from the D3 component"
    # This implies we take the union of (E8 x 0) and (0 x D3)?
    # Let's check if the union satisfies the condition.
    # Vectors in E8 x 0 have norm sqrt(8) (scaled).
    # Vectors in 0 x D3 have norm sqrt(2) (unscaled D3) or sqrt(8) if we scale D3 by 2?
    # If we use unscaled D3, norms are different.
    # The condition is: min pairwise distance >= max norm.
    # If max norm is from E8 part (sqrt(8)), and min distance includes distances between E8 and D3 parts,
    # distance between u in E8x0 and v in 0xD3 is ||u|| + ||v||? No, ||u-v|| = sqrt(||u||^2 + ||v||^2).
    # If ||u||^2=8 and ||v||^2=2, distance is sqrt(10) ≈ 3.16.
    # Max norm is sqrt(8) ≈ 2.83.
    # sqrt(10) > sqrt(8), so this is fine.
    # What about distances within D3?
    # Minimal distance in D3 is sqrt(2).
    # But max norm of the whole set is sqrt(8).
    # sqrt(2) < sqrt(8). So the condition fails for pairs within the D3 part!
    
    # Therefore, we must scale the D3 part to have the same norm as the E8 part.
    # Scale D3 by 2: vectors become (±2, ±2, 0) etc. Norm^2 = 8.
    # Then minimal distance in scaled D3 is 2*sqrt(2) = sqrt(8).
    # So min distance within D3 part is sqrt(8), which equals max norm.
    # Distance between E8 and D3 parts: sqrt(8+8) = 4 > sqrt(8).
    # Distance within E8 part: sqrt(8).
    # So the union of (2*E8) x 0 and 0 x (2*D3) works.
    
    # Let's create scaled D3 vectors.
    d3_vecs_scaled = d3_vecs * 2
    
    # Build the 11-dimensional vectors
    # Part 1: E8 vectors in first 8 dims, 0 in last 3
    part1 = np.zeros((240, 11), dtype=np.int64)
    part1[:, :8] = e8_vecs
    
    # Part 2: 0 in first 8 dims, scaled D3 vectors in last 3
    part2 = np.zeros((12, 11), dtype=np.int64)
    part2[:, 8:] = d3_vecs_scaled
    
    # Combine
    all_vecs = np.vstack([part1, part2])
    
    # Verify the condition
    # Calculate all pairwise distances
    # This might be slow for 252 vectors, but 252*252/2 = ~31k pairs, which is fine.
    n = len(all_vecs)
    # Compute norms
    norms = np.linalg.norm(all_vecs, axis=1)
    max_norm = np.max(norms)
    
    # Compute pairwise distances
    # Use broadcasting
    diff = all_vecs[:, np.newaxis, :] - all_vecs[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Check condition
    if min_dist < max_norm - 1e-9:
        # Fallback to just E8 part
        all_vecs = part1
    
    # Time check
    if time.time() > deadline:
        return all_vecs
    
    return all_vecs
# EVOLVE-BLOCK-END
