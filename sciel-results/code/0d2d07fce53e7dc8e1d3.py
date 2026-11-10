"""Exhaustive search with symmetry reduction for optimal ±1 polynomials (24 coeffs)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    N = 24
    HALF = 12
    sqrtN = math.sqrt(N)
    total = 1 << HALF  # 4096

    # Precompute all 2^12 sub-polynomial coefficient vectors
    S = np.zeros((total, HALF), dtype=np.float64)
    for i in range(total):
        for k in range(HALF):
            S[i, k] = 1.0 if (i >> k) & 1 else -1.0

    # Dense grid for DFT
    G = 8192
    angles = 2 * np.pi * np.arange(G) / G
    zs = np.exp(1j * angles)

    # Build V1[j,k] = z_j^k (k=0..11), V2[j,k] = z_j^(12+k)
    V1 = np.ones((G, HALF), dtype=np.complex128)
    for k in range(1, HALF):
        V1[:, k] = V1[:, k-1] * zs

    z12 = zs ** HALF
    V2 = np.ones((G, HALF), dtype=np.complex128)
    V2[:, 0] = z12
    for k in range(1, HALF):
        V2[:, k] = V2[:, k-1] * zs

    # M_A[j,i] = A_i(z_j); M_B[j,i] = B_i(z_j)
    M_A = V1 @ S.T   # (G, 4096) complex
    M_B = V2 @ S.T   # (G, 4096) complex

    def score_pair(i, j):
        val = M_A[:, i] + M_B[:, j]
        return np.max(np.abs(val)) / sqrtN

    def time_ok():
        return time.time() < deadline - 2.0

    # Symmetry reductions:
    # 1. Global sign flip: P(z) -> -P(z) doesn't change |P(z)|. 
    #    This means we can fix c_0 = +1 without loss of generality.
    #    In our (i, j) representation, c_0 is bit 0 of i.
    #    So we only need to consider i with bit 0 = 1 (i.e., i is odd).
    #    This halves the i-space: 2048 values.
    
    # 2. Time reversal: P(z) = sum c_k z^k. 
    #    Q(z) = z^23 P(1/z) = sum c_k z^(23-k) = sum c_{23-k} z^k.
    #    |Q(z)| = |z^23 P(1/z)| = |P(1/z)|. Since |z|=1, |1/z|=1, so sup|Q| = sup|P|.
    #    This maps (i, j) to (reversed_i, reversed_j) where reversal is on the 12-bit halves.
    #    Specifically, if P has coeffs c_0..c_23, Q has coeffs c_23..c_0.
    #    The first 12 coeffs of Q are c_23..c_12, which correspond to reversed j.
    #    The last 12 coeffs of Q are c_11..c_0, which correspond to reversed i.
    #    So (i, j) and (rev(j), rev(i)) have the same score.
    #    We can use this to break ties or reduce search, but it's trickier to implement 
    #    in a simple loop. Let's first just do the c_0=+1 reduction.

    # With c_0 = +1, i must be odd.
    # Total pairs: 2048 * 4096 = 8,388,608.
    # Each score_pair takes ~10-20 us. Total ~80-160 seconds.
    # This is tight. Let's optimize score_pair.

    # Vectorized scoring: we can compute scores for all i (odd) against a fixed j,
    # or all j against a fixed i.
    # Let's loop over i (odd) and vectorize over j.
    
    best_score = np.inf
    best_i = -1
    best_j = -1

    odd_ics = np.arange(1, total, 2)  # All odd i values, 2048 of them

    for idx, i in enumerate(odd_ics):
        if not time_ok():
            break
        # For this i, compute score for all j
        # val = M_A[:, i] + M_B[:, j] for all j
        # M_A[:, i] is (G,). M_B is (G, 4096).
        # We want max_j max_g |M_A[g,i] + M_B[g,j]|
        
        # This is hard to vectorize fully without memory blowup.
        # Let's just loop over j but use numpy for the max.
        # Actually, let's try a different vectorization:
        # For fixed i, the score for j is max_g |M_A[g,i] + M_B[g,j]|.
        # We can precompute M_A_col = M_A[:, i] (G,)
        # Then for each j, we need max_g |M_A_col + M_B[:, j]|.
        # This is still a loop over j.
        
        # Alternative: Precompute for each i the "profile" and use some trick?
        # No, let's just do the double loop but with optimized inner loop.
        
        MA_i = M_A[:, i]  # (G,) complex
        for j in range(total):
            if not time_ok():
                break
            # val = MA_i + M_B[:, j]
            # score = max(|val|) / sqrtN
            # To avoid allocating a new array each time, we can reuse a buffer
            # but numpy will handle it.
            val = MA_i + M_B[:, j]
            s = np.max(np.abs(val)) / sqrtN
            if s < best_score:
                best_score = s
                best_i = i
                best_j = j

    # Build final coefficients
    coeffs = np.zeros(N, dtype=np.float64)
    coeffs[:HALF] = S[best_i, :]
    coeffs[HALF:] = S[best_j, :]

    return coeffs.astype(int).tolist()
# EVOLVE-BLOCK-END
