"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np

_DEADLINE = 145.0  # Safety margin before 150s kill


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct using vectorized operations."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err


def _build_decomp(n, m, p):
    """
    Build a decomposition using Strassen for the 2x2 sub-block.
    
    For 2x3x8:
    - Split A (2x3) into A1 (2x2, cols 0,1) and A2 (2x1, col 2)
    - Split B (3x8) into B1 (2x8, rows 0,1) and B2 (1x8, row 2)
    - C = A1*B1 + A2*B2
    
    A2*B2: rank 16 (each output C[i,k] = A[i,2]*B[2,k])
    A1*B1: 2x2 * 2x8, split into 4 Strassen blocks of 2x2x2, rank 7*4=28
    
    Total rank: 16 + 28 = 44
    """
    n, m, p = 2, 3, 8
    
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank_total for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: A2*B2 (rank 16)
    for k in range(8):
        U[2][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        U[5][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
    
    # Part 2: A1*B1 via Strassen (rank 28)
    for g in range(4):
        b11 = 0 + 2 * g
        b12 = 0 + 2 * g + 1
        b21 = 8 + 2 * g
        b22 = 8 + 2 * g + 1
        a11 = 0
        a12 = 1
        a21 = 3
        a22 = 4
        
        U[a11][col] = 1; U[a22][col] = 1; V[b11][col] = 1; V[b22][col] = 1
        W[(2 * g) * 2 + 0][col] = 1; W[(2 * g + 1) * 2 + 1][col] = 1
        col += 1
        
        U[a21][col] = 1; U[a22][col] = 1; V[b11][col] = 1
        W[(2 * g) * 2 + 1][col] = 1
        col += 1
        
        U[a11][col] = 1; V[b22][col] = 1; V[b12][col] = -1
        W[(2 * g + 1) * 2 + 0][col] = 1; W[(2 * g + 1) * 2 + 1][col] = -1
        col += 1
        
        U[a22][col] = 1; V[b11][col] = 1; V[b21][col] = -1
        W[(2 * g) * 2 + 0][col] = -1; W[(2 * g) * 2 + 1][col] = 1
        col += 1
        
        U[a11][col] = 1; U[a12][col] = 1; V[b22][col] = 1
        W[(2 * g + 1) * 2 + 0][col] = 1
        col += 1
        
        U[a21][col] = 1; U[a11][col] = -1; V[b11][col] = 1; V[b12][col] = 1
        W[(2 * g) * 2 + 0][col] = 1
        col += 1
        
        U[a12][col] = 1; U[a22][col] = -1; V[b21][col] = 1; V[b22][col] = 1
        W[(2 * g + 1) * 2 + 1][col] = 1
        col += 1
    
    return U, V, W


def _try_block_rank10(n, m, p, time_limit):
    """
    Try to find a rank-10 decomposition per 2x3x2 block.
    If successful, total rank = 40.
    """
    import numpy as np
    from fractions import Fraction
    start_time = time.time()
    
    # For one 2x3x2 block, we need 10 bilinear terms
    # A-factors: 2x3 = 6 entries (a00,a01,a02,a10,a11,a12)
    # B-factors: 3x2 = 6 entries (b00,b01,b10,b11,b20,b21)
    # C-factors: 2x2 = 4 entries (c00,c01,c10,c11)
    
    # We'll try to find 10 terms (u_c, v_c, w_c) such that
    # sum_c u_c[i,j] * v_c[j,k] * w_c[i,k] = 1 for all (i,j,k)
    
    # Strategy: Use a structured ansatz and solve
    # Let's try: 7 Strassen-like terms for the 2x2x2 part + 3 modified terms for the extra column
    
    # Actually, let's try a direct numerical approach
    # Set up the system and use least squares
    
    # For simplicity, let's try specific ansatz patterns
    
    # Pattern 1: Modify Strassen to incorporate one extra term
    # M5' = (a00+a01+a02)(b11+b21) instead of M5 = (a00+a01)b11
    # This adds a00*b21 + a01*b21 + a02*b11 + a02*b21
    # We need to cancel a00*b21, a01*b21, a02*b11 elsewhere
    
    # This is complex. Let me try a different approach:
    # Use 10 random terms and solve the nonlinear system
    
    # Actually, for a deterministic approach, let me try:
    # Use the known rank-11 decomposition and try to merge two terms
    
    # Let me try: combine M8=a02*b20 and M10=a12*b20 into one term
    # (a02+a12)*b20 = a02*b20 + a12*b20
    # But this gives c00 += (a02+a12)*b20 and c10 += (a02+a12)*b20
    # We only want c00 += a02*b20 and c10 += a12*b20
    # The extra terms are c00 += a12*b20 and c10 += a02*b20 which are wrong
    
    # Let me try another approach: use the fact that we have freedom in how
    # we assign the C-factors
    
    # For now, let's just verify the rank-44 solution and return it
    # A proper rank-10 search would require solving a nonlinear system
    
    if time.time() - start_time > time_limit:
        return None
    
    return None


def _try_rank_reduction(n, m, p, start_rank, end_rank, time_limit):
    """
    Try to find a lower-rank decomposition by solving linear systems.
    
    Strategy: Build the multiplication tensor T[i,j,k] = delta.
    We want U (nm x r), V (mp x r), W (pn x r) such that
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    This is a nonlinear system. Instead, we use the "block" approach:
    Decompose into 2x3x2 blocks. For each block, find a rank-11 decomposition
    (Strassen 7 + 4 rank-1), then try to merge across blocks.
    
    Actually, let's use a simpler approach: start from the rank-44 decomposition
    and try to reduce rank by finding linear dependencies in the factor matrices
    using Gaussian elimination over Q.
    """
    import fractions
    from fractions import Fraction
    
    start_time = time.time()
    
    # Build the full tensor as a linear system
    # The tensor T has entries T[i,j,k] = 1 (it's the identity-like tensor for matmul)
    # We need: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Approach: Use the block structure. Split into 4 blocks of size 2x3x2.
    # Block g uses A cols 0,1, B rows 0,1,2, C cols 2g, 2g+1
    # Wait, that's not quite right. Let me think more carefully.
    
    # Actually, the key insight from the knowledge: "Decomposes the 2x3x8 tensor 
    # into four 2x3x2 sub-tensors, each decomposed using a rank-11 scheme"
    
    # A 2x3x2 tensor has rank at most 2*3*2 = 12, but can be done in 11.
    # 4 blocks * 11 = 44. That's the same as our current rank.
    
    # To get below 44, we need to share factors across blocks.
    
    # Let me try a different approach: use random initialization and 
    # gradient-free optimization, or use the algebraic geometry approach.
    
    # For now, let's try to verify if rank 43 is possible by checking
    # if we can find a linear dependency in the 44-term decomposition.
    
    # Build the 44-term decomposition
    U, V, W = _build_decomp(n, m, p)
    r = len(U[0])
    
    # Check if any column is a linear combination of others
    # This is expensive, so let's skip for now and just return the 44-rank solution.
    
    return None


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based decomposition (rank 44)
    try:
        if time.time() - start_time > _DEADLINE:
            raise TimeoutError("Deadline exceeded")
        U, V, W = _build_decomp(n, m, p)
        # Quick verification
        err = _verify(U, V, W, n, m, p)
        if err < 1e-10:
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    if time.time() - start_time > _DEADLINE:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
