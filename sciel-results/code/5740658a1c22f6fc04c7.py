"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # First, try to find a rank-25 decomposition through systematic search
    # starting from the rank-26 base and attempting to merge terms.
    
    # The tensor T[i,j,k] = 1 if k == i*p + j (C[i,j] gets A[i,k]*B[k,j])
    # U has shape (nm, rank), row index = i*m + k
    # V has shape (mp, rank), row index = k*p + j  
    # W has shape (pn, rank), row index = i*p + j
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = i * p + j
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Start with the known rank-26 decomposition
    rank26 = _build_rank_26()
    
    # Try to reduce to rank 25 by finding a linear dependency among the 26 terms
    # or by finding a different rank-25 construction
    
    # Strategy: Use the block structure more cleverly
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Let's try a different approach: use Strassen for the 2x2 block but
    # share the Strassen intermediate products more aggressively with the
    # cross-block terms.
    
    # Alternative: Try rank 25 directly with a structured ansatz
    # Based on the block structure, we can try:
    # - 7 terms for A11*B11 (Strassen)
    # - 4 terms for A12*B21 (2x1 * 1x2 = outer product, rank 2, but we need 2)
    #   Actually A12 is 2x1, B21 is 1x2, so A12*B21 is rank 1!
    #   C11 += A12*B21 can be done in rank 1
    # - Similarly for other cross terms
    
    # Wait, let me reconsider the ranks:
    # A11*B11: 2x2 * 2x2, Strassen gives rank 7
    # A12*B21: 2x1 * 1x2 = (a1,b1)^T * (c1,d1) = [[a1*c1, a1*d1], [b1*c1, b1*d1]]
    #   This is rank 1 as a tensor! So it takes 1 term.
    # A11*B12: 2x2 * 2x1 = 2x1 vector, needs 2 terms (one for each row)
    #   Actually: [a11*b12_1 + a12*b22_1, a21*b12_1 + a22*b22_1]
    #   This is 2 scalar multiplications, rank 2
    # A12*B22: 2x1 * 1x1 = 2x1 vector, rank 2 (2 scalar mults)
    #   Actually: [a12_1*b22, a12_2*b22], rank 2
    # A21*B11: 1x2 * 2x2 = 1x2 vector, rank 2
    # A22*B21: 1x1 * 1x2 = 1x2 vector, rank 2
    # A21*B12: 1x2 * 2x1 = 1x1 scalar, rank 1
    # A22*B22: 1x1 * 1x1 = 1x1 scalar, rank 1
    
    # Total: 7 + 1 + 2 + 2 + 2 + 2 + 1 + 1 = 18? That can't be right.
    # The issue is that we're double-counting. The Strassen terms for A11*B11
    # already produce contributions to C11, and the cross terms add to C11, C12, C21, C22.
    
    # Let me just try to verify if rank 25 is achievable by attempting a search.
    
    # For now, return the rank-26 solution which is known to work
    return rank26

def _build_rank_26():
    """Build the known rank-26 decomposition."""
    n, m, p = 3, 3, 3
    nm, mp, pn = n * m, m * p, p * n
    rank = 26
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    
    # 1. A2 B2 (Top-left 2x2 * 2x2) -> Rank 7 (Strassen)
    # U indices: 0, 1, 3, 4 (A00, A01, A10, A11)
    # V indices: 0, 1, 3, 4 (B00, B01, B10, B11)
    # W indices: 0, 1, 3, 4 (C00, C01, C10, C11)
    
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. A12 * B21 (2x1 * 1x2) -> This is rank 1!
    # A12 = [A02, A12]^T (rows 2, 5), B21 = [B20, B21] (cols 6, 7)
    # C11 += A12*B21: C00 += A02*B20, C01 += A02*B21, C10 += A12*B20, C11 += A12*B21
    # This is 4 terms if done directly, but as a tensor it's rank 1:
    # U[2][col]=1, U[5][col]=1, V[6][col]=1, V[7][col]=1
    # W[0][col]=1, W[1][col]=1, W[3][col]=1, W[4][col]=1
    # Wait, that's not rank 1. Let me think again.
    # A12*B21 = [[A02*B20, A02*B21], [A12*B20, A12*B21]]
    # This is (A02, A12)^T * (B20, B21) which is indeed rank 1 as a matrix,
    # but as a tensor term, we need:
    # Term: U has A02 and A12, V has B20 and B21, W has C00,C01,C10,C11
    # But a single tensor term can only multiply one A entry, one B entry, one C entry.
    # So A12*B21 actually requires 4 separate terms (one for each product).
    # Unless we use the structure: (A02, A12) is a vector, (B20, B21) is a vector,
    # and their outer product gives the 2x2 block. But in tensor decomposition,
    # each term is a single rank-1 tensor, so we need 4 terms for the 4 entries.
    
    # Actually, I was wrong. Let me recount. The current seed uses:
    # U[2][col] = 1; V[6][col] = 1; W[0][col] = 1  # A02*B20 -> C00
    # U[2][col] = 1; V[7][col] = 1; W[1][col] = 1  # A02*B21 -> C01
    # U[5][col] = 1; V[6][col] = 1; W[3][col] = 1  # A12*B20 -> C10
    # U[5][col] = 1; V[7][col] = 1; W[4][col] = 1  # A12*B21 -> C11
    # That's 4 terms.
    
    # Hmm, so the total is still 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26.
    
    # Let me just return the seed's construction
    U[2][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
    U[2][col] = 1; V[7][col] = 1; W[1][col] = 1; col += 1
    U[5][col] = 1; V[6][col] = 1; W[3][col] = 1; col += 1
    U[5][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1
    
    # 3. A11 * B12 (2x2 * 2x1) -> Rank 4
    # A00*B02 -> C02, A01*B12 -> C02, A10*B02 -> C12, A11*B12 -> C12
    U[0][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
    U[1][col] = 1; V[5][col] = 1; W[6][col] = 1; col += 1
    U[3][col] = 1; V[2][col] = 1; W[7][col] = 1; col += 1
    U[4][col] = 1; V[5][col] = 1; W[7][col] = 1; col += 1
    
    # 4. A12 * B22 (2x1 * 1x1) -> Rank 2
    U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
    U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
    
    # 5. A21 * B11 (1x2 * 2x2) -> Rank 4
    U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
    U[7][col] = 1; V[3][col] = 1; W[2][col] = 1; col += 1
    U[6][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
    U[7][col] = 1; V[4][col] = 1; W[5][col] = 1; col += 1
    
    # 6. A22 * B21 (1x1 * 1x2) -> Rank 2
    U[8][col] = 1; V[6][col] = 1; W[2][col] = 1; col += 1
    U[8][col] = 1; V[7][col] = 1; W[5][col] = 1; col += 1
    
    # 7. A21 * B12 (1x2 * 2x1) -> Rank 2
    U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
    U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
    
    # 8. A22 * B22 (1x1 * 1x1) -> Rank 1
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
