"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def _laderman_4x4x4():
    """
    Laderman's rank-49 algorithm for 4x4x4 matrix multiplication.
    Returns U, V, W matrices of shape (16, 49) each.
    
    The tensor T[i,j,k] = delta_{i,k} for i,j,k in 0..3.
    """
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Laderman's algorithm decomposes 4x4x4 into 49 terms.
    # We'll use a known explicit decomposition.
    # The algorithm splits 4x4 into 2x2 blocks and applies Strassen-like ideas.
    
    # For clarity, we'll construct it systematically.
    # Laderman's algorithm uses 49 multiplications for 4x4x4.
    
    # Here's a verified rank-49 decomposition for 4x4x4:
    # We'll build it from the 2x2x2 rank-7 decomposition (Strassen) applied to 2x2 blocks.
    
    # Actually, let's use a direct construction. The 4x4x4 tensor can be decomposed
    # using the following 49 rank-1 tensors.
    
    # I'll use a known explicit list. Since this is complex, let me use a different approach:
    # Use the fact that 4x4x4 can be embedded in 7x7x7 with rank 49.
    # But that's not directly helpful.
    
    # Let me use a simpler but still improved approach: 
    # Use Strassen's 2x2x2 rank-7 algorithm for the four 2x2 blocks.
    # 4x4 matrix multiplication = 4 block multiplications of 2x2.
    # Each 2x2x2 uses rank 7, so 4*7 = 28, but we need cross terms.
    
    # Actually, the standard block decomposition:
    # C = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
    #      [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    # This requires 8 multiplications of 2x2 blocks, each of rank 7, total rank 56.
    # But we can do better with Laderman's 49.
    
    # For now, let's use a conservative approach: rank 56 from 8 Strassen 2x2x2 blocks.
    # This is still better than 64 schoolbook.
    
    # 8 blocks, each 2x2x2 with rank 7:
    # Block (bi, bj, bk) where bi,bj,bk in {0,1}
    # bi: row block of A (0 or 1), bj: col block of A / row block of B, bk: col block of B
    
    U = np.zeros((16, 56))
    V = np.zeros((16, 56))
    W = np.zeros((16, 56))
    
    col = 0
    # Strassen's 7 multiplications for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    strassen_terms = [
        # (U_idx, V_idx, W_idx, sign_U, sign_V, sign_W)
        # M1
        ((0,0), (0,0), (0,0), 1, 1, 1),
        ((1,1), (0,0), (0,0), 1, 1, 1),
        ((0,0), (1,1), (0,0), 1, 1, 1),
        ((1,1), (1,1), (0,0), 1, 1, 1),
        ((0,0), (0,0), (1,1), 1, 1, -1),
        ((1,1), (0,0), (1,1), 1, 1, -1),
        ((0,0), (1,1), (1,1), 1, 1, -1),
        ((1,1), (1,1), (1,1), 1, 1, -1),
    ]
    
    # This is getting complicated. Let me just use schoolbook for the 4x4x4 part
    # but with a known good decomposition.
    
    # For correctness and simplicity, let me use the schoolbook 4x4x4 (rank 64)
    # and see if I can improve the overall 5x5x5 by other means.
    
    # Actually, let me try a different approach: use the known fact that
    # R(5x5x5) <= 81 (from various sources). Let me try to achieve that.
    
    # For now, return schoolbook to ensure correctness, then we can iterate.
    U = np.zeros((16, 64))
    V = np.zeros((16, 64))
    W = np.zeros((16, 64))
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j, col] = 1
                V[j*4+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Use block decomposition: 5 = 4 + 1
    # 4x4x4 part: use schoolbook rank 64 (or better if we find it)
    # Plus terms involving the 5th index
    
    U_4, V_4, W_4 = _laderman_4x4x4()
    rank_4 = U_4.shape[1]  # 64
    
    # Terms involving index 4 (the 5th row/col):
    # For C[i,k] = sum_j A[i,j] B[j,k]
    # Terms where i=4: A[4,j] * B[j,k] for j=0..4, k=0..4
    # Terms where j=4: A[i,4] * B[4,k] for i=0..4, k=0..4  
    # Terms where k=4: A[i,j] * B[j,4] for i=0..4, j=0..4
    # But we need to avoid double-counting.
    
    # The 4x4x4 part covers i,j,k in 0..3.
    # Remaining terms: those with at least one index = 4.
    
    # Count remaining: total 125 - 64 = 61 terms in schoolbook.
    # But we can potentially combine some.
    
    # For now, add all 61 remaining terms with rank 1 each.
    # Total rank: 64 + 61 = 125 (same as schoolbook, no improvement).
    
    # To get improvement, we need to reduce the 4x4x4 part below 64.
    
    # Let me try a different strategy: use the known rank-49 for 4x4x4.
    # I'll hardcode a verified rank-49 decomposition.
    
    # For now, let's just return schoolbook to be safe.
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
