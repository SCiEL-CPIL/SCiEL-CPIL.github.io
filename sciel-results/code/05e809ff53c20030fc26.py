"""Matrix multiplication 4x5x5 rank-minimal decomposition."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    start_time = time.time()
    deadline = start_time + 135
    
    # Build 4x4x4 Strassen decomposition (rank 7)
    # Strassen: C = A*B where A,B are 2x2
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4x4, we can use a 2x2 block structure where each block is 2x2
    # Actually, let's use the standard 7-term Strassen for 2x2 and extend
    
    # Simpler: use the known rank-7 decomposition for 4x4x4
    # The 4x4x4 tensor can be decomposed with rank 7 using Strassen's method
    # applied to 2x2 blocks, where each "entry" is a 2x2 matrix
    
    # Let me construct it explicitly for 4x4x4:
    # Indices: i (row of A/C, 0-3), j (col of A/row of B, 0-3), k (col of B/C, 0-3)
    
    # Use 2x2 block structure: split 4 into 2+2
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bj is 2x2
    # C = A*B = [[C11, C12], [C21, C22]] where each Ck is 2x2
    
    # Strassen for 2x2 blocks (each block is 2x2):
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Each Mi is a 2x2 matrix product of 2x2 matrices
    # So each Mi itself can be decomposed with rank 4 (schoolbook for 2x2x2)
    # But we can use Strassen recursively... actually for 2x2x2, rank is 7 (Strassen)
    # Wait, 2x2x2 has rank 7? No, the 2x2x2 matrix multiplication tensor has rank 7.
    
    # Actually, the standard result: R(2,2,2) = 7 (Strassen)
    # R(4,4,4) ≤ 7 * R(2,2,2) = 7*7 = 49? That's not helpful.
    
    # Let me think differently. The 4x4x4 tensor rank is known to be at most 21.
    # But I need an explicit construction.
    
    # Alternative: just use the schoolbook for 4x4x4 (rank 16) and add the cross terms.
    # Total: 16 + 4 + 4 + 1 = 25. Better than 100 but not great.
    
    # Let me try a different approach: use rank-1 terms but try to be clever.
    
    # Actually, let me just implement the schoolbook for the full 4x5x5 but see if I can
    # find any structure. The key constraint is that the answer must be EXACT.
    
    # For safety and correctness, let me use the schoolbook decomposition (rank 100)
    # but structure it cleanly. This guarantees correctness.
    
    rank_school = n * m * p  # 100
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify(U, V, W):
        raise ValueError("Schoolbook decomposition failed verification")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
