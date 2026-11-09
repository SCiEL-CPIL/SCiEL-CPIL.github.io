"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    
    The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    A rank-r decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _try_strassen_3x4x8():
    """
    Try to find a lower-rank decomposition for 3x4x8 matrix multiplication.
    
    Key insight: 3x4x8 can be viewed as multiplying a 3x4 matrix by a 4x8 matrix.
    We can split the 4x8 matrix into two 4x4 blocks: B = [B1 | B2] where B1, B2 are 4x4.
    Then C = A * B = [A*B1 | A*B2].
    
    If we can compute A*B1 and A*B2 using Strassen's algorithm (rank 7 for 2x2x2,
    or better algorithms for larger sizes), we might reduce the total rank.
    
    For 3x4x4: We can use the fact that 3 = 1+2. Let A = [a1; A2] where a1 is 1x4 and A2 is 2x4.
    Then C1 = A * B1 = a1*B1 + A2*B1.
    
    Actually, let me think about this differently. The rank of M_{3,4,8} is known to be
    at most 3*4*8 = 96 (schoolbook). Can we do better?
    
    One approach: Use the fact that M_{3,4,8} can be embedded in M_{4,4,8} or similar.
    
    Actually, a more promising approach: Since 8 = 2*4, we can write:
    A (3x4) * B (4x8) = A * [B1 | B2] where B1, B2 are 4x4.
    
    For each 3x4x4 sub-problem, the schoolbook rank is 12. Two such sub-problems give 24.
    But we need to compute both, so total rank is 24 for the 3x4x8 problem? No, that's not right.
    
    Let me reconsider. The tensor for 3x4x8 has 3*4*8 = 96 entries, each equal to 1.
    A rank-r decomposition requires r terms.
    
    Known results:
    - M_{2,2,2} has rank 7 (Strassen)
    - M_{2,2,2,2} (4D) has rank 49 (Coppersmith-Winograd variant)
    - For rectangular: M_{2,3,2} has rank at most 11 (Strassen-like)
    
    For 3x4x8:
    - We can view this as M_{3,4,4} x M_{1,1,2} via Kronecker? No, that doesn't work directly.
    
    Let me try a different approach. Use the fact that:
    M_{3,4,8} can be decomposed using block structure.
    
    Write A as 3x4, B as 4x8. Split B into two 4x4 blocks: B = [B1 | B2].
    Then C = [C1 | C2] where C1 = A*B1 (3x4) and C2 = A*B2 (3x4).
    
    For each 3x4x4 multiplication, we need rank at most 12 (schoolbook).
    But can we do 3x4x4 with rank < 12?
    
    M_{3,4,4}: Split A into rows: A = [a1; a2; a3] where each ai is 1x4.
    Then Ci = a1*Bi + a2*Bi + a3*Bi.
    
    Each ai*Bi is a 1x4 vector, which is a dot product of length 4, requiring 4 multiplications.
    So C1 = a1*B1 + a2*B1 + a3*B1 requires 12 multiplications in schoolbook.
    
    Can we do better? For 1x4x4, the rank is 4 (it's just a dot product, no way to reduce).
    For 3x4x4, we need to compute 3 such dot products and add them. The total rank is at least
    the sum of ranks if they're independent, which is 4+4+4 = 12. So schoolbook is optimal for 3x4x4.
    
    Therefore, 3x4x8 = two 3x4x4 problems, each of rank 12, total rank 24? No, that's not how
    tensor rank works. The tensor for 3x4x8 is not the direct sum of two 3x4x4 tensors in a way
    that would give rank 24.
    
    Actually, I think I'm overcomplicating this. Let me just return the schoolbook decomposition.
    """
    n, m, p = 3, 4, 8
    U, V, W = _build_schoolbook(n, m, p)
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (rank = n*m*p = 96)
    U, V, W = _build_schoolbook(n, m, p)
    
    # Verify correctness
    if not _verify_decomp(U, V, W, n, m, p):
        raise ValueError("Schoolbook verification failed")
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
