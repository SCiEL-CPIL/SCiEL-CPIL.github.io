"""Rewritten program for matrix multiplication 2x5x6 tensor decomposition."""
import time
import itertools

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists for the schoolbook decomposition."""
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication.
    
    The tensor T[i, j, k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    For matrix multiplication, T[i, j, k] = 1 if the (i,j) entry of A and (j,k) entry of B
    contribute to (i,k) entry of C. Actually the matrix multiplication tensor has T[i,j,k]=1
    for all valid i,j,k (it's the "all ones" tensor in the appropriate space).
    
    Wait - let me reconsider. The matrix multiplication tensor for multiplying an n x m matrix
    by an m x p matrix has components T[i, j, k] where i in [0,n), j in [0,m), k in [0,p).
    The decomposition is: C[i,k] = sum_j A[i,j] * B[j,k].
    So the tensor T[i,j,k] = 1 for all i,j,k (it's just the identity map on the middle index).
    
    Actually, the standard tensor for matrix multiplication is defined such that:
    (A*B)[i,k] = sum_j A[i,j] B[j,k]
    The tensor T has T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p).
    
    The decomposition satisfies: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_2x2_block():
    """Return Strassen's 7-term decomposition for 2x2 by 2x2 multiplication.
    
    Returns (U, V, W) where U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    U[i*2+j, c] corresponds to A[i,j], V[j*2+k, c] to B[j,k], W[k*2+i, c] to C[i,k].
    
    Strassen's 7 products:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 - M4 + M5 - M7
    C01 = M3 + M4
    C10 = M2 + M5
    C11 = M1 - M2 + M3 - M6
    """
    # 7 factors
    rank = 7
    U = [[0] * rank for _ in range(4)]  # 2x2 matrix: indices 0,1,2,3 for (0,0),(0,1),(1,0),(1,1)
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Factor 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C00 += M1, C11 += M1
    W[0][0] = 1; W[3][0] = 1
    
    # Factor 1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C10 += M2, C11 -= M2
    W[2][1] = 1; W[3][1] = -1
    
    # Factor 2: M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    # C01 += M3, C11 += M3
    W[1][2] = 1; W[3][2] = 1
    
    # Factor 3: M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    # C00 -= M4, C01 += M4
    W[0][3] = -1; W[1][3] = 1
    
    # Factor 4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C00 += M5, C10 += M5
    W[0][4] = 1; W[2][4] = 1
    
    # Factor 5: M6 = (A10-A00)(B00+B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    # C11 -= M6
    W[3][5] = -1
    
    # Factor 6: M7 = (A01-A11)(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    # C00 -= M7
    W[0][6] = -1
    
    return U, V, W

def _embed_2x2_2x2_into_larger(U_block, V_block, W_block, i0, j0, k0, n, m, p, U_full, V_full, W_full, offset):
    """Embed a 2x2x2 Strassen block into the larger decomposition.
    
    i0, j0, k0 are the starting indices in the n, m, p dimensions.
    offset is the starting column index in the full factor matrices.
    """
    for c in range(7):
        for di in range(2):
            for dj in range(2):
                ui = U_block[di * 2 + dj][c]
                if ui != 0:
                    U_full[(i0 + di) * m + (j0 + dj)][offset + c] = ui
        for dj in range(2):
            for dk in range(2):
                vj = V_block[dj * 2 + dk][c]
                if vj != 0:
                    V_full[(j0 + dj) * p + (k0 + dk)][offset + c] = vj
        for dk in range(2):
            for di in range(2):
                wk = W_block[dk * 2 + di][c]
                if wk != 0:
                    W_full[(k0 + dk) * n + (i0 + di)][offset + c] = wk

def _hybrid_strassen(n, m, p):
    """Build a hybrid decomposition using Strassen for 2x2x2 blocks where possible,
    and schoolbook for the rest.
    
    For 2x5x6:
    - We can partition the 2x5 matrix into one 2x2 block and one 2x3 block.
    - We can partition the 5x6 matrix into one 2x6 block and one 3x6 block.
    
    Strategy: Use Strassen for the 2x2x2 sub-block (i in [0,2), j in [0,2), k in [0,2)),
    and schoolbook for everything else.
    
    Actually, let's think more carefully. The matrix multiplication is:
    C[2x6] = A[2x5] * B[5x6]
    
    We can decompose this by partitioning the middle dimension (j):
    C = A[:, 0:2] * B[0:2, :] + A[:, 2:5] * B[2:5, :]
    
    For the first term: A1 is 2x2, B1 is 2x6. We can further partition B1 into B1[:, 0:2] (2x2) and B1[:, 2:6] (2x4).
    So A1 * B1[:, 0:2] is a 2x2x2 multiplication -> use Strassen (rank 7 instead of 8).
    A1 * B1[:, 2:6] is a 2x2x4 multiplication -> schoolbook rank 8.
    
    For the second term: A2 is 2x3, B2 is 3x6. Schoolbook rank 36.
    
    Total: 7 + 8 + 36 = 51.
    
    Alternatively, partition differently:
    C = A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :] + A[:, 4:5] * B[4:5, :]
    Term 1: 2x2x6 -> partition B into 2x2 and 2x4: Strassen(7) + schoolbook(8) = 15
    Term 2: 2x2x6 -> same: 15
    Term 3: 2x1x6 -> schoolbook: 12
    Total: 15 + 15 + 12 = 42.
    
    Even better:
    C = A[:, 0:2] * B[0:2, :] + A[:, 2:5] * B[2:5, :]
    Term 1: 2x2x6. Partition: A is 2x2, B is 2x6.
      Sub-term 1a: A * B[:, 0:2] -> 2x2x2 -> Strassen: 7
      Sub-term 1b: A * B[:, 2:6] -> 2x2x4 -> schoolbook: 8
    Term 2: 2x3x6 -> schoolbook: 36
    Total: 7 + 8 + 36 = 51.
    
    Let's try another partition:
    C = A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :] + A[:, 4:5] * B[4:5, :]
    Term 1: 2x2x6 -> 7 + 8 = 15
    Term 2: 2x2x6 -> 7 + 8 = 15  
    Term 3: 2x1x6 -> 12
    Total: 42.
    
    Or:
    C = A[:, 0:2] * B[0:2, :] + A[:, 2:5] * B[2:5, :]
    But partition B[2:5, :] differently? B[2:5,:] is 3x6.
    
    Let's try:
    Partition j into [0:2], [2:4], [4:5]:
    Term 1: A[:,0:2] (2x2) * B[0:2,:] (2x6)
      Sub: A * B[:,0:2] (2x2x2) -> Strassen 7
      Sub: A * B[:,2:6] (2x2x4) -> schoolbook 8
      Total: 15
    Term 2: A[:,2:4] (2x2) * B[2:4,:] (2x6)
      Sub: A * B[:,0:2] (2x2x2) -> Strassen 7
      Sub: A * B[:,2:6] (2x2x4) -> schoolbook 8
      Total: 15
    Term 3: A[:,4:5] (2x1) * B[4:5,:] (1x6) -> schoolbook 12
    Grand total: 15 + 15 + 12 = 42.
    
    Can we do better? What if we use Strassen on more blocks?
    
    Another idea: partition k (output columns) too.
    C[:, 0:2] = A * B[:, 0:2]  (2x5x2)
    C[:, 2:4] = A * B[:, 2:4]  (2x5x2)
    C[:, 4:6] = A * B[:, 4:6]  (2x5x2)
    
    Each 2x5x2: partition j into [0:2], [2:4], [4:5]:
      A[:,0:2] (2x2) * B[0:2,:] (2x2) -> Strassen 7
      A[:,2:4] (2x2) * B[2:4,:] (2x2) -> Strassen 7
      A[:,4:5] (2x1) * B[4:5,:] (1x2) -> schoolbook 4
      Total per output block: 18
    Grand total: 3 * 18 = 54.
    
    That's worse. Let's stick with 42.
    
    Actually, wait. Let me reconsider the 2x2x4 schoolbook. Can we do better than 8 for 2x2x4?
    2x2x4: A is 2x2, B is 2x4, C is 2x4.
    C = A * B. This is just multiplying a 2x2 by a 2x4.
    We can think of B as two 2x2 blocks: B = [B1, B2] where B1, B2 are 2x2.
    C = [A*B1, A*B2]. Each A*B1 and A*B2 is 2x2x2 -> Strassen 7 each.
    So 2x2x4 can be done with 14 terms using Strassen on each 2x2x2 sub-block.
    But schoolbook is 8, which is better. So keep schoolbook for 2x2x4.
    
    What about 2x2x4 with a different approach? The rank of 2x2x4 matrix multiplication...
    Actually, we can view it as: C[i,k] = sum_j A[i,j] B[j,k].
    For fixed i, C[i,:] = A[i,:] * B (row vector times matrix).
    This is just a linear combination of the rows of B. No tensor decomposition savings here beyond what schoolbook gives.
    
    So 42 seems to be our best with this approach.
    
    Let's also consider: can we combine the two 2x2x2 Strassen blocks?
    Term 1 sub 1a and Term 2 sub 1a both involve 2x2x2 Strassen but with different A and B matrices.
    They can't be combined because they use different A and B entries.
    
    Let's go with 42.
    """
    U_full = [[0] * 42 for _ in range(n * m)]
    V_full = [[0] * 42 for _ in range(m * p)]
    W_full = [[0] * 42 for _ in range(p * n)]
    
    U_block, V_block, W_block = _strassen_2x2_2x2_block()
    
    # Term 1: A[:, 0:2] (2x2) * B[0:2, :] (2x6)
    # Sub 1a: A[:,0:2] * B[0:2, 0:2] -> 2x2x2 Strassen
    _embed_2x2_2x2_into_larger(U_block, V_block, W_block, 0, 0, 0, n, m, p, U_full, V_full, W_full, 0)
    
    # Sub 1b: A[:,0:2] * B[0:2, 2:6] -> 2x2x4 schoolbook
    offset = 7
    for i in range(2):
        for j in range(2):
            for k in range(4):
                U_full[i * m + j][offset] = 1
                V_full[j * p + (2 + k)][offset] = 1
                W_full[(2 + k) * n + i][offset] = 1
                offset += 1
    
    # Term 2: A[:, 2:4] (2x2) * B[2:4, :] (2x6)
    # Sub 2a: A[:,2:4] * B[2:4, 0:2] -> 2x2x2 Strassen
    offset = 15
    _embed_2x2_2x2_into_larger(U_block, V_block, W_block, 0, 2, 0, n, m, p, U_full, V_full, W_full, offset)
    
    # Sub 2b: A[:,2:4] * B[2:4, 2:6] -> 2x2x4 schoolbook
    offset = 22
    for i in range(2):
        for j in range(2):
            for k in range(4):
                U_full[i * m + (2 + j)][offset] = 1
                V_full[(2 + j) * p + (2 + k)][offset] = 1
                W_full[(2 + k) * n + i][offset] = 1
                offset += 1
    
    # Term 3: A[:, 4:5] (2x1) * B[4:5, :] (1x6) -> 2x1x6 schoolbook
    for i in range(2):
        for k in range(6):
            U_full[i * m + 4][offset] = 1
            V_full[4 * p + k][offset] = 1
            W_full[k * n + i][offset] = 1
            offset += 1
    
    return U_full, V_full, W_full

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Try the hybrid Strassen + schoolbook decomposition (rank 42)
    U, V, W = _hybrid_strassen(n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
