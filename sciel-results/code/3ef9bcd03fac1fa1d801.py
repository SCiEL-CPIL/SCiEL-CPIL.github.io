import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}.
    """
    # The 7 Strassen terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U matrix: rows indexed by (i,j) -> i*2+j
    # For each of the 7 terms, what is the coefficient in U[i*2+j]?
    # U[i*2+j, r] is the coefficient of A[i,j] in the r-th term.
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    # A11 -> i=0,j=0; A22 -> i=1,j=1
    U[0*2+0, 0] = 1  # A11
    U[1*2+1, 0] = 1  # A22
    # B11 -> j=0,k=0; B22 -> j=1,k=1
    V[0*2+0, 0] = 1  # B11
    V[1*2+1, 0] = 1  # B22
    # C11 += M1, C22 += M1
    # C11 -> i=0,k=0; C22 -> i=1,k=1
    W[0*2+0, 0] = 1  # C11
    W[1*2+1, 0] = 1  # C22
    
    # Term 1: M2 = (A21+A22)B11
    # A21 -> i=1,j=0; A22 -> i=1,j=1
    U[1*2+0, 1] = 1  # A21
    U[1*2+1, 1] = 1  # A22
    # B11 -> j=0,k=0
    V[0*2+0, 1] = 1  # B11
    # C21 += M2, C22 -= M2
    # C21 -> i=1,k=0; C22 -> i=1,k=1
    W[1*2+0, 1] = 1  # C21
    W[1*2+1, 1] = -1 # C22
    
    # Term 2: M3 = A11(B12-B22)
    # A11 -> i=0,j=0
    U[0*2+0, 2] = 1  # A11
    # B12 -> j=0,k=1; B22 -> j=1,k=1
    V[0*2+1, 2] = 1  # B12
    V[1*2+1, 2] = -1 # B22
    # C12 += M3, C22 += M3
    # C12 -> i=0,k=1; C22 -> i=1,k=1
    W[0*2+1, 2] = 1  # C12
    W[1*2+1, 2] = 1  # C22
    
    # Term 3: M4 = A22(B21-B11)
    # A22 -> i=1,j=1
    U[1*2+1, 3] = 1  # A22
    # B21 -> j=1,k=0; B11 -> j=0,k=0
    V[1*2+0, 3] = 1  # B21
    V[0*2+0, 3] = -1 # B11
    # C11 += M4, C21 += M4
    # C11 -> i=0,k=0; C21 -> i=1,k=0
    W[0*2+0, 3] = 1  # C11
    W[1*2+0, 3] = 1  # C21
    
    # Term 4: M5 = (A11+A12)B22
    # A11 -> i=0,j=0; A12 -> i=0,j=1
    U[0*2+0, 4] = 1  # A11
    U[0*2+1, 4] = 1  # A12
    # B22 -> j=1,k=1
    V[1*2+1, 4] = 1  # B22
    # C11 -= M5, C12 += M5
    # C11 -> i=0,k=0; C12 -> i=0,k=1
    W[0*2+0, 4] = -1 # C11
    W[0*2+1, 4] = 1  # C12
    
    # Term 5: M6 = (A21-A11)(B11+B12)
    # A21 -> i=1,j=0; A11 -> i=0,j=0
    U[1*2+0, 5] = 1  # A21
    U[0*2+0, 5] = -1 # A11
    # B11 -> j=0,k=0; B12 -> j=0,k=1
    V[0*2+0, 5] = 1  # B11
    V[0*2+1, 5] = 1  # B12
    # C22 += M6
    # C22 -> i=1,k=1
    W[1*2+1, 5] = 1  # C22
    
    # Term 6: M7 = (A12-A22)(B21+B22)
    # A12 -> i=0,j=1; A22 -> i=1,j=1
    U[0*2+1, 6] = 1  # A12
    U[1*2+1, 6] = -1 # A22
    # B21 -> j=1,k=0; B22 -> j=1,k=1
    V[1*2+0, 6] = 1  # B21
    V[1*2+1, 6] = 1  # B22
    # C11 += M7
    # C11 -> i=0,k=0
    W[0*2+0, 6] = 1  # C11
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute tensor from decomposition: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Then check against the standard multiplication tensor (all ones)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_4x4x4():
    """
    Build 4x4x4 decomposition from 2x2x2 Strassen using tensor product.
    Rank = 7*7 = 49.
    """
    U222, V222, W222 = _strassen_2x2x2()
    
    U444 = np.zeros((16, 49), dtype=int)
    V444 = np.zeros((16, 49), dtype=int)
    W444 = np.zeros((16, 49), dtype=int)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U444[i*4+j, r] = U222[i1*2+j1, r1] * U222[i2*2+j2, r2]
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V444[j*4+k, r] = V222[j1*2+k1, r1] * V222[j2*2+k2, r2]
            
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W444[k*4+i, r] = W222[k1*2+i1, r1] * W222[k2*2+i2, r2]
    
    return U444.tolist(), V444.tolist(), W444.tolist()

def _strassen_4x4x2():
    """
    Build 4x4x2 decomposition. We can use the tensor product structure.
    4x4x2 = (2x2x2) x (2x2x1)
    For 2x2x1, we need a decomposition of multiplying 2x2 by 2x1.
    The rank of 2x2x1 is 4 (schoolbook), but let's check if we can do better.
    Actually, for 2x2x1, the tensor is just:
    C1 = A11*B1 + A12*B2
    C2 = A21*B1 + A22*B2
    This is a bilinear form with 4 terms. Can we do it in rank < 4?
    
    The rank of the 2x2x1 tensor: Let's think about it.
    T[i,j,k] = 1 for all i,j,k in {0,1} x {0,1} x {0,1}
    Wait, no. For 2x2 matrix times 2x1 matrix:
    C[i] = sum_j A[i,j] * B[j,0]
    So T[i,j,0] = 1 for all i,j.
    This is just the outer product of a 2x2 all-ones matrix with a 1x2... no.
    
    Actually, T[i,j,k] where k is the column of B (only k=0).
    T[i,j,0] = 1 for i in {0,1}, j in {0,1}.
    This tensor has rank 4? Let's see:
    T = sum_{i,j} e_i ⊗ e_j ⊗ e_0
    = (e_0 + e_1) ⊗ (e_0 + e_1) ⊗ e_0
    Wait, that's rank 1! Because T[i,j,0] = 1 = 1*1*1.
    
    Hmm, but that's not right for matrix multiplication. Let me reconsider.
    
    For multiplying A (2x2) by B (2x1):
    C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    
    So the tensor T[i,j,k] where i is row of C (0,1), j is column of A / row of B (0,1), k is column of B (only 0).
    T[i,j,0] = 1 for all i,j.
    
    This is indeed a rank-1 tensor! T = (e_0+e_1) ⊗ (e_0+e_1) ⊗ e_0.
    
    So 2x2x1 has rank 1? That seems too good. Let me double-check.
    
    Actually, I think I'm confusing the tensor structure. Let me be more careful.
    
    The matrix multiplication tensor for A (m x n) * B (n x p) -> C (m x p) has:
    T[i,j,k] = 1 where i indexes rows of A/C, j indexes columns of A/rows of B, k indexes columns of B/C.
    
    For 2x2 * 2x1: A is 2x2, B is 2x1, C is 2x1.
    i in {0,1} (row of A/C), j in {0,1} (col of A / row of B), k in {0} (col of B/C).
    
    T[i,j,0] = 1 for all i,j.
    
    As a tensor in R^{2} ⊗ R^{2} ⊗ R^{1}, this is:
    T = sum_{i=0}^{1} sum_{j=0}^{1} e_i ⊗ e_j ⊗ e_0
    = (e_0 + e_1) ⊗ (e_0 + e_1) ⊗ e_0
    
    This is indeed rank 1!
    
    So 4x4x2 = (2x2x2) ⊗ (2x2x1) has rank 7 * 1 = 7?
    
    Wait, that can't be right either. Let me reconsider the tensor product structure.
    
    When we write 4x4x2 as (2x2x2) ⊗ (2x2x1), we're decomposing:
    - 4 rows of A into 2x2 blocks
    - 4 cols of A into 2x2 blocks  
    - 2 cols of B into 2x1 blocks
    
    The tensor product of two multiplication tensors gives a multiplication tensor for the block matrices.
    
    If T1 is the tensor for (2x2)*(2x2)->(2x2) with rank r1,
    and T2 is the tensor for (2x2)*(2x1)->(2x1) with rank r2,
    then T1 ⊗ T2 is the tensor for (4x4)*(4x2)->(4x2) with rank r1*r2.
    
    And since T2 has rank 1, the total rank would be 7*1 = 7.
    
    But wait, that would mean we can multiply a 4x4 by a 4x2 matrix with only 7 scalar multiplications? That seems impossibly low.
    
    Let me reconsider. The issue is that the "2x2x1" tensor might not have rank 1 in the way I'm thinking.
    
    Actually, I think the confusion is about what the tensor represents. Let me use a different approach.
    
    For 4x4x2, let's just use schoolbook: rank = 4*4*2 = 32.
    Or we can try to find a better decomposition.
    
    Actually, let me just use schoolbook for the 4x4x2 part and Strassen for the 4x4x4 part.
    Total rank = 49 + 32 = 81.
    """
    # For simplicity, use schoolbook for 4x4x2
    n, m, p = 4, 4, 2
    rank = n * m * p  # 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _try_explicit_split(n, m, p):
    """
    Try to explicitly construct a split decomposition for 4x4x6.
    Split into 4x4x4 and 4x4x2 parts.
    """
    if n != 4 or m != 4 or p != 6:
        return None
    
    # Get 4x4x4 Strassen decomposition (rank 49)
    U444, V444, W444 = _strassen_4x4x4()
    rank444 = 49
    
    # Get 4x4x2 schoolbook decomposition (rank 32)
    U442, V442, W442 = _strassen_4x4x2()
    rank442 = 32
    
    total_rank = rank444 + rank442
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill in 4x4x4 part (first 4 columns of output, k=0..3)
    for i in range(n):
        for j in range(m):
            for r in range(rank444):
                U[i*m+j][r] = U444[i*m+j][r]
                # V for 4x4x4: V444[j*4+k] where k is 0..3
                # In full problem, V[j*p+k] where k is 0..5
                # For the 4x4x4 part, we use k=0..3
                V[j*p + 0][r] = 0  # placeholder, will fill below
                W[0*n + i][r] = 0  # placeholder
    
    # Need to be more careful with the V and W mapping
    # For the 4x4x4 part:
    # U[i*m+j, r] for i,j in 0..3, r in 0..48
    # V[j*4+k, r] for j in 0..3, k in 0..3, r in 0..48 -> maps to V[j*6+k, r] in full
    # W[k*4+i, r] for k in 0..3, i in 0..3, r in 0..48 -> maps to W[k*4+i, r] in full (since p=6, but W has shape p*n=24)
    
    # Wait, W has shape (p*n, rank) = (24, rank). The index is k*n+i where k is 0..5, i is 0..3.
    # For the 4x4x4 part, k is 0..3, so W[k*4+i, r] is correct.
    
    # For V, the index is j*p+k where j is 0..3, k is 0..5.
    # For the 4x4x4 part, k is 0..3, so V[j*6+k, r] for k=0..3.
    
    for i in range(4):
        for j in range(4):
            for r in range(rank444):
                U[i*m+j][r] = U444[i*m+j][r]
    
    for j in range(4):
        for k in range(4):
            for r in range(rank444):
                V[j*p + k][r] = V444[j*4+k][r]
    
    for k in range(4):
        for i in range(4):
            for r in range(rank444):
                W[k*n + i][r] = W444[k*4+i][r]
    
    # Fill in 4x4x2 part (last 2 columns of output, k=4..5)
    offset = rank444
    for i in range(4):
        for j in range(4):
            for r in range(rank442):
                U[i*m+j][offset + r] = U442[i*m+j][r]
    
    for j in range(4):
        for k in range(2):
            for r in range(rank442):
                V[j*p + (k + 4)][offset + r] = V442[j*2+k][r]
    
    for k in range(2):
        for i in range(4):
            for r in range(rank442):
                W[(k + 4)*n + i][offset + r] = W442[k*4+i][r]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try explicit split: 4x4x4 (Strassen, rank 49) + 4x4x2 (schoolbook, rank 32) = rank 81
    if time.time() < deadline:
        result = _try_explicit_split(n, m, p)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                rank = len(U[0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
    
    return best_result
