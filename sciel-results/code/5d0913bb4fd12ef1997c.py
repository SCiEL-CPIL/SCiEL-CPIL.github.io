"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Kronecker product structure
    - 4x4x6 = (2x2x2) ⊗ (2x2x3)
    - 2x2x2 uses Strassen (rank 7)
    - 2x2x3 uses Strassen for first 2 cols (rank 7) + trivial for 3rd col (rank 4) = rank 11
    - Total rank = 7 * 11 = 77
    
    Then attempt rank reduction via linear dependency detection on the coefficient tensors.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strassen decomposition for 2x2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2x2: n=2, m=2, p=2
    # U rows: i*m+j = i*2+j for i,j in 0..1
    # V rows: j*p+k = j*2+k for j,k in 0..1
    # W rows: k*n+i = k*2+i for k,i in 0..1
    
    U1 = np.zeros((4, 7), dtype=int)
    V1 = np.zeros((4, 7), dtype=int)
    W1 = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00+a11)(b00+b11)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # M2 = (a10+a11)b00
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1; W1[3][1] = -1
    
    # M3 = a00(b01-b11)
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1
    
    # M4 = a11(b10-b00)
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1
    
    # M5 = (a00+a01)b11
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1
    
    # For 2x2x3: A is 2x2, B is 2x3, C is 2x3
    # Rank 11: Strassen for first 2 columns (7 terms) + trivial for 3rd column (4 terms)
    
    U2 = np.zeros((4, 11), dtype=int)
    V2 = np.zeros((6, 11), dtype=int)
    W2 = np.zeros((6, 11), dtype=int)
    
    # Strassen part (columns 0-1), terms 0-6
    # V2 rows: j*p+k = j*3+k for j in 0..1, k in 0..2
    # W2 rows: k*n+i = k*2+i for k in 0..2, i in 0..1
    
    for c in range(7):
        U2[0][c] = U1[0][c]
        U2[1][c] = U1[1][c]
        U2[2][c] = U1[2][c]
        U2[3][c] = U1[3][c]
        
        # V1 rows: j*2+k -> V2 rows: j*3+k
        # j=0,k=0 -> V2[0], j=0,k=1 -> V2[1], j=1,k=0 -> V2[3], j=1,k=1 -> V2[4]
        V2[0][c] = V1[0][c]
        V2[1][c] = V1[1][c]
        V2[3][c] = V1[2][c]
        V2[4][c] = V1[3][c]
        V2[2][c] = 0
        V2[5][c] = 0
        
        # W1 rows: k*2+i -> W2 rows: k*2+i
        W2[0][c] = W1[0][c]
        W2[1][c] = W1[1][c]
        W2[2][c] = W1[2][c]
        W2[3][c] = W1[3][c]
        W2[4][c] = 0
        W2[5][c] = 0
    
    # Trivial part (column 2), terms 7-10
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # Term 7: A[0,0]*B[0,2] -> C[0,2]
    U2[0][7] = 1
    V2[2][7] = 1  # j=0, k=2 -> row 0*3+2 = 2
    W2[4][7] = 1  # k=2, i=0 -> row 2*2+0 = 4
    
    # Term 8: A[0,1]*B[1,2] -> C[0,2]
    U2[1][8] = 1
    V2[5][8] = 1  # j=1, k=2 -> row 1*3+2 = 5
    W2[4][8] = 1  # k=2, i=0 -> row 2*2+0 = 4
    
    # Term 9: A[1,0]*B[0,2] -> C[1,2]
    U2[2][9] = 1
    V2[2][9] = 1  # j=0, k=2 -> row 0*3+2 = 2
    W2[5][9] = 1  # k=2, i=1 -> row 2*2+1 = 5
    
    # Term 10: A[1,1]*B[1,2] -> C[1,2]
    U2[3][10] = 1
    V2[5][10] = 1  # j=1, k=2 -> row 1*3+2 = 5
    W2[5][10] = 1  # k=2, i=1 -> row 2*2+1 = 5
    
    # Now Kronecker product:
    # Total rank = 7 * 11 = 77
    rank = 7 * 11
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((24, rank), dtype=int)
    W = np.zeros((24, rank), dtype=int)
    
    # Index mapping:
    # A (4x4) = A1 (2x2) ⊗ A2 (2x2)
    # B (4x6) = B1 (2x2) ⊗ B2 (2x3)
    # C (4x6) = C1 (2x2) ⊗ C2 (2x3)
    
    # A[i,j] where i = i1*2+i2, j = j1*2+j2
    # B[j,k] where j = j1*2+j2, k = k1*3+k2
    # C[i,k] where i = i1*2+i2, k = k1*3+k2
    
    # U for full: row index = i*4+j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
    # U1 row = i1*2+j1, U2 row = i2*2+j2
    # So U_full[i1*8+i2*4+j1*2+j2, c1*11+c2] = U1[i1*2+j1][c1] * U2[i2*2+j2][c2]
    
    # V for full: row index = j*6+k = (j1*2+j2)*6 + (k1*3+k2) = j1*12 + j2*6 + k1*3 + k2
    # V1 row = j1*2+k1, V2 row = j2*3+k2
    # So V_full[j1*12+j2*6+k1*3+k2, c1*11+c2] = V1[j1*2+k1][c1] * V2[j2*3+k2][c2]
    
    # W for full: row index = k*4+i = (k1*3+k2)*4 + (i1*2+i2) = k1*12 + k2*4 + i1*2 + i2
    # W1 row = k1*2+i1, W2 row = k2*2+i2
    # So W_full[k1*12+k2*4+i1*2+i2, c1*11+c2] = W1[k1*2+i1][c1] * W2[k2*2+i2][c2]
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1][c1]
                            u2 = U2[i2*2+j2][c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U[row][c] = u1 * u2
            
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2+k1][c1]
                            v2 = V2[j2*3+k2][c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*12 + j2*6 + k1*3 + k2
                                V[row][c] = v1 * v2
            
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1][c1]
                            w2 = W2[k2*2+i2][c2]
                            if w1 != 0 and w2 != 0:
                                row = k1*12 + k2*4 + i1*2 + i2
                                W[row][c] = w1 * w2
    
    # Verify the decomposition
    def _verify(U_mat, V_mat, W_mat, n, m, p):
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            alpha_r = np.sum(A_test * U_mat[:, r].reshape(n, m))
            beta_r = np.sum(B_test * V_mat[:, r].reshape(m, p))
            C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=int)
        V = np.zeros((m * p, rank_school), dtype=int)
        W = np.zeros((p * n, rank_school), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt rank reduction by finding linear dependencies
    # The key insight: if we have a set of rank-1 terms that sum to zero as a tensor,
    # we can eliminate one term. We detect this by checking if the coefficient
    # tensors are linearly dependent.
    
    # Build the tensor of coefficients: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Actually, the tensor is: T = sum_r U_r ⊗ V_r ⊗ W_r (as vectors)
    # We want to check if the set of vectors {U_r ⊗ V_r ⊗ W_r} is linearly independent.
    
    # Flatten U, V, W into vectors of length n*m, m*p, p*n respectively
    # The tensor T is in R^{n*m * m*p * p*n}, but that's huge.
    
    # Instead, use the fact that the tensor decomposes as:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We can check linear independence by building a matrix where each column r is
    # the vectorization of the rank-1 tensor U_r ⊗ V_r ⊗ W_r, but that's too large.
    
    # Alternative: use random linear projections to check for dependencies.
    # If we pick random vectors x in R^{n*m}, y in R^{m*p}, z in R^{p*n},
    # then the scalar <x, U_r> * <y, V_r> * <z, W_r> should satisfy the same
    # linear dependencies as the tensors.
    
    # But this only detects dependencies that are "visible" through the projections.
    
    # A more direct approach: since the entries are integers, we can try to find
    # integer linear combinations that sum to zero. But this is hard in general.
    
    # For now, let's just check if any two columns are proportional (a simple dependency).
    # If U[:,r1] = alpha * U[:,r2], V[:,r1] = beta * V[:,r2], W[:,r1] = gamma * W[:,r2]
    # with alpha*beta*gamma = 1, then we have a dependency.
    
    # Actually, the simplest check: if two rank-1 tensors are identical, we can
    # combine them. But in our construction, all terms are distinct.
    
    # Let's try a different approach: use the fact that the Kronecker product
    # structure may introduce redundancies. Specifically, if the 2x2x2 Strassen
    # and 2x2x3 decompositions have internal dependencies, the Kronecker product
    # will amplify them.
    
    # For now, return the rank-77 decomposition. Further optimization would require
    # more sophisticated linear algebra to detect and eliminate dependencies.
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find linear dependencies by checking the rank of the coefficient matrix
    # formed by flattening the tensors.
    
    # The tensor T can be viewed as a matrix of shape (n*m*p, rank) if we fix
    # one mode, but that's not quite right.
    
    # Actually, the correct way: T is a 3-tensor of shape (n*m, m*p, p*n) but
    # with the constraint that the indices are linked: i from U, j from U and V,
    # k from V and W, i from W. So it's really a 3-tensor with shape (n*m, m*p, p*n)
    # where the modes are linked.
    
    # Let's just check if the rank-77 decomposition can be improved by removing
    # terms that are linear combinations of others.
    
    # Build a matrix M where M[row, r] is the coefficient of term r in the
    # flattened tensor. The flattened tensor has size n*m*p = 4*4*6 = 96.
    
    # Actually, the tensor T has shape (n, m, p) = (4, 4, 6), so it has 96 entries.
    # Each term r contributes a rank-1 tensor of shape (4, 4, 6).
    
    # Let's build the matrix of size (96, 77) where entry [i*m*p + j*p + k, r]
    # = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Wait, that's not right either. The tensor entry T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # where U[i,j,r] is the (i,j) entry of the r-th U matrix, etc.
    
    # So T[i,j,k] = sum_r U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
    
    # To check linear independence of the terms, we can build a matrix where each
    # column r is the vector [U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]]
    # for all (i,j,k). But this is a 96x77 matrix, and we want to check if the
    # columns are linearly independent.
    
    # Actually, that's not the right way to check. The tensor T is the sum of
    # rank-1 tensors, and we want to know if the set of rank-1 tensors is linearly
    # independent as elements of the tensor space.
    
    # The tensor space has dimension n*m*p = 96. Each rank-1 tensor is a vector in
    # this 96-dimensional space. So we can build a 96x77 matrix where column r is
    # the vectorization of the r-th rank-1 tensor, and check its rank.
    
    T_mat = np.zeros((96, 77), dtype=float)
    for r in range(77):
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    T_mat[i*24 + j*6 + k, r] = U[i*4+j, r] * V[j*6+k, r] * W[k*4+i, r]
    
    # Check the rank of T_mat
    try:
        rank_T = np.linalg.matrix_rank(T_mat, tol=1e-10)
    except:
        rank_T = 77
    
    # If rank_T < 77, there are linear dependencies, and we can potentially
    # reduce the rank. But finding the actual reduced decomposition is non-trivial.
    
    # For now, just return the rank-77 decomposition.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
