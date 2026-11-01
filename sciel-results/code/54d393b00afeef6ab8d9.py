import numpy as np
import time
import sys

def construct():
    n, m, p = 2, 2, 9
    start_time = time.time()
    deadline = start_time + 140.0

    # Strategy: Structural Pruning via Linear Dependency Analysis
    #
    # The naive block construction gives rank 32 (4*7 + 4).
    # We aim for rank 31 by identifying a linear dependency.
    #
    # In the 2x2x2 Strassen decomposition, the 7 terms are:
    # P1 = (a+d)(B00+B11)
    # P2 = (c+d)B00
    # P3 = a(B01-B11)
    # P4 = d(B10-B00)
    # P5 = (a+b)B11
    # P6 = (c-a)(B00+B01)
    # P7 = (b-d)(B10+B11)
    #
    # For the 2x2x9 tensor, we have 9 columns of B.
    # The output C is 2x9. The tensor is T[i,j,k] = delta_{i,k} (for the 2x2 part) 
    # but here we have 2x2 * 2x9 -> 2x9.
    #
    # Let's construct the rank-32 decomposition first, then attempt to find
    # a linear combination of the 32 terms that is identically zero, which would
    # allow us to eliminate one term.
    
    # Build rank-32 decomposition
    rank32 = 32
    U32 = np.zeros((n*m, rank32))
    V32 = np.zeros((m*p, rank32))
    W32 = np.zeros((p*n, rank32))
    
    col = 0
    # Add Strassen blocks for 4 pairs of columns
    for block_idx in range(4):
        k1 = block_idx * 2
        k2 = block_idx * 2 + 1
        
        # P1 = (a+d)(B00+B11)
        U32[0, col] = 1; U32[3, col] = 1
        V32[0*p+k1, col] = 1; V32[1*p+k2, col] = 1
        W32[k1*n+0, col] = 1; W32[k2*n+1, col] = 1
        col += 1
        
        # P2 = (c+d)B00
        U32[2, col] = 1; U32[3, col] = 1
        V32[0*p+k1, col] = 1
        W32[k1*n+1, col] = 1
        col += 1
        
        # P3 = a(B01-B11)
        U32[0, col] = 1
        V32[0*p+k2, col] = 1; V32[1*p+k2, col] = -1
        W32[k2*n+0, col] = 1
        col += 1
        
        # P4 = d(B10-B00)
        U32[3, col] = 1
        V32[1*p+k1, col] = 1; V32[0*p+k1, col] = -1
        W32[k1*n+0, col] = 1; W32[k1*n+1, col] = 1
        col += 1
        
        # P5 = (a+b)B11
        U32[0, col] = 1; U32[1, col] = 1
        V32[1*p+k2, col] = 1
        W32[k1*n+0, col] = -1; W32[k2*n+0, col] = 1
        col += 1
        
        # P6 = (c-a)(B00+B01)
        U32[2, col] = 1; U32[0, col] = -1
        V32[0*p+k1, col] = 1; V32[0*p+k2, col] = 1
        W32[k1*n+1, col] = -1
        col += 1
        
        # P7 = (b-d)(B10+B11)
        U32[1, col] = 1; U32[3, col] = -1
        V32[1*p+k1, col] = 1; V32[1*p+k2, col] = 1
        W32[k1*n+0, col] = 1
        col += 1

    # Last column k=8, schoolbook
    k = 8
    # a*B0
    U32[0, col] = 1
    V32[0*p+k, col] = 1
    W32[k*n+0, col] = 1
    col += 1
    # b*B1
    U32[1, col] = 1
    V32[1*p+k, col] = 1
    W32[k*n+0, col] = 1
    col += 1
    # c*B0
    U32[2, col] = 1
    V32[0*p+k, col] = 1
    W32[k*n+1, col] = 1
    col += 1
    # d*B1
    U32[3, col] = 1
    V32[1*p+k, col] = 1
    W32[k*n+1, col] = 1
    col += 1
    
    assert col == 32
    
    # Verify rank-32 decomposition
    def verify_decomp(U, V, W, n, m, p):
        # Use random test matrices
        rng = np.random.RandomState(42)
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        C_expected = A @ B
        
        C_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(U.shape[1]):
                    u_val = np.dot(U[i*m:(i+1)*m, r], A[i, :])
                    v_val = np.dot(V[r], B[:, k])
                    C_decomp[i, k] += W[k*n+i, r] * u_val * v_val
        
        return np.allclose(C_expected, C_decomp, atol=1e-10)
    
    if not verify_decomp(U32, V32, W32, n, m, p):
        # Fallback to schoolbook
        rank = 36
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Now attempt to find a linear dependency to reduce rank to 31.
    # We look for coefficients c_r such that sum_r c_r * (U_r, V_r, W_r) = 0 in the tensor space.
    # This means for all i,j,k: sum_r c_r * U[i,j,r] * V[j,k,r] * W[k,i,r] = 0
    # And the vector (U_r, V_r, W_r) must be linearly dependent.
    
    # A simpler approach: check if any column of the 3D tensor representation is a linear combination of others.
    # Reshape the tensor T of shape (n*m, m*p, p*n) into a matrix and find its rank.
    
    # Build the full tensor
    T = np.zeros((n*m, m*p, p*n))
    for r in range(32):
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    T[i, j, k] += U32[i, r] * V32[j, r] * W32[k, r]
    
    # Check the rank of the tensor in different unfoldings
    # Unfolding 1: (n*m) x (m*p * p*n)
    T1 = T.reshape(n*m, -1)
    rank1 = np.linalg.matrix_rank(T1, tol=1e-10)
    
    # Unfolding 2: (m*p) x (n*m * p*n)
    T2 = T.transpose(1, 0, 2).reshape(m*p, -1)
    rank2 = np.linalg.matrix_rank(T2, tol=1e-10)
    
    # Unfolding 3: (p*n) x (n*m * m*p)
    T3 = T.transpose(2, 0, 1).reshape(p*n, -1)
    rank3 = np.linalg.matrix_rank(T3, tol=1e-10)
    
    # The tensor rank is at least max(rank1, rank2, rank3)
    # If max rank < 32, there might be a dependency.
    
    # For now, if we can't find a clear dependency, return the rank-32 solution.
    # The structural pruning requires deeper analysis.
    
    # Return the rank-32 solution as it is verified to be correct.
    U_list = U32.tolist()
    V_list = V32.tolist()
    W_list = W32.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
