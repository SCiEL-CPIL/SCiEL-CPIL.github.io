import numpy as np
import time

def construct():
    n, m, p = 2, 2, 9
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # Use block decomposition: 4 pairs of columns (rank 7 each) + 1 single column (rank 4)
    # Total rank: 4*7 + 4 = 32
    
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0.0] * rank for _ in range(m * p)]  # 18 x 32
    W = [[0.0] * rank for _ in range(p * n)]  # 18 x 32
    
    col = 0
    
    # For each pair of columns (k, k+1) for k = 0,2,4,6
    for pair_idx in range(4):
        k1 = pair_idx * 2
        k2 = pair_idx * 2 + 1
        
        # Rank-7 decomposition for 2x2 x 2x2
        # m1 = (a+d)(x+w), m2 = (a+d)(z+w), m3 = (a+b)(x+z)
        # m4 = (c+d)(y+w), m5 = (a-b)(x-z), m6 = (c-d)(y-w), m7 = (c+d)(x-w)
        # where A = [[a,b],[c,d]], B = [[x,y],[z,w]] with x=B[0,k1], y=B[0,k2], z=B[1,k1], w=B[1,k2]
        
        # m1: (a+d)(x+w)
        U[0][col] = 1; U[3][col] = 1  # a=U[0], d=U[3] (A[1,1])
        V[0 * p + k1][col] = 1; V[1 * p + k2][col] = 1  # x=B[0,k1], w=B[1,k2]
        W[k1 * n + 0][col] = 1  # C[0,k1] += m1
        W[k2 * n + 0][col] = 1  # C[0,k2] += m1
        W[k1 * n + 1][col] = 1  # C[1,k1] += m1
        W[k2 * n + 1][col] = 1  # C[1,k2] += m1
        col += 1
        
        # m2: (a+d)(z+w)
        U[0][col] = 1; U[3][col] = 1
        V[1 * p + k1][col] = 1; V[1 * p + k2][col] = 1  # z=B[1,k1], w=B[1,k2]
        W[k1 * n + 0][col] = -1  # C[0,k1] -= m2
        W[k2 * n + 0][col] = 1   # C[0,k2] += m2
        W[k1 * n + 1][col] = -1  # C[1,k1] -= m2
        W[k2 * n + 1][col] = 1   # C[1,k2] += m2
        col += 1
        
        # m3: (a+b)(x+z)
        U[0][col] = 1; U[1][col] = 1  # a=U[0], b=U[1]
        V[0 * p + k1][col] = 1; V[1 * p + k1][col] = 1  # x=B[0,k1], z=B[1,k1]
        W[k1 * n + 0][col] = 0
        W[k2 * n + 0][col] = 0
        W[k1 * n + 1][col] = 0
        W[k2 * n + 1][col] = 1   # C[1,k2] += m3
        col += 1
        
        # m4: (c+d)(y+w)
        U[2][col] = 1; U[3][col] = 1  # c=U[2], d=U[3]
        V[0 * p + k2][col] = 1; V[1 * p + k2][col] = 1  # y=B[0,k2], w=B[1,k2]
        W[k1 * n + 0][col] = 0
        W[k2 * n + 0][col] = 0
        W[k1 * n + 1][col] = 1   # C[1,k1] += m4
        W[k2 * n + 1][col] = -1  # C[1,k2] -= m4
        col += 1
        
        # m5: (a-b)(x-z)
        U[0][col] = 1; U[1][col] = -1
        V[0 * p + k1][col] = 1; V[1 * p + k1][col] = -1  # x=B[0,k1], z=B[1,k1]
        W[k1 * n + 0][col] = 1   # C[0,k1] += m5
        W[k2 * n + 0][col] = 0
        W[k1 * n + 1][col] = 0
        W[k2 * n + 1][col] = 0
        col += 1
        
        # m6: (c-d)(y-w)
        U[2][col] = 1; U[3][col] = -1
        V[0 * p + k2][col] = 1; V[1 * p + k2][col] = -1  # y=B[0,k2], w=B[1,k2]
        W[k1 * n + 0][col] = 0
        W[k2 * n + 0][col] = 0
        W[k1 * n + 1][col] = 1   # C[1,k1] += m6
        W[k2 * n + 1][col] = -1  # C[1,k2] -= m6
        col += 1
        
        # m7: (c+d)(x-w)
        U[2][col] = 1; U[3][col] = 1
        V[0 * p + k1][col] = 1; V[1 * p + k2][col] = -1  # x=B[0,k1], w=B[1,k2]
        W[k1 * n + 0][col] = -1  # C[0,k1] -= m7
        W[k2 * n + 0][col] = 1   # C[0,k2] += m7
        W[k1 * n + 1][col] = -1  # C[1,k1] -= m7
        W[k2 * n + 1][col] = 0
        col += 1
    
    # Last single column k=8: rank-4 (schoolbook for 2x2 x 2x1)
    k = 8
    # C[0,k] = a*B[0,k] + b*B[1,k]
    # C[1,k] = c*B[0,k] + d*B[1,k]
    
    # Term 1: a * B[0,k]
    U[0][col] = 1
    V[0 * p + k][col] = 1
    W[k * n + 0][col] = 1
    col += 1
    
    # Term 2: b * B[1,k]
    U[1][col] = 1
    V[1 * p + k][col] = 1
    W[k * n + 0][col] = 1
    col += 1
    
    # Term 3: c * B[0,k]
    U[2][col] = 1
    V[0 * p + k][col] = 1
    W[k * n + 1][col] = 1
    col += 1
    
    # Term 4: d * B[1,k]
    U[3][col] = 1
    V[1 * p + k][col] = 1
    W[k * n + 1][col] = 1
    col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify the decomposition with random test
    def verify(U, V, W, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Random test matrices
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Compute via tensor decomposition
        # C[i,k] = sum_r U[i*m+j? ...] 
        # U has shape (n*m, rank), indexed by [i*m+j]
        # V has shape (m*p, rank), indexed by [j*p+k]
        # W has shape (p*n, rank), indexed by [k*n+i]
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    # C[i,k] gets contribution from W[k*n+i][r] * (U part) * (V part)
                    # Actually: C[i,k] = sum_r W[k*n+i][r] * (sum_j U[i*m+j][r] * A[i,j]) * ... 
                    # No, the tensor is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # And C[i,k] = sum_j A[i,j] * B[j,k]
                    # So we need: sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) (sum_j' V[j'*p+k,r] B[j',k]) W[k*n+i,r]
                    # This is the standard tensor decomposition check
                    pass
        
        # Better: verify by checking that the tensor matches
        # Build the multiplication tensor T where T[i,j,k] = 1 if this term contributes
        # Actually, just check: for random A, B, compute C = A@B and also compute via the decomposition
        
        # The decomposition says: C[i,k] = sum_r W[k*n+i, r] * (U[i*m+0,r]*A[i,0] + U[i*m+1,r]*A[i,1]) * (V[0*p+k,r]*B[0,k] + V[1*p+k,r]*B[1,k])
        # Wait, that's not right either. Let me think...
        
        # U[i*m+j, r] is the coefficient for A[i,j] in term r
        # V[j*p+k, r] is the coefficient for B[j,k] in term r  
        # W[k*n+i, r] is the coefficient for C[i,k] from term r
        
        # So: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        
        C_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    u_val = 0.0
                    for j in range(m):
                        u_val += U[i*m+j, r] * A[i,j]
                    v_val = 0.0
                    for j in range(m):
                        v_val += V[j*p+k, r] * B[j,k]
                    C_decomp[i,k] += W[k*n+i, r] * u_val * v_val
        
        return np.allclose(C_expected, C_decomp, atol=1e-10)
    
    if not verify(U, V, W, n, m, p):
        # Fall back to schoolbook rank 36
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
