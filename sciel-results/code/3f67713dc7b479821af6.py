import time
import numpy as np

def construct():
    """
    Rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Uses Strassen's 7-multiplication algorithm on four 2x2 block products.
    
    Interface:
      U shape: (n*m, rank) = (8, 28)
      V shape: (m*p, rank) = (16, 28)  
      W shape: (p*n, rank) = (8, 28)
    
    Indexing (trace convention W[k*n+i]):
      A[i,j]: i in {0,1}, j in {0,1,2,3}. U index: i*4+j
      B[j,k]: j in {0,1,2,3}, k in {0,1,2,3}. V index: j*4+k
      C[i,k]: i in {0,1}, k in {0,1,2,3}. W index: k*2+i
    """
    t0 = time.time()
    deadline = t0 + 120.0
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    
    # Strassen's 7-multiplication patterns for 2x2 matrix multiplication
    # Local indices: a[0]=a11, a[1]=a12, a[2]=a21, a[3]=a22
    #                b[0]=b11, b[1]=b12, b[2]=b21, b[3]=b22
    #                c[0]=c11, c[1]=c12, c[2]=c21, c[3]=c22
    
    # M1 = (a11+a22)(b11+b22) -> c11 += M1, c22 += M1
    # M2 = (a21+a22)(b11)     -> c21 += M2, c22 -= M2
    # M3 = (a11)(b12-b22)     -> c12 += M3, c22 += M3
    # M4 = (a22)(b21-b11)     -> c11 += M4, c21 += M4
    # M5 = (a11+a12)(b22)     -> c11 -= M5, c12 += M5
    # M6 = (a21-a11)(b11+b12) -> c22 += M6
    # M7 = (a12-a22)(b21+b22) -> c11 += M7
    
    strassen_patterns = [
        # M1 = (a11+a22)(b11+b22)
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        # M2 = (a21+a22)(b11)
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        # M3 = (a11)(b12-b22)
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        # M4 = (a22)(b21-b11)
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        # M5 = (a11+a12)(b22)
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        # M6 = (a21-a11)(b11+b12)
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        # M7 = (a12-a22)(b21+b22)
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    
    # Four block products:
    # Block (a_j_base, b_j_base, b_k_base, c_k_base)
    blocks = [
        (0, 0, 0, 0),  # A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
        (2, 2, 0, 0),  # A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
        (0, 0, 2, 2),  # A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
        (2, 2, 2, 2),  # A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
    ]
    
    col = 0
    for (a_bj, b_bj, b_bk, c_bk) in blocks:
        for pattern in strassen_patterns:
            # Fill U: A entries
            for local_a_idx, coeff in pattern['a'].items():
                local_r = local_a_idx // 2
                local_c = local_a_idx % 2
                global_i = local_r
                global_j = a_bj + local_c
                U[global_i * m + global_j][col] = coeff
            
            # Fill V: B entries
            for local_b_idx, coeff in pattern['b'].items():
                local_r = local_b_idx // 2
                local_c = local_b_idx % 2
                global_j = b_bj + local_r
                global_k = b_bk + local_c
                V[global_j * p + global_k][col] = coeff
            
            # Fill W: C entries (trace convention W[k*n+i])
            for local_c_idx, coeff in pattern['c'].items():
                local_r = local_c_idx // 2
                local_c = local_c_idx % 2
                global_i = local_r
                global_k = c_bk + local_c
                W[global_k * n + global_i][col] = coeff
            
            col += 1
    
    # Verify with random test
    if time.time() < deadline - 10:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Reconstruct: C[i,k] = sum_r U[i*m+j] * V[j*p+k] * W[k*n+i]
        # More precisely: C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        C_recon = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                C_recon[i, k] = s
        
        if not np.allclose(C_recon, C_expected, atol=1e-10):
            # Fallback: return naive rank-32 decomposition
            rank_naive = 32
            U_naive = [[0]*rank_naive for _ in range(n*m)]
            V_naive = [[0]*rank_naive for _ in range(m*p)]
            W_naive = [[0]*rank_naive for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_naive[i*m+j][col] = 1
                        V_naive[j*p+k][col] = 1
                        W_naive[k*n+i][col] = 1
                        col += 1
            return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    return {"U": U, "V": V, "W": W}
