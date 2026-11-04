import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _construct_strassen_2x2():
    """
    Returns U, V, W for 2x2 Strassen decomposition.
    U shape: (4, 7) -> indices i*2+j
    V shape: (4, 7) -> indices j*2+k
    W shape: (4, 7) -> indices k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen products M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coefficients for A (U)
    # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # M1: A00(1), A11(1)
    U[0, 0] = 1; U[3, 0] = 1
    # M2: A10(1), A11(1)
    U[2, 1] = 1; U[3, 1] = 1
    # M3: A00(1)
    U[0, 2] = 1
    # M4: A11(1)
    U[3, 3] = 1
    # M5: A00(1), A01(1)
    U[0, 4] = 1; U[1, 4] = 1
    # M6: A10(1), A00(-1)
    U[2, 5] = 1; U[0, 5] = -1
    # M7: A01(1), A11(-1)
    U[1, 6] = 1; U[3, 6] = -1

    # Coefficients for B (V)
    # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (j*2+k)
    # M1: B00(1), B11(1)
    V[0, 0] = 1; V[3, 0] = 1
    # M2: B00(1)
    V[0, 1] = 1
    # M3: B10(1), B11(-1)
    V[2, 2] = 1; V[3, 2] = -1
    # M4: B01(1), B00(-1)
    V[1, 3] = 1; V[0, 3] = -1
    # M5: B11(1)
    V[3, 4] = 1
    # M6: B00(1), B01(1)
    V[0, 5] = 1; V[1, 5] = 1
    # M7: B10(1), B11(1)
    V[2, 6] = 1; V[3, 6] = 1

    # Coefficients for C (W)
    # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*2+i)
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 -> C00(1), C11(1)
    W[0, 0] = 1; W[3, 0] = 1
    # M2 -> C10(1), C11(-1)
    W[2, 1] = 1; W[3, 1] = -1
    # M3 -> C01(1), C11(1)
    W[1, 2] = 1; W[3, 2] = 1
    # M4 -> C00(-1), C10(1)
    W[0, 3] = -1; W[2, 3] = 1
    # M5 -> C00(-1), C01(1)
    W[0, 4] = -1; W[1, 4] = 1
    # M6 -> C11(1)
    W[3, 5] = 1
    # M7 -> C00(1)
    W[0, 6] = 1
    
    return U, V, W

def _construct_schoolbook_3x3():
    """
    Returns U, V, W for 3x3 Schoolbook decomposition.
    U shape: (9, 27)
    V shape: (9, 27)
    W shape: (9, 27)
    """
    n = 3
    rank = n * n * n
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i*n + j, col] = 1
                V[j*n + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(U1, V1, W1, U2, V2, W2):
    """
    Combines two tensor decompositions.
    If T1 = sum U1_r V1_r W1_r and T2 = sum U2_s V2_s W2_s
    Then T1 x T2 (tensor product) = sum_{r,s} (U1_r U2_s) (V1_r V2_s) (W1_r W2_s)
    This corresponds to block multiplication where each element is a sub-matrix.
    """
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 * rank2
    
    n1 = U1.shape[0]
    n2 = U2.shape[0]
    total_dim = n1 * n2
    
    U = np.zeros((total_dim, total_rank))
    V = np.zeros((total_dim, total_rank))
    W = np.zeros((total_dim, total_rank))
    
    for r in range(rank1):
        for s in range(rank2):
            c = r * rank2 + s
            U[:, c] = U1[:, r] * U2[:, s] # Kronecker product of vectors
            V[:, c] = V1[:, r] * V2[:, s]
            W[:, c] = W1[:, r] * W2[:, s]
            
    return U, V, W

def construct():
    start_time = time.time()
    
    # Fallback: Schoolbook 6x6
    def schoolbook_6x6():
        n, m, p = DIMENSIONS
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

    try:
        # Get 2x2 Strassen
        U2, V2, W2 = _construct_strassen_2x2()
        
        # Get 3x3 Schoolbook
        U3, V3, W3 = _construct_schoolbook_3x3()
        
        # Combine
        U6, V6, W6 = _combine_decompositions(U2, V2, W2, U3, V3, W3)
        
        # Convert to list of lists for interface
        U_list = U6.tolist()
        V_list = V6.tolist()
        W_list = W6.tolist()
        
        # Optional Verification (Fast for 36x36x36?)
        # 36*36*36 = 46656 elements.
        # Checking tensor contraction might be O(R * N^3) which is 189 * 46656 ~ 8.8M ops.
        # This is fast enough in NumPy.
        
        if time.time() - start_time < 10:
             # Verify
             # T[I, J, K] = sum_c U[I,c] V[J,c] W[K,c]
             # Target: 1 if indices match, 0 else.
             
             # Reshape U, V, W for einsum
             # U is (36, 189), V (36, 189), W (36, 189)
             # We want T (36, 36, 36)
             
             # This might be memory heavy if done naively, but 36^3 is small.
             # T = np.einsum('ic,jc,kc->ijk', U6, V6, W6)
             
             T_calc = np.einsum('ic,jc,kc->ijk', U6, V6, W6)
             
             # Construct Target T
             I, J, K = np.indices((36, 36, 36))
             # I = i*6+j, J = j*6+k, K = k*6+i
             # i = I // 6, j = I % 6
             # j = J // 6, k = J % 6
             # k = K // 6, i = K % 6
             
             i_U = I // 6
             j_U = I % 6
             j_V = J // 6
             k_V = J % 6
             k_W = K // 6
             i_W = K % 6
             
             # Check consistency
             # i_U == i_W, j_U == j_V, k_V == k_W
             target = (i_U == i_W) & (j_U == j_V) & (k_V == k_W)
             target = target.astype(float)
             
             if not np.allclose(T_calc, target, atol=1e-9):
                 # Mismatch, return schoolbook
                 return schoolbook_6x6()
        
        return {"U": U_list, "V": V_list, "W": W_list}

    except Exception as e:
        # If anything fails, return schoolbook
        return schoolbook_6x6()

# EVOLVE-BLOCK-END
