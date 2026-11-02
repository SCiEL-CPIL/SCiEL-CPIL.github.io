"""Matrix multiplication 4x4x6 - tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_strassen_2x2x2():
    """
    Returns the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Each term is (U_dict, V_dict, W_dict) with keys (i,j) for U/V/W.
    U: A indices, V: B indices, W: C indices.
    """
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    terms = []
    
    # M1
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    w1 = {(0,0): 1, (1,1): 1}
    terms.append((u1, v1, w1))
    
    # M2
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    w2 = {(1,0): 1, (1,1): -1}
    terms.append((u2, v2, w2))
    
    # M3
    u3 = {(0,0): 1}
    v3 = {(0,1): 1, (1,1): -1}
    w3 = {(0,1): 1, (1,1): 1}
    terms.append((u3, v3, w3))
    
    # M4
    u4 = {(1,1): 1}
    v4 = {(1,0): 1, (0,0): -1}
    w4 = {(0,0): 1, (1,0): 1}
    terms.append((u4, v4, w4))
    
    # M5
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    w5 = {(0,0): -1, (0,1): 1}
    terms.append((u5, v5, w5))
    
    # M6
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    w6 = {(1,1): 1}
    terms.append((u6, v6, w6))
    
    # M7
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    w7 = {(0,0): 1}
    terms.append((u7, v7, w7))
    
    return terms

def _kronecker_decomposition(terms_2x2, n_dim, m_dim, p_dim):
    """
    Constructs a decomposition for n_dim x m_dim x p_dim using Kronecker product
    of the 2x2x2 terms. Assumes n_dim, m_dim, p_dim are powers of 2.
    Here we use it for 4x4x4.
    """
    # n_dim=4, m_dim=4, p_dim=4
    # We split into 2x2 blocks.
    # The indices for the 4x4 matrices are composed of two 2-bit indices.
    # i = i1*2 + i2, j = j1*2 + j2, etc.
    
    terms_4x4 = []
    
    for u1, v1, w1 in terms_2x2:
        for u2, v2, w2 in terms_2x2:
            u_new = {}
            v_new = {}
            w_new = {}
            
            # Kronecker product of U
            for (i1, j1), c1 in u1.items():
                for (i2, j2), c2 in u2.items():
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u_new[(i, j)] = u_new.get((i, j), 0) + c1 * c2
            
            # Kronecker product of V
            for (j1, k1), c1 in v1.items():
                for (j2, k2), c2 in v2.items():
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v_new[(j, k)] = v_new.get((j, k), 0) + c1 * c2
            
            # Kronecker product of W
            for (i1, k1), c1 in w1.items():
                for (i2, k2), c2 in w2.items():
                    i = i1 * 2 + i2
                    k = k1 * 2 + k2
                    w_new[(i, k)] = w_new.get((i, k), 0) + c1 * c2
            
            terms_4x4.append((u_new, v_new, w_new))
            
    return terms_4x4

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    deadline = time.time() + 115
    
    # Strategy:
    # 1. Split B into B1 (4x4) and B2 (4x2).
    # 2. Compute C1 = A * B1 using rank-49 decomposition (Kronecker of Strassen 2x2x2).
    # 3. Compute C2 = A * B2 using schoolbook rank-32 decomposition.
    # 4. Combine. Total rank = 49 + 32 = 81.
    
    rank_terms = []
    
    # Part 1: A * B1 (4x4 * 4x4 -> 4x4)
    # B1 corresponds to columns 0..3 of B.
    # C1 corresponds to columns 0..3 of C.
    strassen_2x2 = _get_strassen_2x2x2()
    terms_4x4 = _kronecker_decomposition(strassen_2x2, 4, 4, 4)
    
    for u, v, w in terms_4x4:
        # Adjust indices for the full problem
        # U: A indices (4x4) -> (4x4)
        # V: B indices (4x4) -> (4x4) but only cols 0..3 of B
        # W: C indices (4x4) -> (4x4) but only cols 0..3 of C
        # Since B1 is the first 4 columns and C1 is the first 4 columns,
        # the indices (j,k) in V and W are already correct for the first 4 columns.
        # We just need to ensure they are mapped to the global V and W matrices.
        # Global V has shape (m*p, rank) = (4*6, rank). Index is j*p + k.
        # Global W has shape (p*n, rank) = (6*4, rank). Index is k*n + i.
        # In terms_4x4, j,k are in 0..3.
        # So V index is j*6 + k.
        # W index is k*4 + i.
        
        u_adj = {}
        v_adj = {}
        w_adj = {}
        
        for (i, j), c in u.items():
            u_adj[(i, j)] = c
            
        for (j, k), c in v.items():
            # k is 0..3, which is correct for B1
            v_adj[(j, k)] = c
            
        for (i, k), c in w.items():
            # k is 0..3, which is correct for C1
            w_adj[(i, k)] = c
            
        rank_terms.append((u_adj, v_adj, w_adj))
        
    # Part 2: A * B2 (4x4 * 4x2 -> 4x2)
    # B2 corresponds to columns 4..5 of B.
    # C2 corresponds to columns 4..5 of C.
    # Schoolbook: C[i, k] = sum_j A[i, j] B[j, k]
    # For each i, j, k (where k is 4,5), we have a term.
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(i, k): 1}
                rank_terms.append((u, v, w))
                
    rank = len(rank_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][r] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][r] = coeff
            
    # Verify correctness
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    U_mat = np.array(U, dtype=float).T
    V_mat = np.array(V, dtype=float).T
    W_mat = np.array(W, dtype=float).T
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_vec = U_mat[r].reshape(n, m)
        v_vec = V_mat[r].reshape(m, p)
        w_vec = W_mat[r].reshape(p, n)
        for i in range(n):
            for k in range(p):
                c = 0.0
                for j in range(m):
                    c += u_vec[i, j] * v_vec[j, k]
                C_computed[i, k] += c * w_vec[k, i]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook rank
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
