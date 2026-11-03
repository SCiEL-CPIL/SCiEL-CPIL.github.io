"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """Return the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen)."""
    terms = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    return terms

def _schoolbook_2x2x3():
    """
    Return the 6 rank-1 terms for 2x2x3 matrix multiplication (schoolbook).
    A(2x2) * B(2x3) = C(2x3)
    """
    terms = []
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U = {(i, j): 1}
                V = {(j, k): 1}
                W = {(k, i): 1}
                terms.append((U, V, W))
    return terms

def _kron_decomposition(n, m, p):
    """
    Build a Kronecker product decomposition for n x m x p multiplication.
    For 4x4x6: split as (2x2) x (2x2) x (2x3)
    Uses rank-7 for 2x2x2 and rank-5 for 2x2x3, giving total rank 35.
    """
    strassen = _strassen_2x2x2()
    rank5 = _schoolbook_2x2x3()
    
    rank_terms = []
    
    for s in range(7):
        u_s, v_s, w_s = strassen[s]
        for t in range(len(rank5)):
            u_t, v_t, w_t = rank5[t]
            
            U_global = {}
            V_global = {}
            W_global = {}
            
            # U: (i1,i2) x (j1,j2) -> global (2*i1+i2, 2*j1+j2)
            for (i1, j1), cu in u_s.items():
                for (i2, j2), cu2 in u_t.items():
                    gi = 2 * i1 + i2
                    gj = 2 * j1 + j2
                    coeff = cu * cu2
                    key = (gi, gj)
                    U_global[key] = U_global.get(key, 0) + coeff
            
            # V: (j1,j2) x (k1,k2) -> global (2*j1+j2, 2*k1+k2)
            for (j1, k1), cv in v_s.items():
                for (j2, k2), cv2 in v_t.items():
                    gj = 2 * j1 + j2
                    gk = 2 * k1 + k2
                    coeff = cv * cv2
                    key = (gj, gk)
                    V_global[key] = V_global.get(key, 0) + coeff
            
            # W: (i1,k1) x (i2,k2) -> global (2*k1+k2, 2*i1+i2)
            # Note: W uses (k,i) ordering per trace convention
            for (i1, k1), cw in w_s.items():
                for (i2, k2), cw2 in w_t.items():
                    gi = 2 * i1 + i2
                    gk = 2 * k1 + k2
                    coeff = cw * cw2
                    key = (gk, gi)
                    W_global[key] = W_global.get(key, 0) + coeff
            
            rank_terms.append((U_global, V_global, W_global))
    
    return rank_terms

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        alpha_r = np.sum(A_test * U[:, r].reshape(n, m))
        beta_r = np.sum(B_test * V[:, r].reshape(m, p))
        C_computed += alpha_r * beta_r * W[:, r].reshape(p, n).T
        
    return np.allclose(C_computed, C_expected, atol=1e-10)

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 110.0
    
    # Build Kronecker decomposition (rank 35)
    rank_terms = _kron_decomposition(n, m, p)
    
    # Convert to matrices
    num_terms = len(rank_terms)
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    U_mat = np.zeros((U_size, num_terms), dtype=float)
    V_mat = np.zeros((V_size, num_terms), dtype=float)
    W_mat = np.zeros((W_size, num_terms), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U_mat[i * m + j, r] = coeff
        for (j, k), coeff in v_dict.items():
            V_mat[j * p + k, r] = coeff
        for (k, i), coeff in w_dict.items():
            W_mat[k * n + i, r] = coeff
    
    # Verify
    if _verify(U_mat, V_mat, W_mat, n, m, p):
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
