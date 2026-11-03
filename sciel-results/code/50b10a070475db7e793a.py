"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_decomposition(n, m, p):
    """
    Builds a rank-77 decomposition for 4x4 x 4x6 matrix multiplication.
    Uses tensor product of 2x2x3 (rank 11) and 2x2x2 Strassen (rank 7).
    (2,2,3) x (2,2,2) = (4,4,6)
    """
    # Strassen 2x2x2 terms: (U_dict, V_dict, W_dict)
    # U: (i,j), V: (j,k), W: (k,i)
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # 2x2x3 terms: rank 11
    # First 7: Strassen on cols 0,1 (same as strassen_2x2 but V,W only use k in {0,1})
    # Last 4: schoolbook for col 2
    terms_2x2x3 = []
    for s in range(7):
        u_s, v_s, w_s = strassen_2x2[s]
        terms_2x2x3.append((u_s, v_s, w_s))
    
    # Schoolbook for 3rd column: C_{i,2} = A_{i,0}*B_{0,2} + A_{i,1}*B_{1,2}
    # 4 terms:
    terms_2x2x3.append((
        {(0,0): 1},  # A[0,0]
        {(0,2): 1},  # B[0,2]
        {(2,0): 1}   # C[2,0] (k=2, i=0)
    ))
    terms_2x2x3.append((
        {(0,1): 1},  # A[0,1]
        {(1,2): 1},  # B[1,2]
        {(2,0): 1}   # C[2,0]
    ))
    terms_2x2x3.append((
        {(1,0): 1},  # A[1,0]
        {(0,2): 1},  # B[0,2]
        {(2,1): 1}   # C[2,1] (k=2, i=1)
    ))
    terms_2x2x3.append((
        {(1,1): 1},  # A[1,1]
        {(1,2): 1},  # B[1,2]
        {(2,1): 1}   # C[2,1]
    ))
    
    rank_terms = []
    
    # Tensor product: (2,2,3) x (2,2,2) -> (4,4,6)
    # Local indices for first factor (2x2x3): i1,j1 in {0,1}, k1 in {0,1,2}
    # Local indices for second factor (2x2x2): i2,j2 in {0,1}, k2 in {0,1}
    # Global: i = 2*i1 + i2, j = 2*j1 + j2, k = 3*k1 + k2
    for a in range(11):
        u_a, v_a, w_a = terms_2x2x3[a]
        for b in range(7):
            u_b, v_b, w_b = strassen_2x2[b]
            
            U_global = {}
            V_global = {}
            W_global = {}
            
            # U: (i,j)
            for (i1, j1), cu1 in u_a.items():
                for (i2, j2), cu2 in u_b.items():
                    gi = 2 * i1 + i2
                    gj = 2 * j1 + j2
                    coeff = cu1 * cu2
                    key = (gi, gj)
                    U_global[key] = U_global.get(key, 0) + coeff
            
            # V: (j,k)
            for (j1, k1), cv1 in v_a.items():
                for (j2, k2), cv2 in v_b.items():
                    gj = 2 * j1 + j2
                    gk = 3 * k1 + k2
                    coeff = cv1 * cv2
                    key = (gj, gk)
                    V_global[key] = V_global.get(key, 0) + coeff
            
            # W: (k,i)
            for (k1, i1), cw1 in w_a.items():
                for (k2, i2), cw2 in w_b.items():
                    gk = 3 * k1 + k2
                    gi = 2 * i1 + i2
                    coeff = cw1 * cw2
                    key = (gk, gi)
                    W_global[key] = W_global.get(key, 0) + coeff
            
            rank_terms.append((U_global, V_global, W_global))
    
    rank = len(rank_terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j, r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k, r] = coeff
        for (k, i), coeff in w_dict.items():
            W[k * n + i, r] = coeff
            
    return U, V, W

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
    
    U, V, W = _build_decomposition(n, m, p)
    
    if not _verify(U, V, W, n, m, p):
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
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
