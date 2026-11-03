import time
import numpy as np

def _get_strassen_2x2():
    """Returns rank-7 decomposition for 2x2x2."""
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

def _get_2x2x3_hybrid():
    """
    Returns rank-11 decomposition for 2x2x3.
    Uses Strassen (7) for cols 0,1 and Schoolbook (4) for col 2.
    """
    strassen = _get_strassen_2x2()
    terms = []
    
    # Strassen part for 2x2x2 (cols 0,1 of 2x2x3)
    for u, v, w in strassen:
        # v is 2x2, corresponds to cols 0,1.
        # w is 2x2, corresponds to cols 0,1.
        # We need to map this to 2x2x3.
        # V indices (j, k) where k in {0,1}.
        # W indices (k, i) where k in {0,1}.
        # This is already in the right format for the first 2 cols.
        terms.append((u, v, w))
        
    # Schoolbook part for 2x2x1 (col 2 of 2x2x3)
    # C[i, 2] = sum_j A[i, j] B[j, 2]
    # Terms: (i, j) for A, (j, 2) for B, (2, i) for W?
    # W is p x n = 3 x 2.
    # W[k, i].
    # For col 2, k=2.
    for i in range(2):
        for j in range(2):
            u = {(i, j): 1}
            v = {(j, 2): 1}
            w = {(2, i): 1}
            terms.append((u, v, w))
            
    return terms

def _kron_decomposition(terms1, terms2):
    """
    Kronecker product of two decompositions.
    terms1: list of (U, V, W) dicts for n1 x m1 x p1
    terms2: list of (U, V, W) dicts for n2 x m2 x p2
    Result: decomposition for (n1*n2) x (m1*m2) x (p1*p2)
    """
    n1, m1, p1 = 2, 2, 2 # Dimensions of terms1 (Strassen 2x2x2)
    n2, m2, p2 = 2, 2, 3 # Dimensions of terms2 (Hybrid 2x2x3)
    
    rank = len(terms1) * len(terms2)
    U = np.zeros((n1*n2, m1*m2, rank), dtype=float) # Actually we build global matrices
    # Let's build global U, V, W directly.
    
    global_rank = 0
    U_global = []
    V_global = []
    W_global = []
    
    for t1 in terms1:
        u1, v1, w1 = t1
        for t2 in terms2:
            u2, v2, w2 = t2
            
            U_term = {}
            V_term = {}
            W_term = {}
            
            for (i1, j1), c1 in u1.items():
                for (i2, j2), c2 in u2.items():
                    gi = i1 * n2 + i2
                    gj = j1 * m2 + j2
                    coeff = c1 * c2
                    U_term[(gi, gj)] = U_term.get((gi, gj), 0) + coeff
                    
            for (j1, k1), c1 in v1.items():
                for (j2, k2), c2 in v2.items():
                    gj = j1 * m2 + j2
                    gk = k1 * p2 + k2
                    coeff = c1 * c2
                    V_term[(gj, gk)] = V_term.get((gj, gk), 0) + coeff
                    
            for (k1, i1), c1 in w1.items():
                for (k2, i2), c2 in w2.items():
                    gk = k1 * p2 + k2
                    gi = i1 * n2 + i2
                    coeff = c1 * c2
                    W_term[(gk, gi)] = W_term.get((gk, gi), 0) + coeff
                    
            U_global.append(U_term)
            V_global.append(V_term)
            W_global.append(W_term)
            global_rank += 1
            
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n * m, global_rank), dtype=float)
    V = np.zeros((m * p, global_rank), dtype=float)
    W = np.zeros((p * n, global_rank), dtype=float)
    
    for r in range(global_rank):
        for (i, j), c in U_global[r].items():
            U[i * m + j, r] = c
        for (j, k), c in V_global[r].items():
            V[j * p + k, r] = c
        for (k, i), c in W_global[r].items():
            W[k * n + i, r] = c
            
    return U, V, W

def _verify(U, V, W, n, m, p):
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
    deadline = start_time + 140.0
    
    strassen_2x2 = _get_strassen_2x2()
    hybrid_2x2x3 = _get_2x2x3_hybrid()
    
    U, V, W = _kron_decomposition(strassen_2x2, hybrid_2x2x3)
    
    if not _verify(U, V, W, n, m, p):
        # Fallback
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
