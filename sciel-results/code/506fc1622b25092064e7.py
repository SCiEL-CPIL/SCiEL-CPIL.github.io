"""Matrix multiplication 4x4x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# Strassen 2x2x2 terms: (U_dict, V_dict, W_dict)
# Indices are (row, col) for each matrix
# U: A indices (i,j), V: B indices (j,k), W: C indices (k,i) -> W stored as (k,i)
STRASSEN_TERMS = [
    ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
    ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
    ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
    ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
    ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
    ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
]

def _build_rank81_decomposition(n, m, p):
    """
    Build rank-81 decomposition for 4x4 x 4x6 matrix multiplication.
    
    Strategy:
    - Split into 4x4x4 part (columns 0-3 of B) and 4x4x2 part (columns 4-5 of B)
    - 4x4x4: Use tensor product of two Strassen 2x2x2 decompositions -> rank 49
    - 4x4x2: Schoolbook -> rank 32
    - Total rank: 81
    
    Indexing convention:
    - U[i*m + j, r] for A_{i,j}
    - V[j*p + k, r] for B_{j,k}
    - W[k*n + i, r] for C_{i,k} (trace convention W[k*n+i, r])
    """
    rank_terms = []
    
    # Part 1: 4x4x4 via tensor product of Strassen (rank 49)
    # Global indices: i, j in 0..3 for A; j, k in 0..3 for B; k, i in 0..3 for C
    # Local indices: i1, i2, j1, j2, k1, k2 in 0..1
    # Global i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    
    for s in range(7):
        u_s, v_s, w_s = STRASSEN_TERMS[s]
        for t in range(7):
            u_t, v_t, w_t = STRASSEN_TERMS[t]
            
            U_global = {}
            V_global = {}
            W_global = {}
            
            # U part: A indices (i, j)
            for (i1, j1), cu in u_s.items():
                for (i2, j2), cu2 in u_t.items():
                    gi = 2 * i1 + i2
                    gj = 2 * j1 + j2
                    coeff = cu * cu2
                    key = (gi, gj)
                    U_global[key] = U_global.get(key, 0) + coeff
            
            # V part: B indices (j, k)
            for (j1, k1), cv in v_s.items():
                for (j2, k2), cv2 in v_t.items():
                    gj = 2 * j1 + j2
                    gk = 2 * k1 + k2
                    coeff = cv * cv2
                    key = (gj, gk)
                    V_global[key] = V_global.get(key, 0) + coeff
            
            # W part: C indices (k, i) -> W[k*n + i]
            for (k1, i1), cw in w_s.items():
                for (k2, i2), cw2 in w_t.items():
                    gk = 2 * k1 + k2
                    gi = 2 * i1 + i2
                    coeff = cw * cw2
                    key = (gk, gi)
                    W_global[key] = W_global.get(key, 0) + coeff
            
            rank_terms.append((U_global, V_global, W_global))
    
    # Part 2: 4x4x2 schoolbook (rank 32)
    # For columns k=4,5 of B, we need A_{i,j} * B_{j,k} -> C_{i,k}
    for i in range(4):
        for j in range(4):
            for k in range(2):
                gk = 4 + k  # global column index in B and C
                U_global = {(i, j): 1}
                V_global = {(j, gk): 1}
                W_global = {(gk, i): 1}
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

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition as fallback."""
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify the decomposition is correct by testing with random matrices.
    Uses the trace convention: W[k*n + i, r]
    """
    np.random.seed(12345)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    rank = U.shape[1]
    
    for r in range(rank):
        alpha_r = np.sum(A_test * U[:, r].reshape(n, m))
        beta_r = np.sum(B_test * V[:, r].reshape(m, p))
        # W[:, r] has shape (p*n,), indexed as W[k*n + i]
        # We need to add alpha_r * beta_r * W[k*n+i] to C[i,k]
        W_vec = W[:, r].reshape(p, n)  # W_vec[k, i] = W[k*n + i]
        C_computed += alpha_r * beta_r * W_vec.T  # C[i,k] += alpha * beta * W[k,i]
    
    return np.allclose(C_computed, C_expected, atol=1e-10)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin within 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Try rank-81 decomposition first
    try:
        U, V, W = _build_rank81_decomposition(n, m, p)
        
        if _verify_decomposition(U, V, W, n, m, p):
            if time.time() < deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If verification fails, fall through to schoolbook
    except Exception:
        pass
    
    # Fallback: schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
