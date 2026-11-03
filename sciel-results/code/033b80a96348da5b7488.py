"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_decomposition(n, m, p):
    """
    Builds a rank-81 decomposition for 4x4 x 4x6 matrix multiplication.
    Uses tensor product of Strassen 2x2x2 (rank 7) for the 4x4x4 part (rank 49)
    and Schoolbook for the 4x4x2 part (rank 32).
    """
    # 1. Define Strassen 2x2x2 terms as sparse dicts
    # Format: (U_dict, V_dict, W_dict)
    # Indices are (row, col) for 2x2 matrices
    strassen_terms_2x2 = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    
    rank_terms = []
    
    # 2. Generate 49 terms for 4x4x4 part via tensor product
    # Global indices: i, j in 0..3 for A; j, k in 0..3 for B; i, k in 0..3 for C
    # Local indices: i1, i2, j1, j2, k1, k2 in 0..1
    # Global i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    
    for s in range(7):
        u_s, v_s, w_s = strassen_terms_2x2[s]
        for t in range(7):
            u_t, v_t, w_t = strassen_terms_2x2[t]
            
            U_global = {}
            V_global = {}
            W_global = {}
            
            for (i1, j1), cu in u_s.items():
                for (i2, j2), cu2 in u_t.items():
                    gi = 2 * i1 + i2
                    gj = 2 * j1 + j2
                    coeff = cu * cu2
                    key = (gi, gj)
                    U_global[key] = U_global.get(key, 0) + coeff
                    
            for (j1, k1), cv in v_s.items():
                for (j2, k2), cv2 in v_t.items():
                    gj = 2 * j1 + j2
                    gk = 2 * k1 + k2
                    coeff = cv * cv2
                    key = (gj, gk)
                    V_global[key] = V_global.get(key, 0) + coeff
                    
            for (i1, k1), cw in w_s.items():
                for (i2, k2), cw2 in w_t.items():
                    gi = 2 * i1 + i2
                    gk = 2 * k1 + k2
                    coeff = cw * cw2
                    key = (gk, gi) # W index is (k, i)
                    W_global[key] = W_global.get(key, 0) + coeff
                    
            rank_terms.append((U_global, V_global, W_global))
            
    # 3. Add 28 terms for 4x4x2 part (4 Strassen 2x2x2 blocks)
    # We decompose the 4x4x2 multiplication into four 2x2x2 multiplications:
    # C_11 = A_11 B_11 + A_12 B_21
    # C_21 = A_21 B_11 + A_22 B_21
    # Where A_ij are 2x2 blocks of A, B_j1 are 2x2 blocks of B (B is 4x2, so B_11 is top 2 rows, B_21 is bottom 2 rows),
    # and C_i1 are 2x2 blocks of C.
    
    # Define block offsets
    # A blocks:
    # A_11: rows 0-1, cols 0-1
    # A_12: rows 0-1, cols 2-3
    # A_21: rows 2-3, cols 0-1
    # A_22: rows 2-3, cols 2-3
    # B blocks (4x2):
    # B_11: rows 0-1, cols 0-1
    # B_21: rows 2-3, cols 0-1
    # C blocks (4x2):
    # C_11: rows 0-1, cols 0-1
    # C_21: rows 2-3, cols 0-1
    
    blocks = [
        # (A_row_off, A_col_off, B_row_off, C_row_off)
        (0, 0, 0, 0), # A_11, B_11 -> C_11
        (0, 2, 2, 0), # A_12, B_21 -> C_11
        (2, 0, 0, 2), # A_21, B_11 -> C_21
        (2, 2, 2, 2), # A_22, B_21 -> C_21
    ]
    
    for a_ro, a_co, b_ro, c_ro in blocks:
        for s in range(7):
            u_s, v_s, w_s = strassen_terms_2x2[s]
            
            U_global = {}
            V_global = {}
            W_global = {}
            
            for (i1, j1), cu in u_s.items():
                gi = a_ro + i1
                gj = a_co + j1
                U_global[(gi, gj)] = U_global.get((gi, gj), 0) + cu
                
            for (j1, k1), cv in v_s.items():
                gj = b_ro + j1
                gk = k1
                V_global[(gj, gk)] = V_global.get((gj, gk), 0) + cv
                
            for (i1, k1), cw in w_s.items():
                gi = c_ro + i1
                gk = k1
                # W index is (k, i)
                W_global[(gk, gi)] = W_global.get((gk, gi), 0) + cw
                
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

def _remove_redundant(U, V, W, n, m, p, tol=1e-10):
    """
    Attempt to reduce rank by finding linearly dependent terms.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    # Identify columns that are all zeros
    keep_mask = [True] * rank
    for r in range(rank):
        if np.allclose(U[:, r], 0, atol=tol) or np.allclose(V[:, r], 0, atol=tol) or np.allclose(W[:, r], 0, atol=tol):
            keep_mask[r] = False
    
    # Check for scalar multiples
    for r in range(rank):
        if not keep_mask[r]:
            continue
        for s in range(r + 1, rank):
            if not keep_mask[s]:
                continue
            if np.linalg.norm(V[:, r]) > tol:
                alpha = np.dot(V[:, s], V[:, r]) / np.dot(V[:, r], V[:, r])
                if np.allclose(V[:, s], alpha * V[:, r], atol=tol) and \
                   np.allclose(W[:, s], alpha * W[:, r], atol=tol) and \
                   np.allclose(U[:, s], alpha * U[:, r], atol=tol):
                    keep_mask[s] = False
    
    if any(not k for k in keep_mask):
        keep = [r for r in range(rank) if keep_mask[r]]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 110.0 # Safety margin
    
    # Build decomposition
    U, V, W = _build_decomposition(n, m, p)
    
    # Verify
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
    
    # Try to reduce rank
    if time.time() < deadline - 5.0:
        U_red, V_red, W_red = _remove_redundant(U, V, W, n, m, p)
        if _verify(U_red, V_red, W_red, n, m, p):
            U, V, W = U_red, V_red, W_red
    
    # Final time check
    if time.time() > deadline:
        # Return best we have
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
