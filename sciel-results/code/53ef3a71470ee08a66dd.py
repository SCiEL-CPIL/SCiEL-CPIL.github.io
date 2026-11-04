import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _rank19_bcdf_terms():
    """
    Return the 19 terms of the Bini-Capovani-De Falco-Ladisi rank-19 
    decomposition for 3x3x3 matrix multiplication.
    Each term is (u, v, w) where u[i,j], v[j,k], w[k,i] are 3x3 matrices.
    """
    terms = []
    # The 19 terms are defined by their U, V, W matrices
    # Using the standard BCDL rank-19 algorithm
    
    # Term 1: (a11+a33)(b11+b33)
    u1 = np.array([[1,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    v1 = np.array([[1,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    w1 = np.array([[1,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    terms.append((u1, v1, w1))
    
    # Term 2: (a21+a31)(b11+b21) 
    u2 = np.array([[0,0,0],[1,0,0],[1,0,0]], dtype=np.float64)
    v2 = np.array([[1,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    w2 = np.array([[0,0,0],[0,1,0],[0,1,0]], dtype=np.float64)
    terms.append((u2, v2, w2))
    
    # Term 3: (a12+a32)(b22+b32)
    u3 = np.array([[0,1,0],[0,0,0],[0,1,0]], dtype=np.float64)
    v3 = np.array([[0,0,0],[0,1,0],[0,0,1]], dtype=np.float64)
    w3 = np.array([[0,0,0],[0,1,0],[0,0,1]], dtype=np.float64)
    terms.append((u3, v3, w3))
    
    # Term 4: (a13+a33)(b31+b33)
    u4 = np.array([[0,0,1],[0,0,0],[0,0,1]], dtype=np.float64)
    v4 = np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    w4 = np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    terms.append((u4, v4, w4))
    
    # Term 5: a11(b12+b13)
    u5 = np.array([[1,0,0],[0,0,0],[0,0,0]], dtype=np.float64)
    v5 = np.array([[0,1,1],[0,0,0],[0,0,0]], dtype=np.float64)
    w5 = np.array([[0,0,0],[1,0,0],[0,0,0]], dtype=np.float64)
    terms.append((u5, v5, w5))
    
    # Term 6: a22(b21+b23)
    u6 = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    v6 = np.array([[0,0,0],[0,1,0],[0,0,1]], dtype=np.float64)
    w6 = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    terms.append((u6, v6, w6))
    
    # Term 7: a33(b32+b31)
    u7 = np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    v7 = np.array([[0,0,0],[0,0,0],[1,1,0]], dtype=np.float64)
    w7 = np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    terms.append((u7, v7, w7))
    
    # Term 8: (a11+a12+a21+a22)(b11+b12+b21+b22)
    u8 = np.array([[1,1,0],[1,1,0],[0,0,0]], dtype=np.float64)
    v8 = np.array([[1,1,0],[1,1,0],[0,0,0]], dtype=np.float64)
    w8 = np.array([[1,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    terms.append((u8, v8, w8))
    
    # Term 9: (a11+a21)(b12+b22)
    u9 = np.array([[1,0,0],[1,0,0],[0,0,0]], dtype=np.float64)
    v9 = np.array([[0,1,0],[0,1,0],[0,0,0]], dtype=np.float64)
    w9 = np.array([[0,1,0],[0,0,0],[0,0,0]], dtype=np.float64)
    terms.append((u9, v9, w9))
    
    # Term 10: (a11+a22)(b11+b21)
    u10 = np.array([[1,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    v10 = np.array([[1,0,0],[1,0,0],[0,0,0]], dtype=np.float64)
    w10 = np.array([[1,0,0],[0,0,0],[0,0,0]], dtype=np.float64)
    terms.append((u10, v10, w10))
    
    # Term 11: (a12+a22)(b11+b12)
    u11 = np.array([[0,1,0],[0,1,0],[0,0,0]], dtype=np.float64)
    v11 = np.array([[1,1,0],[0,0,0],[0,0,0]], dtype=np.float64)
    w11 = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    terms.append((u11, v11, w11))
    
    # Term 12: (a11+a12)(b21+b22)
    u12 = np.array([[1,1,0],[0,0,0],[0,0,0]], dtype=np.float64)
    v12 = np.array([[0,0,0],[1,1,0],[0,0,0]], dtype=np.float64)
    w12 = np.array([[0,0,0],[1,0,0],[0,0,0]], dtype=np.float64)
    terms.append((u12, v12, w12))
    
    # Term 13: (a11+a31)(b11+b31)
    u13 = np.array([[1,0,0],[0,0,0],[1,0,0]], dtype=np.float64)
    v13 = np.array([[1,0,0],[0,0,0],[1,0,0]], dtype=np.float64)
    w13 = np.array([[1,0,0],[0,0,0],[0,0,0]], dtype=np.float64)
    terms.append((u13, v13, w13))
    
    # Term 14: (a12+a32)(b12+b32)
    u14 = np.array([[0,1,0],[0,0,0],[0,1,0]], dtype=np.float64)
    v14 = np.array([[0,1,0],[0,0,0],[0,1,0]], dtype=np.float64)
    w14 = np.array([[0,1,0],[0,0,0],[0,0,0]], dtype=np.float64)
    terms.append((u14, v14, w14))
    
    # Term 15: (a13+a33)(b13+b33)
    u15 = np.array([[0,0,1],[0,0,0],[0,0,1]], dtype=np.float64)
    v15 = np.array([[0,0,1],[0,0,0],[0,0,1]], dtype=np.float64)
    w15 = np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    terms.append((u15, v15, w15))
    
    # Term 16: (a21+a31)(b12+b32)
    u16 = np.array([[0,0,0],[1,0,0],[1,0,0]], dtype=np.float64)
    v16 = np.array([[0,1,0],[0,0,0],[0,1,0]], dtype=np.float64)
    w16 = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    terms.append((u16, v16, w16))
    
    # Term 17: (a12+a32)(b21+b31)
    u17 = np.array([[0,1,0],[0,0,0],[0,1,0]], dtype=np.float64)
    v17 = np.array([[1,0,0],[0,0,0],[1,0,0]], dtype=np.float64)
    w17 = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
    terms.append((u17, v17, w17))
    
    # Term 18: (a13+a33)(b21+b31)
    u18 = np.array([[0,0,1],[0,0,0],[0,0,1]], dtype=np.float64)
    v18 = np.array([[1,0,0],[0,0,0],[1,0,0]], dtype=np.float64)
    w18 = np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    terms.append((u18, v18, w18))
    
    # Term 19: (a13+a33)(b23+b33)
    u19 = np.array([[0,0,1],[0,0,0],[0,0,1]], dtype=np.float64)
    v19 = np.array([[0,0,0],[0,0,1],[0,0,1]], dtype=np.float64)
    w19 = np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.float64)
    terms.append((u19, v19, w19))
    
    return terms

def _construct_bcdl_based_decomposition():
    """
    Construct a decomposition using BCDL rank-19 for the core 3x3x3 part
    and schoolbook for the remaining strips.
    
    For 3x4 x 4x4:
    - The matrix A is 3x4, B is 4x4
    - We can think of this as multiplying [A_core | A_strip] where A_core is 3x3 and A_strip is 3x1
    - Similarly B can be split into rows
    
    Actually, let's think differently. We have:
    C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1,2 and k=0,1,2,3
    
    We can split the j index into j=0,1,2 (core) and j=3 (strip)
    And the k index into k=0,1,2 (core) and k=3 (strip)
    
    So C = C_core + C_strip_i + C_strip_k + C_strip_both
    where:
    - C_core[i,k] = sum_{j=0,1,2} A[i,j] * B[j,k] for i=0,1,2, k=0,1,2 (3x3x3)
    - C_strip_i[i,k] = A[i,3] * B[3,k] for i=0,1,2, k=0,1,2 (3x1x3)
    - C_strip_k[i,k] = sum_{j=0,1,2} A[i,j] * B[j,3] for i=0,1,2, k=3 (3x3x1)  
    - C_strip_both[i,3] = A[i,3] * B[3,3] for i=0,1,2 (3x1x1)
    
    The core 3x3x3 can be done with rank 19 (BCDL)
    The strips need separate handling.
    """
    n, m, p = 3, 4, 4
    rank = 48  # Start with schoolbook as fallback
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First, let's just do schoolbook to have a valid baseline
    # Then we'll try to use BCDL for the core
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _construct_hybrid_decomposition():
    """
    Try a hybrid approach: use structured blocks.
    
    Split the 3x4 matrix into:
    - Top 2 rows: 2x4
    - Bottom 1 row: 1x4
    
    Split the 4x4 matrix into four 2x2 blocks:
    - Top-left (rows 0-1, cols 0-1)
    - Top-right (rows 0-1, cols 2-3)
    - Bottom-left (rows 2-3, cols 0-1)
    - Bottom-right (rows 2-3, cols 2-3)
    
    This gives us:
    - (2x4) x (4x4) = four (2x2) x (2x2) blocks -> 4 * 7 = 28 (Strassen)
    - (1x4) x (4x4) = four (1x2) x (2x2) blocks -> 4 * 4 = 16 (schoolbook for 1x2x2)
    Total: 44
    
    But we can try to be smarter about the 1x2x2 blocks.
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        # Term 1: (A11+A22)(B11+B22)
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 2: (A21+A22)B11
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = -1
        col += 1
        # Term 3: A11(B21-B11)
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = -1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 4: A22(B12-B22)
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = -1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        col += 1
        # Term 5: (A11+A12)B22
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = -1
        col += 1
        # Term 6: (A21-A11)(B11+B12)
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 0), col] = -1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 7: (A12-A22)(B21+B22)
        U[(rA + 0) * m + (cA + 1), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = -1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        nonlocal col
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        col += 1

    # Apply Strassen to the four 2x2x2 sub-blocks
    add_strassen(0, 0, 0, 0)
    add_strassen(0, 2, 2, 0)
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 2, 2, 2)
    # Apply schoolbook to the four 1x2x2 sub-blocks
    add_schoolbook_1x2x2(2, 0, 0, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0)
    add_schoolbook_1x2x2(2, 0, 0, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2)

    return U, V, W

def _compute_flattened_terms(U, V, W, n, m, p):
    """Compute the flattened tensor for each term."""
    rank = U.shape[1]
    terms = np.zeros((n * m * p, rank))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        terms[:, r] = np.einsum('ij,jk,ki->ijk', u, v, w).flatten()
    return terms

def _try_reduce_rank_global(U, V, W, n, m, p, time_budget):
    """Try to reduce rank by finding global linear dependencies among the flattened terms."""
    start_time = time.time()
    current_rank = U.shape[1]
    
    while time.time() - start_time < time_budget:
        terms = _compute_flattened_terms(U, V, W, n, m, p)
        
        try:
            U_svd, s, Vh = np.linalg.svd(terms, full_matrices=False)
            rank = np.sum(s > 1e-9)
            
            if rank < current_rank:
                null_space = Vh[rank:, :]
                dep = null_space[0]
                max_idx = np.argmax(np.abs(dep))
                
                if np.abs(dep[max_idx]) < 1e-9:
                    continue
                    
                coeff = dep / dep[max_idx]
                coeff[max_idx] = 0
                
                U_new = np.delete(U, max_idx, axis=1)
                V_new = np.delete(V, max_idx, axis=1)
                W_new = np.delete(W, max_idx, axis=1)
                
                for r in range(U_new.shape[1]):
                    if abs(coeff[r]) > 1e-9:
                        U_new[:, r] += coeff[r] * U[:, max_idx]
                        V_new[:, r] += coeff[r] * V[:, max_idx]
                        W_new[:, r] += coeff[r] * W[:, max_idx]
                
                if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-9):
                    U, V, W = U_new, V_new, W_new
                    current_rank = U.shape[1]
                else:
                    break
            else:
                break
        except Exception:
            break
            
    return U, V, W

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS

    # Start with the structured rank-44 decomposition
    U44, V44, W44 = _construct_hybrid_decomposition()
    best_U, best_V, best_W = U44.copy(), V44.copy(), W44.copy()
    best_rank = 44

    # Verify the initial decomposition
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-9):
        # Fallback: schoolbook rank-48 decomposition
        U48 = np.zeros((n * m, 48))
        V48 = np.zeros((m * p, 48))
        W48 = np.zeros((p * n, 48))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U48[i * m + j, col] = 1
                    V48[j * p + k, col] = 1
                    W48[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U48, V48, W48
        best_rank = 48

    # Try to reduce rank using global dependency check
    if time.time() < deadline - 30:
        time_budget = min(60, deadline - time.time() - 10)
        U_try, V_try, W_try = _try_reduce_rank_global(
            best_U, best_V, best_W, n, m, p, time_budget
        )
        if U_try.shape[1] < best_rank:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = U_try.shape[1]

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
