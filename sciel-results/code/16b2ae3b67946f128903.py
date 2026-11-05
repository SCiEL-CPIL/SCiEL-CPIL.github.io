import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _is_half_integer(M, tol=1e-8):
    """Check if all entries of M are half-integers."""
    M2 = np.array(M, dtype=float) * 2
    return np.allclose(M2, np.round(M2), atol=tol)

def _build_strassen_rank28():
    """
    Build the known rank-28 Strassen-based decomposition.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    
    # Part 2: A1 B13 (j=0,1, k=4) - 2 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    
    # Part 3: Strassen for (j=0,1, k=0,1) - 7 terms
    U[0][r] = 1; U[4][r] = 1; V[0][r] = 1; V[6][r] = 1; W[0][r] = 1; W[3][r] = 1; r += 1
    U[3][r] = 1; U[4][r] = 1; V[0][r] = 1; W[1][r] = 1; W[3][r] = -1; r += 1
    U[0][r] = 1; V[1][r] = 1; V[6][r] = -1; W[2][r] = 1; W[3][r] = 1; r += 1
    U[4][r] = 1; V[5][r] = 1; V[0][r] = -1; W[0][r] = 1; W[1][r] = 1; r += 1
    U[0][r] = 1; U[1][r] = 1; V[6][r] = 1; W[0][r] = -1; W[2][r] = 1; r += 1
    U[3][r] = 1; U[0][r] = -1; V[0][r] = 1; V[1][r] = 1; W[3][r] = 1; r += 1
    U[1][r] = 1; U[4][r] = -1; V[5][r] = 1; V[6][r] = 1; W[0][r] = 1; r += 1
    
    # Part 4: Strassen for (j=0,1, k=2,3) - 7 terms
    U[0][r] = 1; U[4][r] = 1; V[2][r] = 1; V[8][r] = 1; W[4][r] = 1; W[7][r] = 1; r += 1
    U[3][r] = 1; U[4][r] = 1; V[2][r] = 1; W[5][r] = 1; W[7][r] = -1; r += 1
    U[0][r] = 1; V[3][r] = 1; V[8][r] = -1; W[6][r] = 1; W[7][r] = 1; r += 1
    U[4][r] = 1; V[7][r] = 1; V[2][r] = -1; W[4][r] = 1; W[5][r] = 1; r += 1
    U[0][r] = 1; U[1][r] = 1; V[8][r] = 1; W[4][r] = -1; W[6][r] = 1; r += 1
    U[3][r] = 1; U[0][r] = -1; V[2][r] = 1; V[3][r] = 1; W[7][r] = 1; r += 1
    U[1][r] = 1; U[4][r] = -1; V[7][r] = 1; V[8][r] = 1; W[4][r] = 1; r += 1
    
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    return U, V, W, rank

def _try_rank_reduction(U, V, W, rank, n, m, p, deadline):
    """
    Attempt to reduce rank by finding linear dependencies.
    For each term t, check if it can be expressed as a linear combination of others.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for t_remove in range(rank):
        if time.time() > deadline - 20:
            break
        
        n_unknowns = rank - 1
        n_equations = n * m * p
        
        A_mat = np.zeros((n_equations, n_unknowns))
        b_vec = np.zeros(n_equations)
        
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    
                    b_vec[eq] = U_arr[a, t_remove] * V_arr[b, t_remove] * W_arr[c, t_remove]
                    
                    unknown_idx = 0
                    for r in range(rank):
                        if r == t_remove:
                            continue
                        A_mat[eq, unknown_idx] = U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
                        unknown_idx += 1
                    
                    eq += 1
        
        try:
            alpha, residuals, rk, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            continue
        
        if residuals is None or residuals[0] > 1e-6:
            continue
        
        alpha_rounded = np.round(alpha * 2) / 2
        if not np.allclose(alpha, alpha_rounded, atol=1e-6):
            continue
        
        new_rank = rank - 1
        new_U = np.zeros((n * m, new_rank))
        new_V = np.zeros((m * p, new_rank))
        new_W = np.zeros((p * n, new_rank))
        
        new_idx = 0
        for r in range(rank):
            if r == t_remove:
                continue
            new_U[:, new_idx] = U_arr[:, r] - alpha[new_idx] * U_arr[:, t_remove]
            new_V[:, new_idx] = V_arr[:, r] - alpha[new_idx] * V_arr[:, t_remove]
            new_W[:, new_idx] = W_arr[:, r] - alpha[new_idx] * W_arr[:, t_remove]
            new_idx += 1
        
        new_U = np.round(new_U * 2) / 2
        new_V = np.round(new_V * 2) / 2
        new_W = np.round(new_W * 2) / 2
        
        if _verify_decomposition(new_U.tolist(), new_V.tolist(), new_W.tolist(), n, m, p):
            if _is_half_integer(new_U) and _is_half_integer(new_V) and _is_half_integer(new_W):
                return new_U.tolist(), new_V.tolist(), new_W.tolist(), new_rank
    
    return None

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # Build the rank-28 decomposition
    result = _build_strassen_rank28()
    if result is None:
        return _schoolbook(n, m, p)
    
    U, V, W, rank = result
    
    # Try to reduce the rank
    reduced = _try_rank_reduction(U, V, W, rank, n, m, p, deadline)
    if reduced is not None:
        U_new, V_new, W_new, new_rank = reduced
        return {"U": U_new, "V": V_new, "W": W_new}
    
    return {"U": U, "V": V, "W": W}
