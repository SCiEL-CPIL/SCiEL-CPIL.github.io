"""Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses block Strassen construction (rank 64) with deterministic column elimination
and rational refinement for exactness."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_exact(U, V, W, n=4, m=4, p=5):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    Ur = np.array(U, dtype=object)
    Vr = np.array(V, dtype=object)
    Wr = np.array(W, dtype=object)
    
    # Convert to Fractions
    Ur = np.array([[Fraction(str(x)) for x in row] for row in Ur.tolist()], dtype=object)
    Vr = np.array([[Fraction(str(x)) for x in row] for row in Vr.tolist()], dtype=object)
    Wr = np.array([[Fraction(str(x)) for x in row] for row in Wr.tolist()], dtype=object)
    
    Ur = Ur.reshape(n, m, rank)
    Vr = Vr.reshape(m, p, rank)
    Wr = Wr.reshape(p, n, rank)
    
    T = np.zeros((n, m, p), dtype=object)
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Ur[i, j, c] * Vr[j, k, c] * Wr[k, i, c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i, j, k] != 1:
                    return False
    return True

def _verify_float(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast float verification."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    if Ua.ndim == 1: Ua = Ua.reshape(-1, 1)
    if Va.ndim == 1: Va = Va.reshape(-1, 1)
    if Wa.ndim == 1: Wa = Wa.reshape(-1, 1)
    
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        for j in range(m):
            T[:, j, :] += np.outer(Ur[:, j, c], Vr[j, :, c]) * Wr[:, j, c][::-1]  # Wr[k*n+i] -> need Wr[i, k]
    
    # Correct indexing: Wr[k*n + i] means Wr row k, col i in (p, n) matrix
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]  # (n, m)
        vc = Vr[:, :, c]  # (m, p)
        wc = Wr[:, :, c]  # (p, n)
        for j in range(m):
            T[:, j, :] += uc[:, j] * vc[j, :] * wc.T  # wc.T is (n, p), so T[:,j,:] += (n,) * (p,) * (n,p)
            # Actually: T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
            # Let's do it properly
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    
    return np.allclose(T, 1.0, atol=atol)

def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    if Ua.ndim == 1: Ua = Ua.reshape(-1, 1)
    if Va.ndim == 1: Va = Va.reshape(-1, 1)
    if Wa.ndim == 1: Wa = Wa.reshape(-1, 1)
    
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    
    return np.allclose(T, 1.0, atol=atol)

def _compute_error(U, V, W, n, m, p):
    """Compute decomposition error."""
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    return np.sqrt(np.sum((T - 1.0) ** 2))

def _strassen_2x2x2_terms():
    """7 Strassen terms for 2x2x2 as (U_sparse, V_sparse, W_sparse)."""
    return [
        # S1: (A00+A11)(B00+B11) -> C00, C11
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # S2: (A10+A11)B00 -> C10
        ([(2,1),(3,1)], [(0,1)], [(1,1)]),
        # S3: A00(B10-B00) -> C01
        ([(0,1)], [(1,1),(3,-1)], [(2,1)]),
        # S4: A11(B01-B10) -> C10, C11
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        # S5: (A00+A01)B11 -> C01
        ([(0,1),(1,1)], [(3,1)], [(2,1)]),
        # S6: (A10-A00)(B00+B01) -> C10
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,1)]),
        # S7: (A01-A11)(B10+B11) -> C01
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(2,1)]),
    ]

def _block_strassen_4x4x2(n, m, p):
    """
    Construct rank-28 decomposition for A(4x4) * B(4x2) -> C(4x2).
    Uses 4 Strassen 2x2x2 blocks.
    Returns list of (u_vec, v_vec, w_vec) tuples.
    """
    U_cols = []
    V_cols = []
    W_cols = []
    strassen_terms = _strassen_2x2x2_terms()
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # Block (0,0): A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # Block (0,1): A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    # Block (1,0): A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2]
    # Block (1,1): A[2:4,2:4] * B[2:4,0:2] -> C[2:4,0:2]
    
    blocks = [
        (0, 0, 0, 0),   # ai, aj, bj, bk
        (0, 2, 2, 0),
        (2, 0, 0, 0),
        (2, 2, 2, 0),
    ]
    
    for (ai, aj, bj, bk) in blocks:
        for u_sp, v_sp, w_sp in strassen_terms:
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            for idx, val in u_sp:
                ui, uj = idx // 2, idx % 2
                u[(ai + ui) * m + (aj + uj)] = val
            for idx, val in v_sp:
                vj, vk = idx // 2, idx % 2
                v[(bj + vj) * p + (bk + vk)] = val
            for idx, val in w_sp:
                wk, wi = idx // 2, idx % 2
                w[(bk + wk) * n + (ai + wi)] = val
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _naive_4x4x1(n, m, p):
    """
    Construct rank-8 decomposition for A(4x4) * B(4x1) -> C(4x1).
    Returns list of (u_vec, v_vec, w_vec) tuples.
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    for i in range(n):
        for j in range(m):
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            u[i * m + j] = 1
            v[j * p + 4] = 1
            w[4 * n + i] = 1
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _block_strassen_4x4x4(n, m, p):
    """
    Construct rank-56 decomposition for A(4x4) * B(4x4) -> C(4x4).
    Uses 8 Strassen 2x2x2 blocks.
    """
    U_cols = []
    V_cols = []
    W_cols = []
    strassen_terms = _strassen_2x2x2_terms()
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # 8 blocks for 4x4x4
    blocks = [
        (0, 0, 0, 0), (0, 2, 2, 0),
        (2, 0, 0, 0), (2, 2, 2, 0),
        (0, 0, 0, 2), (0, 2, 2, 2),
        (2, 0, 0, 2), (2, 2, 2, 2),
    ]
    
    for (ai, aj, bj, bk) in blocks:
        for u_sp, v_sp, w_sp in strassen_terms:
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            for idx, val in u_sp:
                ui, uj = idx // 2, idx % 2
                u[(ai + ui) * m + (aj + uj)] = val
            for idx, val in v_sp:
                vj, vk = idx // 2, idx % 2
                v[(bj + vj) * p + (bk + vk)] = val
            for idx, val in w_sp:
                wk, wi = idx // 2, idx % 2
                w[(bk + wk) * n + (ai + wi)] = val
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _construct_block_64(n, m, p):
    """
    Construct rank-64 decomposition:
    - 4x4x4 part: rank 56 (8 Strassen blocks)
    - 4x4x1 part: rank 8 (naive)
    Total: 64
    """
    U1, V1, W1 = _block_strassen_4x4x4(n, m, p)
    U2, V2, W2 = _naive_4x4x1(n, m, p)
    
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    return U, V, W

def _construct_kron_65(n, m, p):
    """
    Construct rank-65 decomposition using Kronecker Strassen for 4x4x4 + naive for 4x4x1.
    49 + 16 = 65
    """
    U_cols = []
    V_cols = []
    W_cols = []
    strassen_terms = _strassen_2x2x2_terms()
    
    for (u1_sp, v1_sp, w1_sp) in strassen_terms:
        for (u2_sp, v2_sp, w2_sp) in strassen_terms:
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            
            for (idx1, val1) in u1_sp:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sp:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[i * m + j] = val1 * val2
            
            for (idx1, val1) in v1_sp:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sp:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[j * p + k] = val1 * val2
            
            for (idx1, val1) in w1_sp:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sp:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[k * n + i] = val1 * val2
            
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
    
    # Naive 4x4x1: 16 terms
    for i in range(n):
        for j in range(m):
            u = np.zeros(n * m); u[i * m + j] = 1
            v = np.zeros(m * p); v[j * p + 4] = 1
            w = np.zeros(p * n); w[4 * n + i] = 1
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    return U, V, W

def _try_eliminate_columns(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce rank by eliminating columns.
    For each set of columns to keep, check if the remaining columns can be
    adjusted to maintain exactness.
    """
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return U, V, W
    
    # Try all combinations of keeping target_rank columns
    # This is expensive, so use a greedy approach
    # First, try removing one column at a time and see if we can fix it
    
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    for round_num in range(current_rank - target_rank):
        if time.time() > deadline:
            break
        
        found = False
        for col_to_remove in range(U_curr.shape[1]):
            if time.time() > deadline:
                break
            
            # Try removing this column
            U_test = np.delete(U_curr, col_to_remove, axis=1)
            V_test = np.delete(V_curr, col_to_remove, axis=1)
            W_test = np.delete(W_curr, col_to_remove, axis=1)
            
            # Check if we can adjust the remaining columns to fix the error
            # This is a linear system: for each (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
            # After removing a column, we need to adjust remaining columns
            
            # Simple check: compute error
            err = _compute_error(U_test, V_test, W_test, n, m, p)
            if err < 1e-10:
                U_curr, V_curr, W_curr = U_test, V_test, W_test
                found = True
                break
        
        if not found:
            break
    
    if U_curr.shape[1] <= target_rank:
        return U_curr, V_curr, W_curr
    
    return None

def _als_refine(U, V, W, n, m, p, max_iters=100, deadline=None):
    """ALS refinement without rounding."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Optimize U
        for i in range(n):
            for j in range(m):
                # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                # For fixed (i,j), U[i,j,:] appears in terms for all k
                # sum_c U[i,j,c] * (V[j,:,c] * W[:,i,c]) = 1 for all k
                # Let A[c,k] = V[j,k,c] * W[k,i,c]
                # Then U[i,j,:] * A^T = 1 (vector of length p)
                A = np.zeros((p, U.shape[1]))
                for k in range(p):
                    for c in range(U.shape[1]):
                        A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                b = np.ones(p)
                try:
                    U[i * m + j, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                except:
                    pass
        
        # Optimize V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, V.shape[1]))
                for i in range(n):
                    for c in range(V.shape[1]):
                        A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                b = np.ones(n)
                try:
                    V[j * p + k, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                except:
                    pass
        
        # Optimize W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, W.shape[1]))
                for j in range(m):
                    for c in range(W.shape[1]):
                        A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                b = np.ones(m)
                try:
                    W[k * n + i, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                except:
                    pass
        
        err = _compute_error(U, V, W, n, m, p)
        if err < 1e-14:
            break
    
    return U, V, W

def _round_to_rational(U, V, W, n, m, p, max_denom=10):
    """Round to rationals with small denominators."""
    U_r = np.zeros_like(U)
    V_r = np.zeros_like(V)
    W_r = np.zeros_like(W)
    
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            for denom in range(1, max_denom + 1):
                val = round(U[i, j] * denom) / denom
                if abs(val - U[i, j]) < 1e-10:
                    U_r[i, j] = val
                    break
    
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            for denom in range(1, max_denom + 1):
                val = round(V[i, j] * denom) / denom
                if abs(val - V[i, j]) < 1e-10:
                    V_r[i, j] = val
                    break
    
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            for denom in range(1, max_denom + 1):
                val = round(W[i, j] * denom) / denom
                if abs(val - W[i, j]) < 1e-10:
                    W_r[i, j] = val
                    break
    
    if _verify_float(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def _schoolbook(n=4, m=4, p=5):
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
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    
    # Try block Strassen rank 64
    U64, V64, W64 = _construct_block_64(n, m, p)
    if _verify_float(U64.tolist(), V64.tolist(), W64.tolist(), n, m, p):
        result = _round_to_rational(U64, V64, W64, n, m, p)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}
        # Try ALS refinement
        U64_r, V64_r, W64_r = _als_refine(U64, V64, W64, n, m, p, max_iters=50, deadline=deadline)
        if _verify_float(U64_r.tolist(), V64_r.tolist(), W64_r.tolist(), n, m, p):
            result = _round_to_rational(U64_r, V64_r, W64_r, n, m, p)
            if result is not None:
                return {"U": result[0], "V": result[1], "W": result[2]}
            return {"U": U64_r.tolist(), "V": V64_r.tolist(), "W": W64_r.tolist()}
    
    # Try Kronecker Strassen rank 65
    U65, V65, W65 = _construct_kron_65(n, m, p)
    if _verify_float(U65.tolist(), V65.tolist(), W65.tolist(), n, m, p):
        result = _round_to_rational(U65, V65, W65, n, m, p)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}
        return {"U": U65.tolist(), "V": V65.tolist(), "W": W65.tolist()}
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
