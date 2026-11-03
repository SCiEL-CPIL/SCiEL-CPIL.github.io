"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _get_strassen_222():
    """
    Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Strassen terms M1..M7
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_school_221():
    """
    Returns U, V, W for 2x2x1 multiplication using schoolbook (Rank 4).
    U shape: (4, 4), V shape: (2, 4), W shape: (2, 4)
    Indices: U[i*2+j], V[j*1+k], W[k*2+i]
    """
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
        
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify if U, V, W satisfy the tensor identity exactly.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Identity: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
    """
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _compute_error_and_grad(U, V, W, n, m, p, rank):
    """Compute error tensor and gradients efficiently."""
    err = np.zeros((n, m, m, p))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        err[i, j, jp, k] += uc[i*m+j] * vc[jp*p+k] * wc[k*n+i]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                err[i, j, j, k] -= 1.0
    
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        e = err[i, j, jp, k]
                        dU[i*m+j, c] += e * vc[jp*p+k] * wc[k*n+i]
                        dV[jp*p+k, c] += e * uc[i*m+j] * wc[k*n+i]
                        dW[k*n+i, c] += e * uc[i*m+j] * vc[jp*p+k]
    return err, dU, dV, dW

def _try_random_decomposition(n, m, p, rank, deadline, seed=42):
    """Try to find a valid decomposition with given rank using iterative refinement."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.05
    stuck_count = 0
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        err, dU, dV, dW = _compute_error_and_grad(U, V, W, n, m, p, rank)
        cur_err = np.sum(err**2)
        if cur_err < best_err:
            best_err = cur_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            stuck_count = 0
        else:
            stuck_count += 1
        
        if best_err < 1e-12:
            break
        
        if stuck_count > 50:
            lr *= 0.7
            stuck_count = 0
            if lr < 1e-6:
                break
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    if best_err < 1e-9:
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2
        if _verify(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
        U_r = np.round(best_U)
        V_r = np.round(best_V)
        W_r = np.round(best_W)
        if _verify(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # First, build the known rank-50 decomposition
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    m_blocks = [(0, 2), (2, 4)]
    p_blocks = [
        (0, 2, 'strassen'),
        (2, 4, 'strassen'),
        (4, 6, 'strassen'),
        (6, 7, 'school')
    ]
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    
    for j_start, j_end in m_blocks:
        for k_start, k_end, p_type in p_blocks:
            if p_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
            
            p_sub = k_end - k_start
            m_sub = j_end - j_start
            
            for i in range(n):
                for j_sub in range(m_sub):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            for j_sub in range(m_sub):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            col_offset += r_sub
    
    # Verify the rank-50 decomposition
    if _verify(U, V, W, n, m, p):
        known_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        r = n * m * p
        U_fb = [[0] * r for _ in range(n * m)]
        V_fb = [[0] * r for _ in range(m * p)]
        W_fb = [[0] * r for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        known_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try to find a lower-rank decomposition with multiple seeds
    for rank in range(49, 44, -1):
        for seed_offset in range(5):
            if time.time() > deadline - 10:
                break
            seed = rank * 100 + 42 + seed_offset * 17
            result = _try_random_decomposition(n, m, p, rank, deadline - 5, seed=seed)
            if result is not None:
                U_r, V_r, W_r = result
                if _verify(U_r, V_r, W_r, n, m, p):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return known_result

# EVOLVE-BLOCK-END
