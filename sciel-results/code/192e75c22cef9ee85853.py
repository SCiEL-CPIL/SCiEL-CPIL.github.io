"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _get_strassen_222():
    """
    Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7).
    """
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_school_221():
    """
    Returns U, V, W for 2x2x1 multiplication using schoolbook (Rank 4).
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

def _get_school_217():
    """
    Returns U, V, W for 2x1x7 multiplication using schoolbook (Rank 14).
    """
    n, m, p = 2, 1, 7
    r = 14
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        i = c // (m * p) # This logic is wrong for schoolbook
        # Schoolbook: each (i,j,k) is a separate term.
        # Here j=0 always.
        # c = i * p + k
        i = c // p
        k = c % p
        j = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
        
    return U, V, W

def _get_school_231():
    """
    Returns U, V, W for 2x3x1 multiplication using schoolbook (Rank 6).
    """
    n, m, p = 2, 3, 1
    r = 6
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        i = c // (m * p) # c = i * m + j
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

def _find_decomposition(n, m, p, rank, deadline, seed=42):
    """
    Try to find a valid decomposition with given rank using iterative refinement.
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.05
    stuck_count = 0
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Compute error and gradients
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
        
        # Compute gradients
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
                            
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    if best_err < 1e-9:
        # Try to snap to integers/half-integers
        for scale in [1, 2]:
            U_r = np.round(best_U * scale) / scale
            V_r = np.round(best_V * scale) / scale
            W_r = np.round(best_W * scale) / scale
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

    # Try to find a lower-rank decomposition using the 2x3x3 block idea
    # Rank 48: M_2,1,7 (14) + M_2,3,7 (34)
    # M_2,3,7 = M_2,3,3 (14) + M_2,3,3 (14) + M_2,3,1 (6)
    
    # Try to find M_2,3,3 with Rank 14
    found_233 = False
    U_233, V_233, W_233 = None, None, None
    
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        result = _find_decomposition(2, 3, 3, 14, deadline - 10, seed=seed)
        if result is not None:
            U_233, V_233, W_233 = result
            found_233 = True
            break
            
    if found_233:
        # Build the rank-48 solution
        total_rank_48 = 48
        U_new = np.zeros((n * m, total_rank_48))
        V_new = np.zeros((m * p, total_rank_48))
        W_new = np.zeros((p * n, total_rank_48))
        
        col_offset = 0
        
        # Part 1: M_2,1,7 (j=0)
        U_217, V_217, W_217 = _get_school_217()
        r_217 = 14
        for i in range(2):
            U_new[i * 4 + 0, col_offset:col_offset + r_217] = U_217[i * 1 + 0, :]
        for k in range(7):
            V_new[0 * 7 + k, col_offset:col_offset + r_217] = V_217[0 * 7 + k, :]
            W_new[k * 2 + i, col_offset:col_offset + r_217] = W_217[k * 2 + i, :]
        col_offset += r_217
        
        # Part 2: M_2,3,7 (j=1,2,3)
        # Split p into 3, 3, 1
        p_blocks_237 = [
            (0, 3, '233'),
            (3, 6, '233'),
            (6, 7, '231')
        ]
        
        U_231, V_231, W_231 = _get_school_231()
        
        for k_start, k_end, p_type in p_blocks_237:
            if p_type == '233':
                U_sub, V_sub, W_sub = U_233, V_233, W_233
                r_sub = 14
                p_sub = 3
            else:
                U_sub, V_sub, W_sub = U_231, V_231, W_231
                r_sub = 6
                p_sub = 1
                
            m_sub = 3 # j=1,2,3
            
            for i in range(2):
                for j_sub in range(m_sub):
                    j = 1 + j_sub
                    U_new[i * 4 + j, col_offset:col_offset + r_sub] = U_sub[i * 3 + j_sub, :]
            for j_sub in range(m_sub):
                j = 1 + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V_new[j * 7 + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(2):
                    W_new[k * 2 + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            col_offset += r_sub
            
        if _verify(U_new, V_new, W_new, n, m, p):
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

    return known_result

# EVOLVE-BLOCK-END
