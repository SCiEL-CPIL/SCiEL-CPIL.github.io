"""Matrix multiplication 2x4x7 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 150s kill

    # Block decomposition: m=4 split into two 2-blocks, p=7 split into 2+2+2+1
    # Each 2x2x2 uses Strassen (rank 7), each 2x2x1 uses schoolbook (rank 4)
    # Total: 2 * (3*7 + 4) = 50

    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    m_blocks = [(0, 2), (2, 4)]
    k_blocks = [
        (0, 2, 'strassen'),
        (2, 4, 'strassen'),
        (4, 6, 'strassen'),
        (6, 7, 'school')
    ]
    
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    
    for j_start, j_end in m_blocks:
        for k_start, k_end, k_type in k_blocks:
            if k_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
            
            p_sub = k_end - k_start
            
            for i in range(n):
                for j_sub in range(j_end - j_start):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            
            for j_sub in range(j_end - j_start):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            
            col_offset += r_sub
    
    # Verify the rank-50 solution
    if _verify(U, V, W, n, m, p):
        # Try to reduce rank by finding linear dependencies
        if time.time() < deadline - 5:
            U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p, deadline)
            if U_red is not None:
                return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
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
        return {"U": U_fb, "V": V_fb, "W": W_fb}

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to find a lower-rank decomposition by identifying linear dependencies."""
    r = U.shape[1]
    
    # For each candidate column to eliminate, check if the remaining columns
    # can span the same "subspace" in the tensor product sense.
    # 
    # We need: for all (i,j,jp,k):
    #   sum_c U[i*m+j,c] * V[jp*p+k,c] * W[k*n+i,c] = delta_{j,jp}
    #
    # If we remove column c0, we need to find alpha_c for c != c0 such that:
    #   sum_{c != c0} alpha_c * U[i*m+j,c] * V[jp*p+k,c] * W[k*n+i,c] 
    #     = U[i*m+j,c0] * V[jp*p+k,c0] * W[k*n+i,c0]
    #
    # Then the new decomposition uses U_new[:,c] = U[:,c]*alpha_c, V_new=V, W_new=W
    
    idx = 0
    rows = []
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    rows.append((i, j, jp, k))
                    idx += 1
    
    n_constraints = len(rows)
    
    # Try columns in order of likelihood of being redundant
    # Schoolbook columns (last 8) are most likely to be redundant
    order = list(range(r - 1, r - 9, -1)) + list(range(r - 9, -1, -1))
    
    for c0 in order:
        if time.time() > deadline:
            return None, None, None
        
        # Build the system: A @ alpha = b
        # A[idx, col_idx] = U[i*m+j,c] * V[jp*p+k,c] * W[k*n+i,c] for c != c0
        # b[idx] = U[i*m+j,c0] * V[jp*p+k,c0] * W[k*n+i,c0]
        
        A = np.zeros((n_constraints, r - 1))
        b = np.zeros(n_constraints)
        
        col_idx = 0
        for c in range(r):
            if c == c0:
                continue
            for idx2, (i, j, jp, k) in enumerate(rows):
                A[idx2, col_idx] = U[i * m + j, c] * V[jp * p + k, c] * W[k * n + i, c]
            col_idx += 1
        
        for idx2, (i, j, jp, k) in enumerate(rows):
            b[idx2] = U[i * m + j, c0] * V[jp * p + k, c0] * W[k * n + i, c0]
        
        # Solve least squares
        try:
            alpha, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=1e-12)
        except Exception:
            continue
        
        # Check if the solution is accurate
        if len(residuals) > 0:
            if residuals[0] > 1e-9:
                continue
        
        # Verify the solution by checking the actual tensor identity
        U_new = np.zeros((n * m, r - 1))
        V_new = np.zeros((m * p, r - 1))
        W_new = np.zeros((p * n, r - 1))
        
        col_idx = 0
        for c in range(r):
            if c == c0:
                continue
            U_new[:, col_idx] = U[:, c] * alpha[col_idx]
            V_new[:, col_idx] = V[:, c]
            W_new[:, col_idx] = W[:, c]
            col_idx += 1
        
        # Verify
        if _verify(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, V_new, W_new
    
    return None, None, None

def _verify(U, V, W, n, m, p):
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

def _get_strassen_222():
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_school_221():
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

# EVOLVE-BLOCK-END
