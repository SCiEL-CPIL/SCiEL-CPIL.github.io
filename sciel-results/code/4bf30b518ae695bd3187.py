import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
_U_cache = None
_V_cache = None
_W_cache = None

def _build_strassen_4x4():
    """
    Constructs the Rank-49 Strassen decomposition for 4x4 matrix multiplication.
    Returns U, V, W as numpy arrays.
    """
    n = m = p = 4
    rank = 49
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    def u_idx(a_row, a_col):
        return a_row * m + a_col
    
    def v_idx(b_row, b_col):
        return b_row * p + b_col
    
    def w_idx(c_row, c_col):
        return c_col * n + c_row
    
    outer_terms = []
    
    # Term 0: M1 = (A11 + A22)(B11 + B22)
    L0 = {}
    R0 = {}
    for r in range(2):
        for c in range(2):
            L0[(r,c)] = {(r,c): 1, (r+2,c+2): 1}
            R0[(r,c)] = {(r,c): 1, (r+2,c+2): 1}
    C0 = {(0,0): 1, (1,1): 1}
    outer_terms.append((L0, R0, C0))
    
    # Term 1: M2 = (A21 + A22)B11
    L1 = {}
    R1 = {}
    for r in range(2):
        for c in range(2):
            L1[(r,c)] = {(r+2,c): 1, (r+2,c+2): 1}
            R1[(r,c)] = {(r,c): 1}
    C1 = {(1,0): 1, (1,1): -1}
    outer_terms.append((L1, R1, C1))
    
    # Term 2: M3 = A11(B12 - B22)
    L2 = {}
    R2 = {}
    for r in range(2):
        for c in range(2):
            L2[(r,c)] = {(r,c): 1}
            R2[(r,c)] = {(r,c+2): 1, (r+2,c+2): -1}
    C2 = {(0,1): 1, (1,1): 1}
    outer_terms.append((L2, R2, C2))
    
    # Term 3: M4 = A22(B21 - B11)
    L3 = {}
    R3 = {}
    for r in range(2):
        for c in range(2):
            L3[(r,c)] = {(r+2,c+2): 1}
            R3[(r,c)] = {(r+2,c): 1, (r,c): -1}
    C3 = {(0,0): 1, (1,0): 1}
    outer_terms.append((L3, R3, C3))
    
    # Term 4: M5 = (A11 + A12)B22
    L4 = {}
    R4 = {}
    for r in range(2):
        for c in range(2):
            L4[(r,c)] = {(r,c): 1, (r,c+2): 1}
            R4[(r,c)] = {(r+2,c+2): 1}
    C4 = {(0,0): -1, (0,1): 1}
    outer_terms.append((L4, R4, C4))
    
    # Term 5: M6 = (A21 - A11)(B11 + B12)
    L5 = {}
    R5 = {}
    for r in range(2):
        for c in range(2):
            L5[(r,c)] = {(r+2,c): 1, (r,c): -1}
            R5[(r,c)] = {(r,c): 1, (r,c+2): 1}
    C5 = {(1,1): 1}
    outer_terms.append((L5, R5, C5))
    
    # Term 6: M7 = (A12 - A22)(B21 + B22)
    L6 = {}
    R6 = {}
    for r in range(2):
        for c in range(2):
            L6[(r,c)] = {(r,c+2): 1, (r+2,c+2): -1}
            R6[(r,c)] = {(r+2,c): 1, (r+2,c+2): 1}
    C6 = {(0,0): 1}
    outer_terms.append((L6, R6, C6))
    
    inner_strassen = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for t in range(7):
        L_t, R_t, C_contrib = outer_terms[t]
        
        for s in range(7):
            r_idx = t * 7 + s
            L_coeffs, R_coeffs, P_contrib = inner_strassen[s]
            
            u_vec = np.zeros(16)
            for (lr, lc), l_coeff in L_coeffs.items():
                if l_coeff == 0:
                    continue
                for (ar, ac), a_coeff in L_t[(lr, lc)].items():
                    idx = u_idx(ar, ac)
                    u_vec[idx] += l_coeff * a_coeff
            
            v_vec = np.zeros(16)
            for (rr, rc), r_coeff in R_coeffs.items():
                if r_coeff == 0:
                    continue
                for (br, bc), b_coeff in R_t[(rr, rc)].items():
                    idx = v_idx(br, bc)
                    v_vec[idx] += r_coeff * b_coeff
            
            w_vec = np.zeros(16)
            for (pr, pc), p_coeff in P_contrib.items():
                if p_coeff == 0:
                    continue
                for (block_row, block_col), c_coeff in C_contrib.items():
                    total_coeff = p_coeff * c_coeff
                    if total_coeff == 0:
                        continue
                    c_row = block_row * 2 + pr
                    c_col = block_col * 2 + pc
                    idx = w_idx(c_row, c_col)
                    w_vec[idx] += total_coeff
            
            U[:, r_idx] = u_vec
            V[:, r_idx] = v_vec
            W[:, r_idx] = w_vec
            
    return U, V, W

def construct():
    global _U_cache, _V_cache, _W_cache
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    if _U_cache is None:
        _U_cache, _V_cache, _W_cache = _build_strassen_4x4()
        
    if time.time() > deadline:
        # If we somehow took too long, we still return the cached result if available
        pass
        
    return {"U": _U_cache.tolist(), "V": _V_cache.tolist(), "W": _W_cache.tolist()}
# EVOLVE-BLOCK-END
