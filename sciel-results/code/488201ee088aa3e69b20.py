"""Tensor decomposition for 4x4 @ 4x6 matrix multiplication.
Uses Strassen recursion to reduce rank below schoolbook 96.
"""

DIMENSIONS = [4, 4, 6]

# Base 2x2x2 Strassen: 7 rank-1 terms
# Each term: (A_coeffs, B_coeffs) where coeffs are for [a00, a01, a10, a11] and [b00, b01, b10, b11]
# Output contribution: (c00, c01, c10, c11)
_STRASSEN_2x2x2 = [
    # term 0: M0 = (a00+a11)*(b00+b11)
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),
    # term 1: M1 = (a10+a11)*b00
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
    # term 2: M2 = a00*(b01-b11)
    ([1, 0, 0, 0], [0, 1, 0, -1], [0, -1, 0, 1]),
    # term 3: M3 = a11*(b10-b00)
    ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 0, 0]),
    # term 4: M4 = (a00+a10)*b11
    ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
    # term 5: M5 = (a01-a11)*(b10+b11)
    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, -1, 0]),
    # term 6: M6 = (a10-a00)*(b00+b01)
    ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 1, -1, 0]),
]

def _build_strassen_4x4x4_terms():
    """Build 49 rank-1 terms for 4x4x4 Strassen.
    Returns list of (U_coeffs[16], V_coeffs[16], W_coeffs[16]).
    """
    # Block index mappings for 4x4 matrix (row-major, 16 elements)
    # Block (r,c) where r,c in {0,1} is 2x2 at rows 2*r:2*r+2, cols 2*c:2*c+2
    block_indices = {}
    for r in range(2):
        for c in range(2):
            idxs = []
            for lr in range(2):
                for lc in range(2):
                    idxs.append((2*r + lr) * 4 + (2*c + lc))
            block_indices[(r, c)] = idxs
    
    terms = []
    for outer_s in range(7):
        a_outer, b_outer, c_outer = _STRASSEN_2x2x2[outer_s]
        
        for inner_s in range(7):
            a_inner, b_inner, c_inner = _STRASSEN_2x2x2[inner_s]
            
            U_coeffs = [0] * 16
            V_coeffs = [0] * 16
            W_coeffs = [0] * 16
            
            # U: outer combo of A-blocks, then inner combo within each block
            for block_idx, a_coeff in enumerate(a_outer):
                if a_coeff == 0:
                    continue
                r, c = divmod(block_idx, 2)
                for li, idx in enumerate(block_indices[(r, c)]):
                    U_coeffs[idx] += a_coeff * a_inner[li]
            
            for block_idx, b_coeff in enumerate(b_outer):
                if b_coeff == 0:
                    continue
                r, c = divmod(block_idx, 2)
                for li, idx in enumerate(block_indices[(r, c)]):
                    V_coeffs[idx] += b_coeff * b_inner[li]
            
            for block_idx, c_coeff in enumerate(c_outer):
                if c_coeff == 0:
                    continue
                r, c = divmod(block_idx, 2)
                for li, idx in enumerate(block_indices[(r, c)]):
                    W_coeffs[idx] += c_coeff * c_inner[li]
            
            terms.append((U_coeffs, V_coeffs, W_coeffs))
    
    return terms

def construct():
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    strassen_terms = _build_strassen_4x4x4_terms()
    
    total_rank = len(strassen_terms) + 32  # 49 + 32
    U = [[0] * total_rank for _ in range(n * m)]  # 16
    V = [[0] * total_rank for _ in range(m * p)]  # 24
    W = [[0] * total_rank for _ in range(p * n)]  # 24
    
    col = 0
    for U_coeffs, V_coeffs, W_coeffs in strassen_terms:
        for idx in range(16):
            if U_coeffs[idx] != 0:
                U[idx][col] += U_coeffs[idx]
            if V_coeffs[idx] != 0:
                V[idx][col] += V_coeffs[idx]
            if W_coeffs[idx] != 0:
                W[idx][col] += W_coeffs[idx]
        col += 1
    
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U[i * 4 + j][col] = 1
                V[j * 6 + (4 + k)][col] = 1
                W[(4 + k) * 4 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
