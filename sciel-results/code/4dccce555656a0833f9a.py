"""Matrix multiplication 4x4x4 tensor decomposition using recursive Strassen (rank 49)."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    
    Convention:
      C[i,j] = sum_k A[i,k] * B[k,j]
      U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] summed over r gives the tensor.
      Here n=m=p=4, so:
        U index: i*4 + k  (A row i, A col k)
        V index: k*4 + j  (B row k, B col j)
        W index: j*4 + i  (C col j, C row i)
    
    Strassen at the 4x4 level: split into 2x2 blocks of 2x2 matrices.
    Each 2x2 sub-multiplication uses Strassen (7 scalar mults).
    Total rank: 7 * 7 = 49.
    """
    t_start = time.time()
    deadline = 110.0  # safety margin before 150s kill
    
    n = m = p = 4
    rank = 49
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Index helpers
    def u_idx(a_row, a_col):
        return a_row * m + a_col
    
    def v_idx(b_row, b_col):
        return b_row * p + b_col
    
    def w_idx(c_row, c_col):
        # W[k*n+i, r] where k is B's col (= C's col), i is A's row (= C's row)
        # So W index = c_col * n + c_row
        return c_col * n + c_row
    
    # Outer Strassen terms for 4x4 = 2x2 block matrices of 2x2 blocks
    # Each term: L (dict of (lr,lc) -> {(bi,bj,er,ec): coeff}), R (same), C_contrib (dict of (pr,pc) -> [(bi,bj,er,ec,coeff)])
    
    outer_terms = []
    
    # Term 0: M1 = (A11 + A22)(B11 + B22)
    L0 = {}
    R0 = {}
    for r in range(2):
        for c in range(2):
            L0[(r,c)] = {(0,0,r,c): 1, (1,1,r,c): 1}
            R0[(r,c)] = {(0,0,r,c): 1, (1,1,r,c): 1}
    C0 = {}
    for r in range(2):
        for c in range(2):
            C0[(r,c)] = [(0,0,r,c,1), (1,1,r,c,1)]
    outer_terms.append((L0, R0, C0))
    
    # Term 1: M2 = (A21 + A22)B11
    L1 = {}
    R1 = {}
    for r in range(2):
        for c in range(2):
            L1[(r,c)] = {(1,0,r,c): 1, (1,1,r,c): 1}
            R1[(r,c)] = {(0,0,r,c): 1}
    C1 = {}
    for r in range(2):
        for c in range(2):
            C1[(r,c)] = [(1,0,r,c,1), (1,1,r,c,-1)]
    outer_terms.append((L1, R1, C1))
    
    # Term 2: M3 = A11(B12 - B22)
    L2 = {}
    R2 = {}
    for r in range(2):
        for c in range(2):
            L2[(r,c)] = {(0,0,r,c): 1}
            R2[(r,c)] = {(0,1,r,c): 1, (1,1,r,c): -1}
    C2 = {}
    for r in range(2):
        for c in range(2):
            C2[(r,c)] = [(0,1,r,c,1), (1,1,r,c,1)]
    outer_terms.append((L2, R2, C2))
    
    # Term 3: M4 = A22(B21 - B11)
    L3 = {}
    R3 = {}
    for r in range(2):
        for c in range(2):
            L3[(r,c)] = {(1,1,r,c): 1}
            R3[(r,c)] = {(1,0,r,c): 1, (0,0,r,c): -1}
    C3 = {}
    for r in range(2):
        for c in range(2):
            C3[(r,c)] = [(0,0,r,c,1), (1,0,r,c,1)]
    outer_terms.append((L3, R3, C3))
    
    # Term 4: M5 = (A11 + A12)B22
    L4 = {}
    R4 = {}
    for r in range(2):
        for c in range(2):
            L4[(r,c)] = {(0,0,r,c): 1, (0,1,r,c): 1}
            R4[(r,c)] = {(1,1,r,c): 1}
    C4 = {}
    for r in range(2):
        for c in range(2):
            C4[(r,c)] = [(0,0,r,c,-1), (0,1,r,c,1)]
    outer_terms.append((L4, R4, C4))
    
    # Term 5: M6 = (A21 - A11)(B11 + B12)
    L5 = {}
    R5 = {}
    for r in range(2):
        for c in range(2):
            L5[(r,c)] = {(1,0,r,c): 1, (0,0,r,c): -1}
            R5[(r,c)] = {(0,0,r,c): 1, (0,1,r,c): 1}
    C5 = {}
    for r in range(2):
        for c in range(2):
            C5[(r,c)] = [(1,1,r,c,1)]
    outer_terms.append((L5, R5, C5))
    
    # Term 6: M7 = (A12 - A22)(B21 + B22)
    L6 = {}
    R6 = {}
    for r in range(2):
        for c in range(2):
            L6[(r,c)] = {(0,1,r,c): 1, (1,1,r,c): -1}
            R6[(r,c)] = {(1,0,r,c): 1, (1,1,r,c): 1}
    C6 = {}
    for r in range(2):
        for c in range(2):
            C6[(r,c)] = [(0,0,r,c,1)]
    outer_terms.append((L6, R6, C6))
    
    # Inner Strassen: 7 scalar products for a 2x2 * 2x2 multiplication
    # Each: (l_coeffs, r_coeffs, p_contrib)
    inner_strassen = [
        # s1 = (L[0,0] + L[1,1]) * (R[0,0] + R[1,1])
        # P[0,0] += s1, P[1,1] += s1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # s2 = (L[1,0] + L[1,1]) * R[0,0]
        # P[1,0] += s2, P[1,1] -= s2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # s3 = L[0,0] * (R[0,1] - R[1,1])
        # P[0,1] += s3, P[1,1] += s3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # s4 = L[1,1] * (R[1,0] - R[0,0])
        # P[0,0] += s4, P[1,0] += s4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # s5 = (L[0,0] + L[0,1]) * R[1,1]
        # P[0,0] -= s5, P[0,1] += s5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # s6 = (L[1,0] - L[0,0]) * (R[0,0] + R[0,1])
        # P[1,1] += s6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # s7 = (L[0,1] - L[1,1]) * (R[1,0] + R[1,1])
        # P[0,0] += s7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Build U, V, W
    for t, (L, R, C_contrib) in enumerate(outer_terms):
        for k, (l_coeffs, r_coeffs, p_contrib) in enumerate(inner_strassen):
            rank_idx = t * 7 + k
            
            if time.time() - t_start > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # U contribution
            for (lr, lc), l_c in l_coeffs.items():
                for (bi, bj, er, ec), a_coeff in L[(lr, lc)].items():
                    u_i = u_idx(2*bi+er, 2*bj+ec)
                    U[u_i, rank_idx] += l_c * a_coeff
            
            # V contribution
            for (lr2, lc2), r_c in r_coeffs.items():
                for (bi, bj, er, ec), b_coeff in R[(lr2, lc2)].items():
                    v_i = v_idx(2*bi+er, 2*bj+ec)
                    V[v_i, rank_idx] += r_c * b_coeff
            
            # W contribution
            for (pr, pc), p_c in p_contrib.items():
                for (bi, bj, er, ec, c_coeff) in C_contrib[(pr, pc)]:
                    w_i = w_idx(2*bi+er, 2*bj+ec)
                    W[w_i, rank_idx] += p_c * c_coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
