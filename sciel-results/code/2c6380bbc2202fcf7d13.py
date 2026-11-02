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
    deadline = 118.0  # safety margin before 150s kill
    
    n = m = p = 4
    rank = 49
    
    # Precompute all 49 bilinear terms as flat arrays
    # Each term r has:
    #   u_terms: list of (u_index, coeff)
    #   v_terms: list of (v_index, coeff)
    #   w_terms: list of (w_index, coeff)
    
    # Index helpers
    def u_idx(a_row, a_col):
        return a_row * m + a_col
    
    def v_idx(b_row, b_col):
        return b_row * p + b_col
    
    def w_idx(c_row, c_col):
        # W[k*n+i, r] where k is B's col (= C's col), i is A's row (= C's row)
        # So W index = c_col * n + c_row
        return c_col * n + c_row
    
    # Outer Strassen terms for 4x4 -> 2x2 blocks
    # Each term t: L_t is 2x2 matrix of linear combos of A elements
    #               R_t is 2x2 matrix of linear combos of B elements
    #               C_t maps P[r][c] to list of (c_row, c_col, coeff)
    
    # A block (bi, bj) element (er, ec): global A[2*bi+er, 2*bj+ec]
    # B block (bi, bj) element (er, ec): global B[2*bi+er, 2*bj+ec]
    
    # Define outer terms as: L[t][r][c] = {(global_a_row, global_a_col): coeff}
    #                        R[t][r][c] = {(global_b_row, global_b_col): coeff}
    #                        C[t][p_r][p_c] = [(c_row, c_col, coeff), ...]
    
    L_outer = []
    R_outer = []
    C_outer = []
    
    # Term 0: M1 = (A11 + A22)(B11 + B22)
    L0 = [[{} for _ in range(2)] for _ in range(2)]
    R0 = [[{} for _ in range(2)] for _ in range(2)]
    C0 = {}
    for r in range(2):
        for c in range(2):
            L0[r][c] = {(0, c): 1, (2, 2+c): 1}
            R0[r][c] = {(0, c): 1, (2, 2+c): 1}
            C0[(r, c)] = [(r, c, 1), (2+r, 2+c, 1)]
    L_outer.append(L0)
    R_outer.append(R0)
    C_outer.append(C0)
    
    # Term 1: M2 = (A21 + A22)B11
    L1 = [[{} for _ in range(2)] for _ in range(2)]
    R1 = [[{} for _ in range(2)] for _ in range(2)]
    C1 = {}
    for r in range(2):
        for c in range(2):
            L1[r][c] = {(2+r, c): 1, (2+r, 2+c): 1}
            R1[r][c] = {(0, c): 1}
            C1[(r, c)] = [(2+r, c, 1), (2+r, 2+c, -1)]
    L_outer.append(L1)
    R_outer.append(R1)
    C_outer.append(C1)
    
    # Term 2: M3 = A11(B12 - B22)
    L2 = [[{} for _ in range(2)] for _ in range(2)]
    R2 = [[{} for _ in range(2)] for _ in range(2)]
    C2 = {}
    for r in range(2):
        for c in range(2):
            L2[r][c] = {(r, c): 1}
            R2[r][c] = {(r, 2+c): 1, (2+r, 2+c): -1}
            C2[(r, c)] = [(r, 2+c, 1), (2+r, 2+c, 1)]
    L_outer.append(L2)
    R_outer.append(R2)
    C_outer.append(C2)
    
    # Term 3: M4 = A22(B21 - B11)
    L3 = [[{} for _ in range(2)] for _ in range(2)]
    R3 = [[{} for _ in range(2)] for _ in range(2)]
    C3 = {}
    for r in range(2):
        for c in range(2):
            L3[r][c] = {(2+r, 2+c): 1}
            R3[r][c] = {(2+r, c): 1, (r, c): -1}
            C3[(r, c)] = [(r, c, 1), (2+r, c, 1)]
    L_outer.append(L3)
    R_outer.append(R3)
    C_outer.append(C3)
    
    # Term 4: M5 = (A11 + A12)B22
    L4 = [[{} for _ in range(2)] for _ in range(2)]
    R4 = [[{} for _ in range(2)] for _ in range(2)]
    C4 = {}
    for r in range(2):
        for c in range(2):
            L4[r][c] = {(r, c): 1, (r, 2+c): 1}
            R4[r][c] = {(2+r, 2+c): 1}
            C4[(r, c)] = [(r, c, -1), (r, 2+c, 1)]
    L_outer.append(L4)
    R_outer.append(R4)
    C_outer.append(C4)
    
    # Term 5: M6 = (A21 - A11)(B11 + B12)
    L5 = [[{} for _ in range(2)] for _ in range(2)]
    R5 = [[{} for _ in range(2)] for _ in range(2)]
    C5 = {}
    for r in range(2):
        for c in range(2):
            L5[r][c] = {(2+r, c): 1, (r, c): -1}
            R5[r][c] = {(r, c): 1, (r, 2+c): 1}
            C5[(r, c)] = [(2+r, 2+c, 1)]
    L_outer.append(L5)
    R_outer.append(R5)
    C_outer.append(C5)
    
    # Term 6: M7 = (A12 - A22)(B21 + B22)
    L6 = [[{} for _ in range(2)] for _ in range(2)]
    R6 = [[{} for _ in range(2)] for _ in range(2)]
    C6 = {}
    for r in range(2):
        for c in range(2):
            L6[r][c] = {(r, 2+c): 1, (2+r, 2+c): -1}
            R6[r][c] = {(2+r, c): 1, (2+r, 2+c): 1}
            C6[(r, c)] = [(r, c, 1)]
    L_outer.append(L6)
    R_outer.append(R6)
    C_outer.append(C6)
    
    # Inner Strassen scalar products for 2x2 * 2x2
    # Each k: l_coeffs = {(lr, lc): coeff}, r_coeffs = {(rr, rc): coeff}, p_contrib = {(pr, pc): coeff}
    inner_strassen = [
        # s1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # s2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # s3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # s4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # s5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # s6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # s7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Precompute all 49 terms
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for t in range(7):
        L = L_outer[t]
        R = R_outer[t]
        C = C_outer[t]
        
        for k in range(7):
            l_coeffs, r_coeffs, p_contrib = inner_strassen[k]
            rank_idx = t * 7 + k
            
            # Build U contribution
            u_dict = {}
            for (lr, lc), l_c in l_coeffs.items():
                for (ar, ac), a_coeff in L[lr][lc].items():
                    u_i = u_idx(ar, ac)
                    u_dict[u_i] = u_dict.get(u_i, 0) + l_c * a_coeff
            
            for u_i, coeff in u_dict.items():
                if coeff != 0:
                    U[u_i, rank_idx] = coeff
            
            # Build V contribution
            v_dict = {}
            for (rr, rc), r_c in r_coeffs.items():
                for (br, bc), b_coeff in R[rr][rc].items():
                    v_i = v_idx(br, bc)
                    v_dict[v_i] = v_dict.get(v_i, 0) + r_c * b_coeff
            
            for v_i, coeff in v_dict.items():
                if coeff != 0:
                    V[v_i, rank_idx] = coeff
            
            # Build W contribution
            w_dict = {}
            for (pr, pc), p_c in p_contrib.items():
                for (cr, cc, c_coeff) in C[(pr, pc)]:
                    w_i = w_idx(cr, cc)
                    w_dict[w_i] = w_dict.get(w_i, 0) + p_c * c_coeff
            
            for w_i, coeff in w_dict.items():
                if coeff != 0:
                    W[w_i, rank_idx] = coeff
    
    # Check deadline
    if time.time() - t_start > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
