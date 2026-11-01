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
    deadline = 120.0  # safety margin before 150s kill
    
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
    
    # Strassen decomposition for 2x2 block matrices where each block is 2x2
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # 7 Strassen products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix product of two 2x2 matrices.
    # We use Strassen on each 2x2 sub-product (7 scalar mults each).
    
    # For a 2x2 Strassen product of L (2x2) and R (2x2):
    # s1 = (L[0,0] + L[1,1]) * (R[0,0] + R[1,1])
    # s2 = (L[1,0] + L[1,1]) * R[0,0]
    # s3 = L[0,0] * (R[0,1] - R[1,1])
    # s4 = L[1,1] * (R[1,0] - R[0,0])
    # s5 = (L[0,0] + L[0,1]) * R[1,1]
    # s6 = (L[1,0] - L[0,0]) * (R[0,0] + R[0,1])
    # s7 = (L[0,1] - L[1,1]) * (R[1,0] + R[1,1])
    
    # P[0,0] = s1 + s4 - s5 + s7
    # P[0,1] = s3 + s5
    # P[1,0] = s2 + s4
    # P[1,1] = s1 - s2 + s3 + s6
    
    # Each s_k is a scalar product of two linear combinations.
    # In the tensor decomposition, each scalar product contributes one rank term.
    
    # For the 4x4 Strassen:
    # Each of the 7 outer products M_t involves a 2x2 * 2x2 multiplication.
    # Each 2x2 * 2x2 uses 7 scalar mults (inner Strassen).
    # Total: 7 * 7 = 49 rank terms.
    
    # Let's define the structure.
    # Outer term t (0-6): defines L_t (2x2, linear combo of A blocks) and R_t (2x2, linear combo of B blocks)
    # Inner Strassen on L_t * R_t: 7 scalar products
    # Each scalar product: (linear combo of A elements) * (linear combo of B elements) -> contributes to C with coefficients
    
    # For outer term t, L_t[r][c] = sum of coeff * A[bi][bj][er][ec]
    # where bi,bj in {0,1} (block indices), er,ec in {0,1} (element within block)
    # Global A index: row = 2*bi+er, col = 2*bj+ec
    # U index: (2*bi+er)*4 + (2*bj+ec)
    
    # Similarly for R_t and B.
    
    # For the inner Strassen scalar product s_k:
    # s_k = (sum of coeff * L_t elements) * (sum of coeff * R_t elements)
    # Each L_t element is a linear combo of A elements
    # Each R_t element is a linear combo of B elements
    # So s_k = (sum of coeff_A * A_elem) * (sum of coeff_B * B_elem)
    # = sum over A_elem, B_elem of (coeff_A * coeff_B) * A_elem * B_elem
    
    # This product contributes to P_t[r][c] for various (r,c), and P_t contributes to C blocks.
    
    # Let me define the outer terms explicitly.
    
    # A block (bi, bj) element (er, ec): global A[2*bi+er, 2*bj+ec]
    # B block (bi, bj) element (er, ec): global B[2*bi+er, 2*bj+ec]
    # C block (bi, bj) element (er, ec): global C[2*bi+er, 2*bj+ec]
    
    # Outer term definitions:
    # For each term, L and R are 2x2 matrices where each entry is a dict:
    # {(bi, bj, er, ec): coeff}
    
    outer_terms = []
    
    # Term 0: M1 = (A11 + A22)(B11 + B22)
    L0 = {}
    R0 = {}
    for r in range(2):
        for c in range(2):
            L0[(r,c)] = {(0,0,r,c): 1, (1,1,r,c): 1}
            R0[(r,c)] = {(0,0,r,c): 1, (1,1,r,c): 1}
    # M1 contributes: C11 += M1, C22 += M1
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
    
    # Inner Strassen: for each outer term, compute 7 scalar products
    # Inner Strassen scalar products for L * R (both 2x2):
    # s1 = (L[0,0] + L[1,1]) * (R[0,0] + R[1,1])
    # s2 = (L[1,0] + L[1,1]) * R[0,0]
    # s3 = L[0,0] * (R[0,1] - R[1,1])
    # s4 = L[1,1] * (R[1,0] - R[0,0])
    # s5 = (L[0,0] + L[0,1]) * R[1,1]
    # s6 = (L[1,0] - L[0,0]) * (R[0,0] + R[0,1])
    # s7 = (L[0,1] - L[1,1]) * (R[1,0] + R[1,1])
    
    # P[0,0] = s1 + s4 - s5 + s7
    # P[0,1] = s3 + s5
    # P[1,0] = s2 + s4
    # P[1,1] = s1 - s2 + s3 + s6
    
    # For each scalar product s_k, we need:
    # - The L-side linear combination (which L elements with what coefficients)
    # - The R-side linear combination (which R elements with what coefficients)
    # - How s_k contributes to P[r,c] for each (r,c)
    
    # Define inner Strassen scalar products:
    # Each is: (L_coeffs, R_coeffs, P_contrib)
    # L_coeffs: dict {(r,c): coeff} for L elements
    # R_coeffs: dict {(r,c): coeff} for R elements
    # P_contrib: dict {(r,c): coeff} for P elements
    
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
    
    # Now build U, V, W
    for t, (L, R, C_contrib) in enumerate(outer_terms):
        for k, (l_coeffs, r_coeffs, p_contrib) in enumerate(inner_strassen):
            rank_idx = t * 7 + k
            
            # Check deadline
            if time.time() - t_start > deadline:
                # Return what we have so far (may be incomplete)
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Build U contribution:
            # For each L element (lr, lc) with coeff l_c, L[lr][lc] is a dict of A elements
            # The scalar product involves (sum l_c * L[lr][lc]) * (sum r_c * R[lr2][lc2])
            # L[lr][lc] = sum of a_coeff * A[bi,bj,er,ec]
            # So the L-side total = sum over lr,lc of l_c * sum over A_elems of a_coeff * A_elem
            
            # U contribution: for each A element, accumulate the coefficient
            for (lr, lc), l_c in l_coeffs.items():
                for (bi, bj, er, ec), a_coeff in L[(lr, lc)].items():
                    u_i = u_idx(2*bi+er, 2*bj+ec)
                    U[u_i, rank_idx] += l_c * a_coeff
            
            # Build V contribution:
            for (lr2, lc2), r_c in r_coeffs.items():
                for (bi, bj, er, ec), b_coeff in R[(lr2, lc2)].items():
                    v_i = v_idx(2*bi+er, 2*bj+ec)
                    V[v_i, rank_idx] += r_c * b_coeff
            
            # Build W contribution:
            # p_contrib maps P[r][c] -> coeff
            # C_contrib maps P[r][c] -> list of (bi, bj, er, ec, coeff)
            for (pr, pc), p_c in p_contrib.items():
                for (bi, bj, er, ec, c_coeff) in C_contrib[(pr, pc)]:
                    w_i = w_idx(2*bi+er, 2*bj+ec)
                    W[w_i, rank_idx] += p_c * c_coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
