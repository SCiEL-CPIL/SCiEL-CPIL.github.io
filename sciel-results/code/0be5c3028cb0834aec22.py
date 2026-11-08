import numpy as np

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    Directly constructs U, V, W matrices.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Index helpers
    # A[i, j] -> U[i*4 + j, :]
    # B[i, j] -> V[i*4 + j, :]
    # C[i, j] -> W[j*4 + i, :] (Column-major for C)
    
    def u_idx(i, j): return i * 4 + j
    def v_idx(i, j): return i * 4 + j
    def w_idx(i, j): return j * 4 + i
    
    # Recursive Strassen for 4x4
    # Split into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # The 7 block-level Strassen products:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # Each block product is a 2x2 matrix multiplication.
    # We can use scalar Strassen for each 2x2 block multiplication?
    # Yes, that gives 7 * 7 = 49 scalar products.
    
    # Block contribution to C:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let's define the scalar Strassen for a 2x2 matrix product X * Y = Z.
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Z = [[z00, z01], [z10, z11]]
    
    # Scalar Strassen terms for X*Y:
    # m1 = (x00 + x11)(y00 + y11)
    # m2 = (x10 + x11)y00
    # m3 = x00(y01 - y11)
    # m4 = x11(y10 - y00)
    # m5 = (x00 + x01)y11
    # m6 = (x10 - x00)(y00 + y01)
    # m7 = (x01 - x11)(y10 + y11)
    
    # Z00 = m1 + m4 - m5 + m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6
    
    # We will iterate over the 7 block terms.
    # For each block term t, we have a 2x2 matrix L (from A) and R (from B).
    # We compute the 7 scalar products for L*R.
    # Each scalar product k contributes to the 2x2 result P_t.
    # P_t then contributes to the global C blocks.
    
    # Define the 7 block terms (L, R) as lists of (row, col, coeff) for A and B.
    # A block (bi, bj) element (r, c) is at global (2*bi+r, 2*bj+c).
    
    def a_ref(bi, bj, r, c, coeff=1):
        return [(2*bi+r, 2*bj+c, coeff)]
        
    def b_ref(bi, bj, r, c, coeff=1):
        return [(2*bi+r, 2*bj+c, coeff)]
        
    def add_ref(ref1, ref2):
        d = {}
        for r, c, v in ref1: d[(r, c)] = d.get((r, c), 0) + v
        for r, c, v in ref2: d[(r, c)] = d.get((r, c), 0) + v
        return [(k[0], k[1], v) for k, v in d.items() if v != 0]
        
    def sub_ref(ref1, ref2):
        d = {}
        for r, c, v in ref1: d[(r, c)] = d.get((r, c), 0) + v
        for r, c, v in ref2: d[(r, c)] = d.get((r, c), 0) - v
        return [(k[0], k[1], v) for k, v in d.items() if v != 0]

    # Block Terms
    # T0: (A00+A11)(B00+B11)
    L0 = [[add_ref(a_ref(0,0,r,c), a_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    R0 = [[add_ref(b_ref(0,0,r,c), b_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    
    # T1: (A10+A11)B00
    L1 = [[add_ref(a_ref(1,0,r,c), a_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    R1 = [[b_ref(0,0,r,c) for c in range(2)] for r in range(2)]
    
    # T2: A00(B01-B11)
    L2 = [[a_ref(0,0,r,c) for c in range(2)] for r in range(2)]
    R2 = [[sub_ref(b_ref(0,1,r,c), b_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    
    # T3: A11(B10-B00)
    L3 = [[a_ref(1,1,r,c) for c in range(2)] for r in range(2)]
    R3 = [[sub_ref(b_ref(1,0,r,c), b_ref(0,0,r,c)) for c in range(2)] for r in range(2)]
    
    # T4: (A00+A01)B11
    L4 = [[add_ref(a_ref(0,0,r,c), a_ref(0,1,r,c)) for c in range(2)] for r in range(2)]
    R4 = [[b_ref(1,1,r,c) for c in range(2)] for r in range(2)]
    
    # T5: (A10-A00)(B00+B01)
    L5 = [[sub_ref(a_ref(1,0,r,c), a_ref(0,0,r,c)) for c in range(2)] for r in range(2)]
    R5 = [[add_ref(b_ref(0,0,r,c), b_ref(0,1,r,c)) for c in range(2)] for r in range(2)]
    
    # T6: (A01-A11)(B10+B11)
    L6 = [[sub_ref(a_ref(0,1,r,c), a_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    R6 = [[add_ref(b_ref(1,0,r,c), b_ref(1,1,r,c)) for c in range(2)] for r in range(2)]
    
    L_terms = [L0, L1, L2, L3, L4, L5, L6]
    R_terms = [R0, R1, R2, R3, R4, R5, R6]
    
    # Block contributions to C
    # C00: T0(+), T3(+), T4(-), T6(+)  -> Wait, let's re-verify Strassen C formulas.
    # Standard Strassen:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # My T indices:
    # T0 -> P1
    # T1 -> P2
    # T2 -> P3
    # T3 -> P4
    # T4 -> P5
    # T5 -> P6
    # T6 -> P7
    
    block_contribs = [
        [(0, 0, 1), (1, 1, 1)],       # T0 (P1) -> C00, C11
        [(1, 0, 1), (1, 1, -1)],      # T1 (P2) -> C10, C11
        [(0, 1, 1), (1, 1, 1)],       # T2 (P3) -> C01, C11
        [(0, 0, 1), (1, 0, 1)],       # T3 (P4) -> C00, C10
        [(0, 0, -1), (0, 1, 1)],      # T4 (P5) -> C00, C01
        [(1, 1, 1)],                  # T5 (P6) -> C11
        [(0, 0, 1)]                   # T6 (P7) -> C00
    ]
    
    # Scalar Strassen contributions within a 2x2 block product
    # Z00: m1(+), m4(+), m5(-), m7(+)
    # Z01: m3(+), m5(+)
    # Z10: m2(+), m4(+)
    # Z11: m1(+), m2(-), m3(+), m6(+)
    
    sub_contrib = [
        [(0, 0, 1), (1, 1, 1)],    # m1 -> Z00, Z11
        [(1, 0, 1), (1, 1, -1)],   # m2 -> Z10, Z11
        [(0, 1, 1), (1, 1, 1)],    # m3 -> Z01, Z11
        [(0, 0, 1), (1, 0, 1)],    # m4 -> Z00, Z10
        [(0, 0, -1), (0, 1, 1)],   # m5 -> Z00, Z01
        [(1, 1, 1)],               # m6 -> Z11
        [(0, 0, 1)]                # m7 -> Z00
    ]
    
    for t in range(7):
        L_t = L_terms[t]
        R_t = R_terms[t]
        contrib_t = block_contribs[t]
        
        a = L_t[0][0]; b = L_t[0][1]; c = L_t[1][0]; d = L_t[1][1]
        e = R_t[0][0]; f = R_t[0][1]; g = R_t[1][0]; h = R_t[1][1]
        
        sub_products = [
            (add_ref(a, d), add_ref(e, h)), # m1
            (add_ref(c, d), e),             # m2
            (a, sub_ref(f, h)),             # m3
            (d, sub_ref(g, e)),             # m4
            (add_ref(a, b), h),             # m5
            (sub_ref(c, a), add_ref(e, f)), # m6
            (sub_ref(b, d), add_ref(g, h))  # m7
        ]
        
        for k in range(7):
            x_k, y_k = sub_products[k]
            rank_idx = t * 7 + k
            
            for (row, col, coeff) in x_k:
                U[u_idx(row, col), rank_idx] += coeff
            for (row, col, coeff) in y_k:
                V[v_idx(row, col), rank_idx] += coeff
            
            for (pr, pc, sub_coeff) in sub_contrib[k]:
                for (br, bc, block_coeff) in contrib_t:
                    out_row = 2 * br + pr
                    out_col = 2 * bc + pc
                    total_coeff = sub_coeff * block_coeff
                    W[w_idx(out_row, out_col), rank_idx] += total_coeff

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
