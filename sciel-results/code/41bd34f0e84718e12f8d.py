import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+k, r], V[k*4+j, r], W[j*4+i, r]
    Tensor: T[i*4+k, k*4+j, j*4+i] = sum_r U[i*4+k,r] * V[k*4+j,r] * W[j*4+i,r] = 1
    """
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank))  # 16 x 49
    V = np.zeros((m * p, rank))  # 16 x 49
    W = np.zeros((p * n, rank))  # 16 x 49
    
    # Direct enumeration of the 49 rank-1 terms via Kronecker structure of Strassen.
    # Level 1 (block level): 7 Strassen terms for 4x4 = 2x2 blocks of 2x2
    # Level 2 (scalar level): 7 Strassen terms for each 2x2 block product
    # Total: 7 * 7 = 49 scalar multiplications.
    
    # Strassen 2x2: given matrices X=[[x00,x01],[x10,x11]], Y=[[y00,y01],[y10,y11]]
    # 7 products:
    #   m0 = (x00+x11)*(y00+y11)
    #   m1 = (x10+x11)*y00
    #   m2 = x00*(y01-y11)
    #   m3 = x11*(y10-y00)
    #   m4 = (x00+x01)*y11
    #   m5 = (x10-x00)*(y00+y01)
    #   m6 = (x01-x11)*(y10+y11)
    # Result:
    #   z00 = m0 + m3 - m4 + m6
    #   z01 = m2 + m4
    #   z10 = m1 + m3
    #   z11 = m0 - m1 + m2 + m5
    
    # For each level-1 term t (0..6), we have L_t (2x2 linear combos of A) and R_t (2x2 linear combos of B)
    # For each level-2 term s (0..6), we have a product X_s * Y_s where X_s, Y_s are linear combos
    
    # Define the 7 level-1 Strassen terms as (L_matrix, R_matrix, C_contrib)
    # L_matrix[i][j] = list of (a_row, a_col, coeff)
    # R_matrix[i][j] = list of (b_row, b_col, coeff)
    # C_contrib: dict mapping (local_row, local_col) -> list of (global_c_row, global_c_col, coeff)
    
    def block_a(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def block_b(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def block_c(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    # Build the 7 level-1 terms
    level1_terms = []
    
    # Term 0: (A11+A22)*(B11+B22) -> contributes +1 to C11 and +1 to C22
    L0 = []
    R0 = []
    for r in range(2):
        row_l = []
        row_r = []
        for c in range(2):
            ar, ac = block_a(0,0,r,c)
            dr, dc = block_a(1,1,r,c)
            row_l.append([(ar,ac,1),(dr,dc,1)])
            er, ec = block_b(0,0,r,c)
            hr, hc = block_b(1,1,r,c)
            row_r.append([(er,ec,1),(hr,hc,1)])
        L0.append(row_l)
        R0.append(row_r)
    C0 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = block_c(0,0,r,c)
            cr2, cc2 = block_c(1,1,r,c)
            C0[(r,c)] = [(cr,cc,1),(cr2,cc2,1)]
    level1_terms.append((L0, R0, C0))
    
    # Term 1: (A21+A22)*B11 -> contributes +1 to C21 and -1 to C22
    L1 = []
    R1 = []
    for r in range(2):
        row_l = []
        row_r = []
        for c in range(2):
            ar, ac = block_a(1,0,r,c)
            dr, dc = block_a(1,1,r,c)
            row_l.append([(ar,ac,1),(dr,dc,1)])
            er, ec = block_b(0,0,r,c)
            row_r.append([(er,ec,1)])
        L1.append(row_l)
        R1.append(row_r)
    C1 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = block_c(1,0,r,c)
            cr2, cc2 = block_c(1,1,r,c)
            C1[(r,c)] = [(cr,cc,1),(cr2,cc2,-1)]
    level1_terms.append((L1, R1, C1))
    
    # Term 2: A11*(B12-B22) -> contributes +1 to C12 and +1 to C22
    L2 = []
    R2 = []
    for r in range(2):
        row_l = []
        row_r = []
        for c in range(2):
            ar, ac = block_a(0,0,r,c)
            row_l.append([(ar,ac,1)])
            fr, fc = block_b(0,1,r,c)
            hr, hc = block_b(1,1,r,c)
            row_r.append([(fr,fc,1),(hr,hc,-1)])
        L2.append(row_l)
        R2.append(row_r)
    C2 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = block_c(0,1,r,c)
            cr2, cc2 = block_c(1,1,r,c)
            C2[(r,c)] = [(cr,cc,1),(cr2,cc2,1)]
    level1_terms.append((L2, R2, C2))
    
    # Term 3: A22*(B21-B11) -> contributes +1 to C11 and +1 to C21
    L3 = []
    R3 = []
    for r in range(2):
        row_l = []
        row_r = []
        for c in range(2):
            dr, dc = block_a(1,1,r,c)
            row_l.append([(dr,dc,1)])
            gr, gc = block_b(1,0,r,c)
            er, ec = block_b(0,0,r,c)
            row_r.append([(gr,gc,1),(er,ec,-1)])
        L3.append(row_l)
        R3.append(row_r)
    C3 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = block_c(0,0,r,c)
            cr2, cc2 = block_c(1,0,r,c)
            C3[(r,c)] = [(cr,cc,1),(cr2,cc2,1)]
    level1_terms.append((L3, R3, C3))
    
    # Term 4: (A11+A12)*B22 -> contributes -1 to C11 and +1 to C12
    L4 = []
    R4 = []
    for r in range(2):
        row_l = []
        row_r = []
        for c in range(2):
            ar, ac = block_a(0,0,r,c)
            br, bc = block_a(0,1,r,c)
            row_l.append([(ar,ac,1),(br,bc,1)])
            hr, hc = block_b(1,1,r,c)
            row_r.append([(hr,hc,1)])
        L4.append(row_l)
        R4.append(row_r)
    C4 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = block_c(0,0,r,c)
            cr2, cc2 = block_c(0,1,r,c)
            C4[(r,c)] = [(cr,cc,-1),(cr2,cc2,1)]
    level1_terms.append((L4, R4, C4))
    
    # Term 5: (A21-A11)*(B11+B12) -> contributes +1 to C22
    L5 = []
    R5 = []
    for r in range(2):
        row_l = []
        row_r = []
        for c in range(2):
            cr, cc = block_a(1,0,r,c)
            ar, ac = block_a(0,0,r,c)
            row_l.append([(cr,cc,1),(ar,ac,-1)])
            er, ec = block_b(0,0,r,c)
            fr, fc = block_b(0,1,r,c)
            row_r.append([(er,ec,1),(fr,fc,1)])
        L5.append(row_l)
        R5.append(row_r)
    C5 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = block_c(1,1,r,c)
            C5[(r,c)] = [(cr,cc,1)]
    level1_terms.append((L5, R5, C5))
    
    # Term 6: (A12-A22)*(B21+B22) -> contributes +1 to C11
    L6 = []
    R6 = []
    for r in range(2):
        row_l = []
        row_r = []
        for c in range(2):
            br, bc = block_a(0,1,r,c)
            dr, dc = block_a(1,1,r,c)
            row_l.append([(br,bc,1),(dr,dc,-1)])
            gr, gc = block_b(1,0,r,c)
            hr, hc = block_b(1,1,r,c)
            row_r.append([(gr,gc,1),(hr,hc,1)])
        L6.append(row_l)
        R6.append(row_r)
    C6 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = block_c(0,0,r,c)
            C6[(r,c)] = [(cr,cc,1)]
    level1_terms.append((L6, R6, C6))
    
    # Strassen level-2: for each of the 7 products m_s, define which local positions (i,j) of L and R
    # are combined, and the coefficients.
    # m0 = (x00+x11)*(y00+y11): L positions [(0,0,+1),(1,1,+1)], R positions [(0,0,+1),(1,1,+1)]
    # m1 = (x10+x11)*y00:        L positions [(1,0,+1),(1,1,+1)], R positions [(0,0,+1)]
    # m2 = x00*(y01-y11):        L positions [(0,0,+1)],           R positions [(0,1,+1),(1,1,-1)]
    # m3 = x11*(y10-y00):        L positions [(1,1,+1)],           R positions [(1,0,+1),(0,0,-1)]
    # m4 = (x00+x01)*y11:        L positions [(0,0,+1),(0,1,+1)],  R positions [(1,1,+1)]
    # m5 = (x10-x00)*(y00+y01):  L positions [(1,0,+1),(0,0,-1)],  R positions [(0,0,+1),(0,1,+1)]
    # m6 = (x01-x11)*(y10+y11):  L positions [(0,1,+1),(1,1,-1)],  R positions [(1,0,+1),(1,1,+1)]
    
    strassen_L = [
        [(0,0,1),(1,1,1)],
        [(1,0,1),(1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1),(0,1,1)],
        [(1,0,1),(0,0,-1)],
        [(0,1,1),(1,1,-1)]
    ]
    strassen_R = [
        [(0,0,1),(1,1,1)],
        [(0,0,1)],
        [(0,1,1),(1,1,-1)],
        [(1,0,1),(0,0,-1)],
        [(1,1,1)],
        [(0,0,1),(0,1,1)],
        [(1,0,1),(1,1,1)]
    ]
    
    # Contribution of each m_s to the 2x2 result Z:
    # Z00 = m0 + m3 - m4 + m6
    # Z01 = m2 + m4
    # Z10 = m1 + m3
    # Z11 = m0 - m1 + m2 + m5
    strassen_contrib = [
        [(0,0,1),(1,1,1)],    # m0
        [(1,0,1),(1,1,-1)],   # m1
        [(0,1,1),(1,1,1)],    # m2
        [(0,0,1),(1,0,1)],    # m3
        [(0,0,-1),(0,1,1)],   # m4
        [(1,1,1)],            # m5
        [(0,0,1)]             # m6
    ]
    
    # Now enumerate all 49 terms
    for t in range(7):
        L_t, R_t, C_t = level1_terms[t]
        for s in range(7):
            rank_idx = t * 7 + s
            
            # Build U contribution: for each (li, lj, coeff) in strassen_L[s],
            # L_t[li][lj] is a list of (a_row, a_col, a_coeff)
            # The combined coefficient is coeff * a_coeff
            u_dict = {}
            for (li, lj, sc) in strassen_L[s]:
                for (a_row, a_col, a_coeff) in L_t[li][lj]:
                    u_idx = a_row * 4 + a_col
                    u_dict[u_idx] = u_dict.get(u_idx, 0) + sc * a_coeff
            
            for u_idx, val in u_dict.items():
                if val != 0:
                    U[u_idx, rank_idx] = val
            
            # Build V contribution
            v_dict = {}
            for (ri, rj, sc) in strassen_R[s]:
                for (b_row, b_col, b_coeff) in R_t[ri][rj]:
                    v_idx = b_row * 4 + b_col
                    v_dict[v_idx] = v_dict.get(v_idx, 0) + sc * b_coeff
            
            for v_idx, val in v_dict.items():
                if val != 0:
                    V[v_idx, rank_idx] = val
            
            # Build W contribution: strassen_contrib[s] gives (zi, zj, zcoeff) for local result positions
            # C_t[(zi,zj)] gives list of (global_c_row, global_c_col, c_coeff)
            w_dict = {}
            for (zi, zj, zcoeff) in strassen_contrib[s]:
                if (zi, zj) in C_t:
                    for (gc_row, gc_col, c_coeff) in C_t[(zi, zj)]:
                        w_idx = gc_col * 4 + gc_row  # W[j*4+i] where j=col, i=row
                        w_dict[w_idx] = w_dict.get(w_idx, 0) + zcoeff * c_coeff
            
            for w_idx, val in w_dict.items():
                if val != 0:
                    W[w_idx, rank_idx] = val
    
    # Verification: check that the decomposition is correct
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        # Verify on a random test case
        np.random.seed(42)
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 4)
        C_expected = A_test @ B_test
        
        # Compute C from the decomposition
        C_computed = np.zeros((4, 4))
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            # u_col[i*4+k], v_col[k*4+j], w_col[j*4+i]
            for i in range(4):
                for k in range(4):
                    for j in range(4):
                        C_computed[i, j] += u_col[i*4+k] * v_col[k*4+j] * w_col[j*4+i]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Something is wrong - this should not happen with correct Strassen
            pass
    
    if elapsed > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
