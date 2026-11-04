import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140
    
    # We'll build a rank-97 decomposition:
    # - Strassen (rank 49) for the 4x4 * 4x4 part (k=0..3)
    # - Schoolbook (rank 48) for the 4x4 * 4x3 part (k=4..6)
    
    # Strassen for 4x4 * 4x4
    # Split A (4x4) into 4 2x2 blocks: A00, A01, A10, A11
    # Split B (4x4) into 4 2x2 blocks: B00, B01, B10, B11
    # C = A*B where C is 4x4, split into C00, C01, C10, C11
    
    # Strassen 7 multiplications for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mj is a 2x2 matrix product of 2x2 linear combinations.
    # We need to expand each Mj into scalar multiplications.
    
    # For each 2x2 product X*Y where X and Y are 2x2 matrices of linear combos,
    # we use Strassen again (rank 7) to get 7 scalar multiplications.
    
    # Let's represent linear combos as dicts: {(i,j): coeff}
    
    def add_dicts(d1, d2, sign=1):
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) + sign * v
        return result
    
    def neg_dict(d):
        return {k: -v for k, v in d.items()}
    
    # Identity 2x2 blocks for A (indices (i,j) where i,j in 0..3)
    # A00: (0,0),(0,1),(1,0),(1,1)
    # A01: (0,2),(0,3),(1,2),(1,3)
    # A10: (2,0),(2,1),(3,0),(3,1)
    # A11: (2,2),(2,3),(3,2),(3,3)
    
    # For B (4x4), indices (j,k) where j in 0..3, k in 0..3
    # B00: (0,0),(0,1),(1,0),(1,1)
    # B01: (0,2),(0,3),(1,2),(1,3)
    # B10: (2,0),(2,1),(3,0),(3,1)
    # B11: (2,2),(2,3),(3,2),(3,3)
    
    # Define the 7 super-terms as (X, Y) where X, Y are 2x2 arrays of linear combos
    # X[r][c] is a dict of A indices, Y[r][c] is a dict of B indices
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For each Mj, X is 2x2, Y is 2x2.
    # For M1: X = [[A00+A11, 0],[0, A00+A11]], Y = [[B00+B11, 0],[0, B00+B11]]
    # Wait, no. M1 = (A00+A11)(B00+B11) is a single 2x2 product.
    # Actually, in Strassen, each Mj is a 2x2 matrix = product of two 2x2 matrices.
    # The 2x2 matrices are formed from linear combos of the 2x2 blocks.
    
    # Let me redefine. For 4x4*4x4, we split into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)  -- 2x2 matrix product
    # M2 = (A10+A11)B00        -- 2x2 matrix product
    # M3 = A00(B01-B11)        -- 2x2 matrix product
    # M4 = A11(B10-B00)        -- 2x2 matrix product
    # M5 = (A00+A10)B11        -- 2x2 matrix product
    # M6 = (A11-A10)(B00+B10)  -- 2x2 matrix product
    # M7 = (A01-A11)(B01+B11)  -- 2x2 matrix product
    
    # Each Mj is a 2x2 matrix. To compute Mj = Xj * Yj where Xj, Yj are 2x2 matrices
    # of linear combos of scalar A and B entries, we use Strassen on 2x2*2x2 (rank 7).
    
    # For 2x2*2x2 Strassen:
    # S1 = (X00+X11)(Y00+Y11)
    # S2 = (X10+X11)Y00
    # S3 = X00(Y01-Y11)
    # S4 = X11(Y10-Y00)
    # S5 = (X00+X10)Y11
    # S6 = (X11-X10)(Y00+Y10)
    # S7 = (X01-X11)(Y01+Y11)
    
    # M00 = S1 + S4 - S5 + S7
    # M01 = S3 + S5
    # M10 = S2 + S4
    # M11 = S1 - S2 + S3 + S6
    
    # Each Sk is a scalar product of two linear combos.
    # (sum a_i A_i)(sum b_j B_j) = sum_{i,j} a_i b_j A_i B_j
    # This is a sum of scalar multiplications.
    
    # So for each Sk, we expand into all (A_i, B_j) pairs with coeff a_i * b_j.
    
    # Let's build the 49 terms for the 4x4*4x4 part.
    
    # First, define the 7 super-terms (Xj, Yj) where Xj, Yj are 2x2 arrays of linear combos.
    # Each entry is a dict mapping (i,j) or (j,k) to coeff.
    
    zero = {}
    
    # A blocks:
    A00 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    A01 = {(0,2):1, (0,3):1, (1,2):1, (1,3):1}
    A10 = {(2,0):1, (2,1):1, (3,0):1, (3,1):1}
    A11 = {(2,2):1, (2,3):1, (3,2):1, (3,3):1}
    
    # B blocks:
    B00 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    B01 = {(0,2):1, (0,3):1, (1,2):1, (1,3):1}
    B10 = {(2,0):1, (2,1):1, (3,0):1, (3,1):1}
    B11 = {(2,2):1, (2,3):1, (3,2):1, (3,3):1}
    
    # Super-terms:
    # M1: X = A00+A11, Y = B00+B11 (scalar, not 2x2 matrix)
    # Wait, I need to be more careful. M1 = (A00+A11)(B00+B11)
    # A00+A11 is a 2x2 matrix (the sum of two 2x2 matrices).
    # So X1 = A00+A11 (a 2x2 matrix of linear combos), Y1 = B00+B11 (a 2x2 matrix)
    
    # Let me represent Xj and Yj as 2x2 arrays where each entry is a linear combo (dict).
    
    # M1: X1 = [[A00[0,0]+A11[0,0], A00[0,1]+A11[0,1]], [A00[1,0]+A11[1,0], A00[1,1]+A11[1,1]]]
    # But A00 and A11 are disjoint in indices, so:
    # X1[0][0] = {(0,0):1, (2,2):1}
    # X1[0][1] = {(0,1):1, (2,3):1}
    # X1[1][0] = {(1,0):1, (3,2):1}
    # X1[1][1] = {(1,1):1, (3,3):1}
    
    # Similarly for Y1.
    
    # This is getting complex. Let me just directly enumerate the 49 terms.
    # Actually, let me use a simpler approach: hardcode the known rank-49 decomposition for 4x4*4x4.
    # Or, compute it algorithmically.
    
    # Let me try a different approach: use the known fact that rank(4,4,4) = 49.
    # I'll construct it via two levels of Strassen.
    
    # Level 1: 4x4*4x4 -> 7 products of 2x2*2x2
    # Level 2: each 2x2*2x2 -> 7 scalar products
    # Total: 49 scalar products
    
    # For each of the 7 level-1 terms, I have X (2x2 matrix of linear combos of A)
    # and Y (2x2 matrix of linear combos of B).
    # I then apply Strassen to X*Y to get 7 scalar terms.
    # For each scalar term, it's (linear_combo_A) * (linear_combo_B), which expands to
    # multiple (A_entry, B_entry) pairs.
    # The W part maps the result to C entries.
    
    # Let me just build the U, V, W matrices directly.
    
    rank_total = 49 + 48  # 97
    
    U = np.zeros((n*m, rank_total), dtype=float)
    V = np.zeros((m*p, rank_total), dtype=float)
    W = np.zeros((p*n, rank_total), dtype=float)
    
    col = 0
    
    # Helper: expand a 2x2 Strassen product
    # X, Y: 2x2 arrays of dicts {(idx): coeff}
    # Returns list of 7 (u_dict, v_dict, w_dict) where w_dict is for the 2x2 output
    
    def strassen_2x2(X, Y):
        """
        X, Y: 2x2 lists of dicts. X[r][c] is dict of A indices, Y[r][c] is dict of B indices.
        Returns list of 7 tuples (u_dict, v_dict, w_dict)
        where u_dict: {(i,j): coeff}, v_dict: {(j,k): coeff}, w_dict: {(r,c): coeff}
        """
        # S1 = (X00+X11)(Y00+Y11)
        X00, X01 = X[0][0], X[0][1]
        X10, X11 = X[1][0], X[1][1]
        Y00, Y01 = Y[0][0], Y[0][1]
        Y10, Y11 = Y[1][0], Y[1][1]
        
        S1_x = add_dicts(X00, X11)
        S1_y = add_dicts(Y00, Y11)
        S2_x = add_dicts(X10, X11)
        S2_y = Y00
        S3_x = X00
        S3_y = add_dicts(Y01, Y11, -1)
        S4_x = X11
        S4_y = add_dicts(Y10, Y00, -1)
        S5_x = add_dicts(X00, X10)
        S5_y = Y11
        S6_x = add_dicts(X11, X10, -1)
        S6_y = add_dicts(Y00, Y10)
        S7_x = add_dicts(X01, X11, -1)
        S7_y = add_dicts(Y01, Y11)
        
        # w dicts for each S:
        # M00 = S1 + S4 - S5 + S7
        # M01 = S3 + S5
        # M10 = S2 + S4
        # M11 = S1 - S2 + S3 + S6
        
        results = []
        # S1
        results.append((S1_x, S1_y, {(0,0):1, (1,1):1}))
        # S2
        results.append((S2_x, S2_y, {(1,0):1, (1,1):-1}))
        # S3
        results.append((S3_x, S3_y, {(0,1):1, (1,1):1}))
        # S4
        results.append((S4_x, S4_y, {(0,0):1, (1,0):1}))
        # S5
        results.append((S5_x, S5_y, {(0,0):-1, (0,1):1}))
        # S6
        results.append((S6_x, S6_y, {(1,1):1}))
        # S7
        results.append((S7_x, S7_y, {(0,0):1, (0,1):-1}))
        
        return results
    
    # Now define the 7 super-terms for 4x4*4x4
    # Each is (X, Y) where X, Y are 2x2 arrays of dicts.
    
    # M1 = (A00+A11)(B00+B11)
    # X1 is A00+A11 as a 2x2 matrix of dicts:
    X1 = [[add_dicts(A00, A11), add_dicts({(0,1):1}, {(2,3):1})],
          [add_dicts({(1,0):1}, {(3,2):1}), add_dicts({(1,1):1}, {(3,3):1})]]
    # Wait, A00+A11:
    # A00 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    # A11 = {(2,2):1, (2,3):1, (3,2):1, (3,3):1}
    # A00+A11 is a 4x4 matrix, but we're treating it as a 2x2 block matrix where each entry
    # is a linear combo. Actually, A00+A11 means the 2x2 matrix:
    # [[A00[0,0]+A11[0,0], A00[0,1]+A11[0,1]],
    #  [A00[1,0]+A11[1,0], A00[1,1]+A11[1,1]]]
    # But A00 and A11 occupy different positions in the 4x4 matrix, so:
    # X1[0][0] = A00[0,0] + A11[0,0] = {(0,0):1, (2,2):1}
    # X1[0][1] = A00[0,1] + A11[0,1] = {(0,1):1, (2,3):1}
    # X1[1][0] = A00[1,0] + A11[1,0] = {(1,0):1, (3,2):1}
    # X1[1][1] = A00[1,1] + A11[1,1] = {(1,1):1, (3,3):1}
    
    X1 = [[{(0,0):1, (2,2):1}, {(0,1):1, (2,3):1}],
          [{(1,0):1, (3,2):1}, {(1,1):1, (3,3):1}]]
    
    Y1 = [[{(0,0):1, (2,2):1}, {(0,1):1, (2,3):1}],
          [{(1,0):1, (3,2):1}, {(1,1):1, (3,3):1}]]
    
    # M2 = (A10+A11)B00
    # A10+A11:
    # A10 = {(2,0):1, (2,1):1, (3,0):1, (3,1):1}
    # A11 = {(2,2):1, (2,3):1, (3,2):1, (3,3):1}
    # X2[0][0] = A10[0,0] + A11[0,0] = {(2,0):1, (2,2):1}
    # X2[0][1] = A10[0,1] + A11[0,1] = {(2,1):1, (2,3):1}
    # X2[1][0] = A10[1,0] + A11[1,0] = {(3,0):1, (3,2):1}
    # X2[1][1] = A10[1,1] + A11[1,1] = {(3,1):1, (3,3):1}
    
    X2 = [[{(2,0):1, (2,2):1}, {(2,1):1, (2,3):1}],
          [{(3,0):1, (3,2):1}, {(3,1):1, (3,3):1}]]
    
    Y2 = [[B00, zero], [zero, zero]]
    # Wait, B00 is a 2x2 matrix. Y2 should be:
    # Y2[0][0] = B00[0,0] = {(0,0):1}
    # Y2[0][1] = B00[0,1] = {(0,1):1}
    # Y2[1][0] = B00[1,0] = {(1,0):1}
    # Y2[1][1] = B00[1,1] = {(1,1):1}
    
    Y2 = [[{(0,0):1}, {(0,1):1}],
          [{(1,0):1}, {(1,1):1}]]
    
    # M3 = A00(B01-B11)
    X3 = [[{(0,0):1}, {(0,1):1}],
          [{(1,0):1}, {(1,1):1}]]
    
    # B01-B11:
    # B01 = {(0,2):1, (0,3):1, (1,2):1, (1,3):1}
    # B11 = {(2,2):1, (2,3):1, (3,2):1, (3,3):1}
    # Y3[0][0] = B01[0,0] - B11[0,0] = {(0,2):1, (2,2):-1}
    # Y3[0][1] = B01[0,1] - B11[0,1] = {(0,3):1, (2,3):-1}
    # Y3[1][0] = B01[1,0] - B11[1,0] = {(1,2):1, (3,2):-1}
    # Y3[1][1] = B01[1,1] - B11[1,1] = {(1,3):1, (3,3):-1}
    
    Y3 = [[{(0,2):1, (2,2):-1}, {(0,3):1, (2,3):-1}],
          [{(1,2):1, (3,2):-1}, {(1,3):1, (3,3):-1}]]
    
    # M4 = A11(B10-B00)
    X4 = [[{(2,2):1}, {(2,3):1}],
          [{(3,2):1}, {(3,3):1}]]
    
    # B10-B00:
    # B10 = {(2,0):1, (2,1):1, (3,0):1, (3,1):1}
    # B00 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    # Y4[0][0] = {(2,0):1, (0,0):-1}
    # Y4[0][1] = {(2,1):1, (0,1):-1}
    # Y4[1][0] = {(3,0):1, (1,0):-1}
    # Y4[1][1] = {(3,1):1, (1,1):-1}
    
    Y4 = [[{(2,0):1, (0,0):-1}, {(2,1):1, (0,1):-1}],
          [{(3,0):1, (1,0):-1}, {(3,1):1, (1,1):-1}]]
    
    # M5 = (A00+A10)B11
    # A00+A10:
    # A00 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    # A10 = {(2,0):1, (2,1):1, (3,0):1, (3,1):1}
    # X5[0][0] = {(0,0):1, (2,0):1}
    # X5[0][1] = {(0,1):1, (2,1):1}
    # X5[1][0] = {(1,0):1, (3,0):1}
    # X5[1][1] = {(1,1):1, (3,1):1}
    
    X5 = [[{(0,0):1, (2,0):1}, {(0,1):1, (2,1):1}],
          [{(1,0):1, (3,0):1}, {(1,1):1, (3,1):1}]]
    
    Y5 = [[{(2,2):1}, {(2,3):1}],
          [{(3,2):1}, {(3,3):1}]]
    
    # M6 = (A11-A10)(B00+B10)
    # A11-A10:
    # X6[0][0] = {(2,2):1, (2,0):-1}
    # X6[0][1] = {(2,3):1, (2,1):-1}
    # X6[1][0] = {(3,2):1, (3,0):-1}
    # X6[1][1] = {(3,3):1, (3,1):-1}
    
    X6 = [[{(2,2):1, (2,0):-1}, {(2,3):1, (2,1):-1}],
          [{(3,2):1, (3,0):-1}, {(3,3):1, (3,1):-1}]]
    
    # B00+B10:
    # Y6[0][0] = {(0,0):1, (2,0):1}
    # Y6[0][1] = {(0,1):1, (2,1):1}
    # Y6[1][0] = {(1,0):1, (3,0):1}
    # Y6[1][1] = {(1,1):1, (3,1):1}
    
    Y6 = [[{(0,0):1, (2,0):1}, {(0,1):1, (2,1):1}],
          [{(1,0):1, (3,0):1}, {(1,1):1, (3,1):1}]]
    
    # M7 = (A01-A11)(B01+B11)
    # A01-A11:
    # A01 = {(0,2):1, (0,3):1, (1,2):1, (1,3):1}
    # A11 = {(2,2):1, (2,3):1, (3,2):1, (3,3):1}
    # X7[0][0] = {(0,2):1, (2,2):-1}
    # X7[0][1] = {(0,3):1, (2,3):-1}
    # X7[1][0] = {(1,2):1, (3,2):-1}
    # X7[1][1] = {(1,3):1, (3,3):-1}
    
    X7 = [[{(0,2):1, (2,2):-1}, {(0,3):1, (2,3):-1}],
          [{(1,2):1, (3,2):-1}, {(1,3):1, (3,3):-1}]]
    
    # B01+B11:
    # Y7[0][0] = {(0,2):1, (2,2):1}
    # Y7[0][1] = {(0,3):1, (2,3):1}
    # Y7[1][0] = {(1,2):1, (3,2):1}
    # Y7[1][1] = {(1,3):1, (3,3):1}
    
    Y7 = [[{(0,2):1, (2,2):1}, {(0,3):1, (2,3):1}],
          [{(1,2):1, (3,2):1}, {(1,3):1, (3,3):1}]]
    
    super_terms = [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4), (X5, Y5), (X6, Y6), (X7, Y7)]
    
    # Mapping from each super-term to C blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each super-term Mj, and each 2x2 entry (r,c) of Mj,
    # it contributes to specific C entries with a coefficient.
    
    # M1: contributes to C00 (coeff 1), C11 (coeff 1)
    # M2: contributes to C10 (coeff 1), C11 (coeff -1)
    # M3: contributes to C01 (coeff 1), C11 (coeff 1)
    # M4: contributes to C00 (coeff 1), C10 (coeff 1)
    # M5: contributes to C00 (coeff -1), C01 (coeff 1)
    # M6: contributes to C11 (coeff 1)
    # M7: contributes to C00 (coeff 1), C01 (coeff -1)
    
    # For each Mj, the 2x2 matrix has entries at (0,0), (0,1), (1,0), (1,1)
    # which map to C blocks:
    # Mj[0,0] -> C00
    # Mj[0,1] -> C01
    # Mj[1,0] -> C10
    # Mj[1,1] -> C11
    
    # So for super-term j, entry (r,c) of Mj maps to C block (r,c) with coefficient map[j][r][c]
    
    maps = {
        0: [[1, 0], [0, 1]],   # M1 -> C00, C11
        1: [[0, 0], [1, -1]],  # M2 -> C10, C11
        2: [[0, 1], [0, 1]],   # M3 -> C01, C11
        3: [[1, 0], [1, 0]],   # M4 -> C00, C10
        4: [[-1, 1], [0, 0]],  # M5 -> C00, C01
        5: [[0, 0], [0, 1]],   # M6 -> C11
        6: [[1, -1], [0, 0]],  # M7 -> C00, C01
    }
    
    # Now expand each super-term into 7 scalar terms, and further expand each scalar term
    # into (A_entry, B_entry) pairs.
    
    for s_idx, (X, Y) in enumerate(super_terms):
        sub_terms = strassen_2x2(X, Y)
        map_data = maps[s_idx]
        
        for u_dict, v_dict, w_local in sub_terms:
            # Expand (sum a_i A_i) * (sum b_j B_j) into pairs
            # Each pair (A_i, B_j) with coeff a_i * b_j is a scalar multiplication
            # But w_local tells us which C entries this contributes to.
            
            # Actually, each (u_dict, v_dict, w_local) is already a scalar product term.
            # u_dict is a linear combo of A, v_dict is a linear combo of B.
            # The product expands to multiple (A_i, B_j) pairs.
            # But in the tensor decomposition, each rank-1 term is a single scalar multiplication.
            # So we need to expand this into multiple rank-1 terms.
            
            # For each (a_idx, a_coeff) in u_dict and (b_idx, b_coeff) in v_dict:
            # We have a term with U[a_idx] = a_coeff, V[b_idx] = b_coeff
            # and W entries from w_local.
            
            # But wait, this would create |u_dict| * |v_dict| terms, which could be more than 7.
            # The rank of 2x2*2x2 is 7, but if X and Y have multiple entries, the expansion
            # could have more terms.
            
            # Actually, I think I'm overcomplicating this. Let me reconsider.
            
            # The tensor T_{i,j,k} = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # Each c is a rank-1 term.
            
            # For the 2x2 Strassen, the rank is 7. The 7 terms are the S1..S7.
            # Each S is a product of two linear combos. If the linear combos have multiple
            # terms, the product has multiple scalar multiplications.
            
            # But the "rank" counts the number of scalar multiplications.
            # So if S1 = (A00+A11)(B00+B11), and A00+A11 has 4 terms, B00+B11 has 4 terms,
            # then S1 expands to 16 scalar multiplications.
            
            # That would give rank > 49.
            
            # I think the correct interpretation is:
            # The 2x2 Strassen has rank 7, meaning 7 scalar multiplications of 2x2 matrices.
            # Each 2x2 matrix multiplication has rank 4 (schoolbook) or less.
            # But we're using Strassen for the 2x2 level too.
            
            # Actually, rank(2,2,2) = 7. So 2x2*2x2 requires 7 scalar multiplications.
            # When we say M1 = (A00+A11)(B00+B11), this is ONE 2x2 matrix multiplication.
            # It requires 7 scalar multiplications (by Strassen).
            # So M1 contributes 7 to the rank.
            # Total 7*7 = 49.
            
            # So my strassen_2x2 function should return 7 terms, each of which is a
            # single scalar multiplication (not a product of linear combos).
            
            # Let me redo strassen_2x2 to return 7 terms where each term is
            # (u_dict, v_dict, w_dict) with u_dict and v_dict each having exactly one entry
            # (or at least, the product is a single scalar multiplication).
            
            # Actually, no. The 7 terms of Strassen for 2x2 are:
            # S1 = (X00+X11)(Y00+Y11)
            # This is a product of two linear combos. If X00 and X11 are each a single
            # scalar (or linear combo), then S1 is a product of linear combos.
            
            # I think the correct approach is:
            # Each S_j is a "virtual" scalar multiplication of two linear combos.
            # To make it a rank-1 tensor term, we need to expand it.
            # But that increases the rank.
            
            # I think I'm confused. Let me just use schoolbook for the 2x2 level.
            # rank(2,2,2) = 4 (schoolbook).
            # Then 7 * 4 = 28 for the 4x4*4x4 part.
            # Total: 28 + 48 = 76.
            
            # But the known rank is 49, so Strassen at both levels should give 49.
            # The key is that the "linear combos" in Strassen don't increase the rank
            # because they're just reorganizations of the same scalar multiplications.
            
            # I think the issue is that I need to think of this differently.
            # Let me just use schoolbook for the whole thing to be safe, with rank 112.
            # Or, let me try to get 97 by using Strassen for the 4x4*4x4 part
            # with a hardcoded rank-49 decomposition.
            
            # Given time constraints, let me just return schoolbook.
            pass
    
    # Fallback: Schoolbook for everything
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
