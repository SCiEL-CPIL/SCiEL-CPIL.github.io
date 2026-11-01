import time

def construct():
    """
    Rank-42 exact decomposition for 2x4 @ 4x6 matrix multiplication.
    
    Strategy: Decompose the multiplication into 6 blocks of 2x2 @ 2x2.
    A is 2x4, B is 4x6.
    Split A into two 2x2 blocks: A_L, A_R.
    Split B into 2x3 blocks of 2x2 matrices.
    C = A_L @ B[0:2, :] + A_R @ B[2:4, :]
    B[0:2, :] is 2x6, split into 3 2x2 blocks.
    B[2:4, :] is 2x6, split into 3 2x2 blocks.
    Total 6 multiplications of 2x2 @ 2x2.
    Each 2x2 @ 2x2 has rank 7 (Strassen).
    Total rank = 6 * 7 = 42.
    """
    n, m, p = 2, 4, 6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen 2x2x2 core
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A @ B
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6
    
    strassen_terms = [
        {'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        {'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)], 'W': [(1,0,1), (1,1,-1)]},
        {'A': [(0,0,1)], 'B': [(0,1,1), (1,1,-1)], 'W': [(0,1,1), (1,1,1)]},
        {'A': [(1,1,1)], 'B': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (1,0,1)]},
        {'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)], 'W': [(0,0,-1), (0,1,1)]},
        {'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
        {'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]}
    ]
    
    col = 0
    start_time = time.time()
    
    def apply_strassen(a_base, b_base, out_base):
        nonlocal col
        if time.time() - start_time > 110:
            return False
        for term in strassen_terms:
            for (ar, ac, sign) in term['A']:
                u_idx = (a_base[0] + ar) * m + (a_base[1] + ac)
                U[u_idx][col] += sign
            for (br, bc, sign) in term['B']:
                v_idx = (b_base[0] + br) * p + (b_base[1] + bc)
                V[v_idx][col] += sign
            for (or_, oc, sign) in term['W']:
                w_idx = (out_base[1] + oc) * n + (out_base[0] + or_)
                W[w_idx][col] += sign
            col += 1
        return True

    # We need to cover the 2x4 x 4x6 multiplication.
    # A is 2x4. Rows 0,1. Cols 0,1 (A_L) and 2,3 (A_R).
    # B is 4x6. Rows 0,1 (B_T) and 2,3 (B_B). Cols 0,1; 2,3; 4,5.
    # C is 2x6.
    
    # Contribution from A_L (rows 0,1 of A, cols 0,1) @ B_T (rows 0,1 of B)
    # This contributes to C cols 0,1; 2,3; 4,5?
    # No, A_L is 2x2. B_T is 2x6.
    # A_L @ B_T is 2x6.
    # We can split B_T into 3 2x2 blocks.
    
    # Block 1: A_L @ B_T[:, 0:2] -> C[:, 0:2]
    if not apply_strassen((0, 0), (0, 0), (0, 0)):
        # Timeout, return best effort (though construction is fast)
        pass
    
    # Block 2: A_L @ B_T[:, 2:4] -> C[:, 2:4]
    if not apply_strassen((0, 0), (0, 2), (0, 2)):
        pass
        
    # Block 3: A_L @ B_T[:, 4:6] -> C[:, 4:6]
    if not apply_strassen((0, 0), (0, 4), (0, 4)):
        pass
        
    # Block 4: A_R (rows 0,1 of A, cols 2,3) @ B_B (rows 2,3 of B)
    # A_R @ B_B[:, 0:2] -> C[:, 0:2]
    if not apply_strassen((0, 2), (2, 0), (0, 0)):
        pass
        
    # Block 5: A_R @ B_B[:, 2:4] -> C[:, 2:4]
    if not apply_strassen((0, 2), (2, 2), (0, 2)):
        pass
        
    # Block 6: A_R @ B_B[:, 4:6] -> C[:, 4:6]
    if not apply_strassen((0, 2), (2, 4), (0, 4)):
        pass

    if col != rank:
        # If timeout occurred early, rank is wrong. 
        # But 42 terms is very fast.
        pass

    return {"U": U, "V": V, "W": W}
