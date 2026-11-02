DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using 6 independent 2x2x2 Strassen blocks
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Strassen coefficients for 2x2x2
    # M1: (X00+X11)(Y00+Y11) -> Z00+=1, Z11+=1
    # M2: (X10+X11)Y00       -> Z10+=1
    # M3: X00(Y01-Y11)        -> Z01+=1
    # M4: X11(Y10-Y00)        -> Z00+=1, Z10+=1
    # M5: (X00+X01)Y11        -> Z00-=1, Z01+=1
    # M6: (X10-X00)(Y00+Y01)  -> Z11+=1
    # M7: (X01-X11)(Y10+Y11)  -> Z00+=1, Z11-=1
    
    # Note: Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Mapping to 0-indexed:
    # X00, X01, X10, X11
    # Y00, Y01, Y10, Y11
    # Z00, Z01, Z10, Z11
    
    # M1: X00, X11 ; Y00, Y11 ; Z00(+1), Z11(+1)
    # M2: X10, X11 ; Y00 ; Z10(+1)
    # M3: X00 ; Y01, Y11(-1) ; Z01(+1)
    # M4: X11 ; Y10, Y00(-1) ; Z00(+1), Z10(+1)
    # M5: X00, X01 ; Y11 ; Z00(-1), Z01(+1)
    # M6: X10, X00(-1) ; Y00, Y01 ; Z11(+1)
    # M7: X01, X11(-1) ; Y10, Y11 ; Z00(+1), Z11(-1)

    strassen_blocks = [
        # (x_indices, coeffs, y_indices, coeffs, z_indices, coeffs)
        # M1
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # M2
        ([2, 3], [1, 1], [0], [1], [2], [1]),
        # M3
        ([0], [1], [1, 3], [1, -1], [1], [1]),
        # M4
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        # M5
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        # M6
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # M7
        ([1, 3], [1, -1], [2, 3], [1, 1], [0, 3], [1, -1]),
    ]

    for s in range(2): # 0, 1 (A col block / B row block)
        for j in range(3): # 0, 1, 2 (B col block / C col block)
            base_col = (s * 3 + j) * 7
            for t in range(7):
                col = base_col + t
                x_idx, x_coeff, y_idx, y_coeff, z_idx, z_coeff = strassen_blocks[t]
                
                for xi, cx in zip(x_idx, x_coeff):
                    # A_s is 2x2.
                    # Global A index: row r (0,1), col c (2s + xi_col)
                    # xi is 0..3 corresponding to [0,0], [0,1], [1,0], [1,1]
                    r = xi // 2
                    c_local = xi % 2
                    c_global = 2 * s + c_local
                    U[r * 4 + c_global][col] += cx
                
                for yi, cy in zip(y_idx, y_coeff):
                    # B_sj is 2x2.
                    # Global B index: row r' (2s + yi_row), col c' (2j + yi_col)
                    r_local = yi // 2
                    c_local = yi % 2
                    r_global = 2 * s + r_local
                    c_global = 2 * j + c_local
                    V[r_global * 6 + c_global][col] += cy
                
                for zi, cz in zip(z_idx, z_coeff):
                    # Z_sj is 2x2.
                    # Global C index: row i (zi_row), col k (2j + zi_col)
                    # W index: k * n + i
                    r_local = zi // 2
                    c_local = zi % 2
                    i_global = r_local
                    k_global = 2 * j + c_local
                    W[k_global * 2 + i_global][col] += cz

    return {"U": U, "V": V, "W": W}
