import numpy as np
import time

DIMENSIONS = (4, 4, 7)

def construct():
    start_time = time.time()
    DEADLINE = 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    nm, mp, pn = n * m, m * p, p * n  # 16, 28, 28
    
    # Part 1: 4x4x4 decomposition using Strassen (Rank 49)
    # Part 2: 4x4x3 decomposition using Winograd 2x2x3 (Rank 20)
    rank_444 = 49
    rank_443 = 20
    total_rank = rank_444 + rank_443
    
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    # --- Part 1: 4x4x4 ---
    # We decompose the 4x4 multiplication into 7 Strassen products of 2x2 blocks.
    # Each 2x2 product is decomposed into 7 Winograd (Strassen) products of scalars.
    # Total 49 scalar products.
    
    # Strassen definitions for 2x2 block multiplication
    # X is 2x2, Y is 2x2, Z is 2x2
    # s: index 0-6
    # X_def: list of (row, col, coeff) for X
    # Y_def: list of (row, col, coeff) for Y
    # Z_def: list of (row, col, coeff) for Z
    
    strassen_2x2 = [
        # 0: M1 = (X00 + X11)(Y00 + Y11) -> Z00 += M1, Z11 += M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # 1: M2 = (X10 + X11)Y00 -> Z10 += M2, Z11 -= M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # 2: M3 = X00(Y01 - Y11) -> Z01 += M3, Z11 += M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # 3: M4 = X11(Y10 - Y00) -> Z00 += M4, Z10 += M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # 4: M5 = (X00 + X01)Y11 -> Z00 -= M5, Z01 += M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # 5: M6 = (X10 - X00)(Y00 + Y01) -> Z11 += M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # 6: M7 = (X01 - X11)(Y10 + Y11) -> Z00 += M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    for s_idx, (X_def, Y_def, Z_def) in enumerate(strassen_2x2):
        for w_idx, (u_def, v_def, out_def) in enumerate(strassen_2x2):
            r = s_idx * 7 + w_idx
            
            # U column: Coefficients for A
            U_col = np.zeros(nm)
            for (bx, by, cx) in X_def:
                for (i_prime, j_prime, cu) in u_def:
                    i = 2 * bx + i_prime
                    j = 2 * by + j_prime
                    a_idx = i * m + j
                    U_col[a_idx] += cx * cu
            U[:, r] = U_col
            
            # V column: Coefficients for B
            V_col = np.zeros(mp)
            for (bx, by, cy) in Y_def:
                for (i_prime, j_prime, cv) in v_def:
                    row_b = 2 * bx + i_prime
                    col_b = 2 * by + j_prime
                    b_idx = row_b * p + col_b
                    V_col[b_idx] += cy * cv
            V[:, r] = V_col
            
            # W column: Coefficients for C
            W_col = np.zeros(pn)
            for (dx, dy, cd) in Z_def:
                for (i_prime, j_prime, co) in out_def:
                    row_c = 2 * dx + i_prime
                    col_c = 2 * dy + j_prime
                    c_idx = col_c * n + row_c
                    W_col[c_idx] += cd * co
            W[:, r] = W_col
            
    # --- Part 2: 4x4x3 ---
    # Use Winograd 2x2x3 decomposition for each 2x2 block of A and B
    # B is 4x3 (cols 4-6 of the original 4x7 matrix)
    # A is 4x4. We split A into 2x2 blocks: A00, A01, A10, A11
    # B is 4x3. We split B into 2x2 blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, col 2), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, col 2)
    # Wait, B is 4x3. So B has rows 0-3 and cols 0-2.
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:3] (2x1)
    # B10 = B[2:4, 0:2], B11 = B[2:4, 2:3] (2x1)
    # C is 4x3. C00 = C[0:2, 0:2], C01 = C[0:2, 2:3], C10 = C[2:4, 0:2], C11 = C[2:4, 2:3]
    
    # Winograd 2x2x3 decomposition:
    # X is 2x2, Y is 2x3, Z is 2x3
    # 5 scalar products:
    # M1 = (X00 + X11)(Y00 + Y10) -> Z00 += M1, Z10 += M1
    # M2 = (X10 + X11)Y00 -> Z10 += M2, Z11 -= M2
    # M3 = X00(Y01 - Y11) -> Z01 += M3, Z11 += M3
    # M4 = X11(Y10 - Y00) -> Z00 += M4, Z10 += M4
    # M5 = (X00 + X01)Y11 -> Z00 -= M5, Z01 += M5
    
    winograd_2x2x3 = [
        # 0: M1 = (X00 + X11)(Y00 + Y10) -> Z00 += M1, Z10 += M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,0,1)], [(0,0,1), (1,0,1)]),
        # 1: M2 = (X10 + X11)Y00 -> Z10 += M2, Z11 -= M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # 2: M3 = X00(Y01 - Y11) -> Z01 += M3, Z11 += M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # 3: M4 = X11(Y10 - Y00) -> Z00 += M4, Z10 += M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # 4: M5 = (X00 + X01)Y11 -> Z00 -= M5, Z01 += M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
    ]
    
    # For each 2x2 block of A (bx, by), we need to multiply by the corresponding 2x3 block of B
    # A block (bx, by) is A[2*bx:2*bx+2, 2*by:2*by+2]
    # B block for row 2*bx: 2*bx+1 is B[2*bx:2*bx+2, 0:3]
    # But we need to match the column indices correctly.
    # Actually, the 4x4x3 multiplication is:
    # C[i, k] = sum_j A[i, j] B[j, k] where k is 0-2 (local), so global col is 4+k
    # A[i, j] is global. B[j, 4+k] is global. C[4+k, i] is global.
    
    # Let's define the blocks:
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B0 (rows 0-1, cols 4-6), B1 (rows 2-3, cols 4-6)
    # C blocks: C0 (rows 4-6, cols 0-1), C1 (rows 4-6, cols 2-3)
    
    # For each A block (bx, by), we multiply by B block bx (rows 2*bx to 2*bx+1, cols 4-6)
    # The result goes to C block (bx, by) (rows 4-6, cols 2*by to 2*by+1)
    
    offset = rank_444
    col = 0
    
    for bx in range(2):
        for by in range(2):
            for w_idx, (X_def, Y_def, Z_def) in enumerate(winograd_2x2x3):
                r = offset + col
                col += 1
                
                # U column: Coefficients for A block (bx, by)
                U_col = np.zeros(nm)
                for (i_prime, j_prime, cu) in X_def:
                    i = 2 * bx + i_prime
                    j = 2 * by + j_prime
                    a_idx = i * m + j
                    U_col[a_idx] += cu
                U[:, r] = U_col
                
                # V column: Coefficients for B block (bx, cols 4-6)
                # Y is 2x3. Y[i_prime, k_prime] where k_prime is 0-2
                # B row is 2*bx + i_prime, B col is 4 + k_prime
                V_col = np.zeros(mp)
                for (i_prime, k_prime, cv) in Y_def:
                    row_b = 2 * bx + i_prime
                    col_b = 4 + k_prime
                    b_idx = row_b * p + col_b
                    V_col[b_idx] += cv
                V[:, r] = V_col
                
                # W column: Coefficients for C block (bx, by)
                # Z is 2x3. Z[i_prime, k_prime] where k_prime is 0-2
                # C row is 4 + k_prime, C col is 2*by + i_prime
                W_col = np.zeros(pn)
                for (i_prime, k_prime, co) in Z_def:
                    row_c = 4 + k_prime
                    col_c = 2 * by + i_prime
                    c_idx = col_c * n + row_c
                    W_col[c_idx] += co
                W[:, r] = W_col
                
    # Validation
    if time.time() - start_time < DEADLINE - 5:
        # Check target tensor entries
        for _ in range(100):
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            a = i * m + j
            b = j * p + k
            c = k * n + i
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val - 1.0) > 1e-9:
                raise ValueError(f"Validation failed for target entry ({a},{b},{c}): {val}")
        
        # Check a sample of non-target entries
        for _ in range(100):
            a = np.random.randint(0, nm)
            b = np.random.randint(0, mp)
            c = np.random.randint(0, pn)
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_c = c // n
            i_c = c % n
            
            if not (i_a == i_c and j_a == j_b and k_b == k_c):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > 1e-9:
                    raise ValueError(f"Validation failed for non-target entry ({a},{b},{c}): {val}")

    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
