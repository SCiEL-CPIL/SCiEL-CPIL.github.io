import time
import sys

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 4, 4, 7
    rank_strassen = 49
    rank_school = 48
    total_rank = rank_strassen + rank_school
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Strassen 4x4x4 ---
    # Base Strassen 2x2 vectors
    # u, v, w are lists of 7 vectors of length 4
    base_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    base_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    base_w = [
        [1, 1, 1, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 0, 1],
        [-1, 0, 0, -1]
    ]
    
    # Super steps for 4x4x4
    # Each step k defines coefficients for A blocks (2x2) and B blocks (2x2)
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: (0,0), (0,1), (1,0), (1,1)
    
    super_steps = [
        # k, a_coeffs (2x2), b_coeffs (2x2)
        (0, [[1, 0], [0, 1]], [[1, 0], [0, 1]]), # M1: (A00+A11)(B00+B11) -> A00, A11; B00, B11
        (1, [[0, 0], [1, 1]], [[1, 0], [0, 0]]), # M2: (A10+A11)B00
        (2, [[1, 0], [0, 0]], [[0, 1], [0, -1]]), # M3: A00(B01-B11)
        (3, [[0, 0], [0, 1]], [[0, 0], [1, -1]]), # M4: A11(B10-B00)
        (4, [[1, 1], [0, 0]], [[0, 0], [0, 1]]), # M5: (A00+A01)B11
        (5, [[-1, 0], [1, 0]], [[1, 1], [0, 0]]), # M6: (A10-A00)(B00+B01)
        (6, [[0, 1], [0, -1]], [[0, 0], [1, 1]])  # M7: (A01-A11)(B10+B11)
    ]
    
    col = 0
    for k, a_coeffs, b_coeffs in super_steps:
        for l in range(7):
            # l is the sub-step (0..6)
            # u_l, v_l, w_l are length 4 vectors for the 2x2 multiplication
            
            # Fill U
            # Global A index: block_r * 2 * m + block_c * 2 + local_r * m + local_c
            # Wait, A is 4x4. m=4.
            # Block (br, bc) contains A[br*2:br*2+2, bc*2:bc*2+2]
            # Element (br, bc, lr, lc) is at row br*2+lr, col bc*2+lc.
            # Index = (br*2+lr)*m + (bc*2+lc)
            
            for br in range(2):
                for bc in range(2):
                    coeff_a = a_coeffs[br][bc]
                    if coeff_a == 0: continue
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            coeff_u = base_u[l][local_idx]
                            if coeff_u == 0: continue
                            global_a_idx = (br * 2 + lr) * m + (bc * 2 + lc)
                            U[global_a_idx][col] += coeff_u * coeff_a
                            
            # Fill V
            # Global B index: block_r * 2 * p + block_c * 2 + local_r * p + local_c
            # B is 4x4 for this part. p=4.
            for br in range(2):
                for bc in range(2):
                    coeff_b = b_coeffs[br][bc]
                    if coeff_b == 0: continue
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            coeff_v = base_v[l][local_idx]
                            if coeff_v == 0: continue
                            global_b_idx = (br * 2 + lr) * p + (bc * 2 + lc)
                            V[global_b_idx][col] += coeff_v * coeff_b
                            
            # Fill W
            # Global C index: block_c * 2 * n + block_r * 2 + local_c * n + local_r
            # C is 4x4. n=4.
            # C[k*n+i]. k is col, i is row.
            # Block (cr, cc) contains C[cr*2:cr*2+2, cc*2:cc*2+2]
            # Element (cr, cc, lr, lc) is at row cr*2+lr, col cc*2+lc.
            # Index = (cc*2+lc)*n + (cr*2+lr)
            
            for cr in range(2):
                for cc in range(2):
                    # w_l corresponds to the C block of the 2x2 multiplication
                    # The 2x2 multiplication result is a 2x2 matrix.
                    # w_l[local_idx] is the coefficient for that C element.
                    # local_idx = lr * 2 + lc
                    # So for each local C element, we add to the global C element.
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            coeff_w = base_w[l][local_idx]
                            if coeff_w == 0: continue
                            global_c_idx = (cc * 2 + lc) * n + (cr * 2 + lr)
                            W[global_c_idx][col] += coeff_w
                            
            col += 1
            
    # --- Schoolbook 4x4x3 ---
    # A is 4x4. B2 is 4x3 (cols 4,5,6 of B).
    # C2 is 4x3 (cols 4,5,6 of C).
    for i in range(n):
        for j in range(m):
            for k in range(3):
                # B col is 4+k
                b_col = 4 + k
                # C col is 4+k
                c_col = 4 + k
                
                U[i * m + j][col] = 1
                V[j * p + b_col][col] = 1
                W[c_col * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
