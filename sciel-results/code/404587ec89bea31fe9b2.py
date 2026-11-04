"""
Matrix multiplication 2x4 by 4x4, exact tensor decomposition.
Rank-28 decomposition via Strassen's algorithm on four 2x2 sub-blocks.
The program is deterministic and fast; no time-budget search is required.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 150.0
START_TIME = time.time()


def check_time_and_return(best_result):
    if best_result is not None and (time.time() - START_TIME) > TIME_LIMIT - 10.0:
        return best_result
    return None


def construct():
    best_result = None
    try:
        best_result = _construct_strassen_28()
        if best_result is not None:
            if _verify_decomposition(best_result):
                return best_result
    except Exception:
        pass
    if best_result is None:
        best_result = _construct_schoolbook_32()
    return best_result


def _verify_decomposition(result):
    """Verify that the decomposition correctly computes matrix multiplication."""
    try:
        n, m, p = DIMENSIONS
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)

        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B

        A_flat = A.flatten()
        A_t = A_flat @ U
        B_flat = B.flatten()
        B_t = B_flat @ V
        C_flat = (A_t * B_t) @ W.T
        C_computed = C_flat.reshape((n, p))

        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
        return True
    except Exception:
        return False


def _construct_schoolbook_32():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _construct_strassen_28():
    """
    2x4 * 4x4 using four 2x2 Strassen blocks.
    Rank 28 (4 blocks x 7 Strassen multiplications).
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices for 2x2 block: 0:00, 1:01, 2:10, 3:11
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: -1, 1: 1, 2: 1}), # M1: C11+=1, C22+=1, C12-=1? No.
        # Let's use the standard mapping directly to avoid errors.
        # M1 -> C11, C22
        # M2 -> C21, C22
        # M3 -> C12, C22
        # M4 -> C11, C21
        # M5 -> C12, C11
        # M6 -> C22
        # M7 -> C11
        # Wait, standard Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
    ]
    
    # Correct Strassen terms (U coeffs, V coeffs, W coeffs)
    # W coeffs are for C11, C12, C21, C22 (indices 0,1,2,3)
    strassen_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    # Define the 4 sub-products
    # A is 2x4. Blocks: A11 (0:2, 0:2), A12 (0:2, 2:4), A21 (2:4, 0:2), A22 (2:4, 2:4)
    # B is 4x4. Blocks: B11 (0:2, 0:2), B12 (0:2, 2:4), B21 (2:4, 0:2), B22 (2:4, 2:4)
    # C is 2x4. Blocks: C11 (0:2, 0:2), C12 (0:2, 2:4), C21 (2:4, 0:2), C22 (2:4, 2:4)
    
    # We compute 4 independent 2x2x2 multiplications:
    # 1. A11 * B11 -> C11_part1
    # 2. A21 * B21 -> C11_part2? No.
    # C11 = A11 B11 + A21 B21.
    # So we need A11 B11 and A21 B21.
    # C12 = A11 B12 + A21 B22.
    # C21 = A12 B11 + A22 B21.
    # C22 = A12 B12 + A22 B22.
    
    # So the 4 products are:
    # P1 = A11 B11
    # P2 = A21 B21
    # P3 = A11 B12
    # P4 = A21 B22
    # P5 = A12 B11
    # P6 = A22 B21
    # P7 = A12 B12
    # P8 = A22 B22
    
    # Wait, C11 = P1 + P2. C12 = P3 + P4. C21 = P5 + P6. C22 = P7 + P8.
    # This requires 8 separate 2x2x2 multiplications?
    # No, the previous code grouped them differently.
    # Previous code:
    # sub_products[0]: A11 B11 -> C11
    # sub_products[1]: A21 B21 -> C11? No, c_map was c*2+r.
    # Let's re-verify the previous code's logic.
    
    # The previous code had 4 sub_products.
    # 0: A11 B11 -> C11
    # 1: A21 B21 -> C11?
    #    a_map: r*4 + (c+2) -> A21
    #    b_map: (r+2)*4 + c -> B21
    #    c_map: c*2 + r -> C11 (col 0)
    # 2: A11 B12 -> C12
    #    a_map: r*4 + c -> A11
    #    b_map: r*4 + (c+2) -> B12
    #    c_map: (c+2)*2 + r -> C12 (col 1)
    # 3: A21 B22 -> C12
    #    a_map: r*4 + (c+2) -> A21
    #    b_map: (r+2)*4 + (c+2) -> B22
    #    c_map: (c+2)*2 + r -> C12 (col 1)
    
    # This only covers C11 and C12. What about C21 and C22?
    # The previous code's `sub_products` list had 4 entries.
    # Entry 0: A11 B11 -> C11
    # Entry 1: A21 B21 -> C11
    # Entry 2: A11 B12 -> C12
    # Entry 3: A21 B22 -> C12
    # This computes C11 and C12.
    # But C is 2x4. C21 and C22 are missing!
    # Ah, the previous code's `c_map` for entry 1 was `c*2 + r`.
    # For 2x2 block, r,c in 0..1.
    # c=0, r=0 -> 0. c=0, r=1 -> 1. c=1, r=0 -> 2. c=1, r=1 -> 3.
    # This maps to C11 (0,1) and C21 (2,3)?
    # No, `c*2 + r` for 2x2 block:
    # (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
    # This is column-major order for the 2x2 block.
    # If the block is C11, it occupies C[0:2, 0:2].
    # Indices: 0,1,2,3.
    # If the block is C21, it occupies C[0:2, 2:4].
    # Indices: 4,5,6,7.
    
    # Let's check the previous code's `c_map` again.
    # sub_products[0] c_map: c*2+r. Maps to 0,1,2,3. (C11)
    # sub_products[1] c_map: c*2+r. Maps to 0,1,2,3. (C11 again?)
    # sub_products[2] c_map: (c+2)*2+r. Maps to 4,5,6,7. (C12)
    # sub_products[3] c_map: (c+2)*2+r. Maps to 4,5,6,7. (C12 again?)
    
    # So the previous code computed C11 and C12.
    # But what about C21 and C22?
    # The matrix C is 2x4.
    # C11 is top-left 2x2.
    # C12 is top-right 2x2.
    # C21 is bottom-left 2x2.
    # C22 is bottom-right 2x2.
    # The previous code only filled C11 and C12.
    # This is incorrect. The verification `_verify_decomposition` would fail?
    # Or did I misread the dimensions?
    # n=2, m=4, p=4.
    # A is 2x4. B is 4x4. C is 2x4.
    # C has 2 rows.
    # So there are no "bottom" rows in C beyond row 1.
    # C11 is C[0:2, 0:2].
    # C12 is C[0:2, 2:4].
    # That's all of C.
    # So C21 and C22 don't exist in the 2x4 output.
    # My confusion: I was thinking of 4x4 output.
    # For 2x4 output, we only have top row blocks.
    # So the previous code is correct for 2x4 output.
    
    # Let's re-verify the block definitions for 2x4 A and 4x4 B.
    # A:
    # A11: rows 0:2, cols 0:2.
    # A12: rows 0:2, cols 2:4.
    # B:
    # B11: rows 0:2, cols 0:2.
    # B12: rows 0:2, cols 2:4.
    # B21: rows 2:4, cols 0:2.
    # B22: rows 2:4, cols 2:4.
    
    # C11 = A11 B11 + A12 B21.
    # C12 = A11 B12 + A12 B22.
    
    # So we need 4 products:
    # P1 = A11 B11
    # P2 = A12 B21
    # P3 = A11 B12
    # P4 = A12 B22
    
    # The previous code's sub_products:
    # 0: A11 B11 -> C11
    # 1: A21 B21?
    #    a_map: r*4 + (c+2). A is 2x4.
    #    r in 0..1, c in 0..1.
    #    r*4 + c+2.
    #    (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7.
    #    This is A[0:2, 2:4] = A12.
    #    So sub_products[1] is A12 B21.
    #    b_map: (r+2)*4 + c.
    #    (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13.
    #    B is 4x4. Flattened.
    #    B[2:4, 0:2] = B21.
    #    So sub_products[1] is A12 B21.
    #    c_map: c*2+r. C11.
    #    So C11 = P1 + P2. Correct.
    
    # 2: A11 B12 -> C12
    # 3: A12 B22 -> C12
    #    So C12 = P3 + P4. Correct.
    
    # So the previous code is correct.
    
    # I will keep the previous code's logic as it is correct and rank 28.
    
    sub_products = [
        {
            'a_map': lambda r, c: r * 4 + c,       # A11
            'b_map': lambda r, c: r * 4 + c,       # B11
            'c_map': lambda r, c: c * 2 + r,       # C11
        },
        {
            'a_map': lambda r, c: r * 4 + (c + 2), # A12
            'b_map': lambda r, c: (r + 2) * 4 + c, # B21
            'c_map': lambda r, c: c * 2 + r,       # C11
        },
        {
            'a_map': lambda r, c: r * 4 + c,       # A11
            'b_map': lambda r, c: r * 4 + (c + 2), # B12
            'c_map': lambda r, c: (c + 2) * 2 + r, # C12
        },
        {
            'a_map': lambda r, c: r * 4 + (c + 2), # A12
            'b_map': lambda r, c: (r + 2) * 4 + (c + 2), # B22
            'c_map': lambda r, c: (c + 2) * 2 + r, # C12
        },
    ]

    col = 0
    for sp in sub_products:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for loc_u, coeff in u_coeffs.items():
                r, c = loc_u // 2, loc_u % 2
                global_u = sp['a_map'](r, c)
                U[global_u][col] += coeff
            for loc_v, coeff in v_coeffs.items():
                r, c = loc_v // 2, loc_v % 2
                global_v = sp['b_map'](r, c)
                V[global_v][col] += coeff
            for loc_w, coeff in w_coeffs.items():
                r, c = loc_w // 2, loc_w % 2
                global_w = sp['c_map'](r, c)
                W[global_w][col] += coeff
            col += 1

    assert col == 28, f"Expected col=28, got {col}"

    return {"U": U, "V": V, "W": W}
