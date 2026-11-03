import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Strassen components for 2x2x2
    # local_a: 0=A11, 1=A12, 2=A21, 3=A22
    # local_b: 0=B11, 1=B12, 2=B21, 3=B22
    # local_w: 0=C11, 1=C12, 2=C21, 3=C22
    strassen = [
        ([(0, 1), (1, 1)], [(0, 1), (1, 1)], [(0, 1), (1, 1)]),   # M1: (A11+A12)(B11+B12) -> C11, C12
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),           # M2: (A21+A22)B11 -> C21, C22
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),           # M3: A11(B12-B22) -> C12, C22
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),           # M4: A22(B21-B11) -> C11, C21
        ([(0, 1), (2, 1)], [(0, 1)], [(0, -1), (2, 1)]),           # M5: (A11+A21)B11 -> C11, C21
        ([(1, 1), (3, -1)], [(0, 1), (2, 1)], [(3, 1)]),           # M6: (A12-A22)(B11+B21) -> C22
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),           # M7: (A12-A22)(B21+B22) -> C11
    ]
    
    # Note: The standard Strassen formulas are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M2 - M7
    # C12 = M3 + M5
    # C21 = M6 + M2
    # C22 = M1 - M3 + M5 - M6
    
    # My previous strassen list was slightly off in mapping. Let's use the explicit correct one.
    
    strassen_correct = [
        # M1: (A11+A22)(B11+B22)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, -1), (1, 1), (2, 1)]), 
        # M2: (A21+A22)B11
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3: A11(B12-B22)
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, -1)]),
        # M4: A22(B21-B11)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5: (A11+A12)B22
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (1, 1)]),
        # M6: (A21-A11)(B11+B12)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(2, 1), (3, -1)]),
        # M7: (A12-A22)(B21+B22)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, -1)]),
    ]
    
    # Re-verify M1 contribution to C:
    # C11 += M1, C22 += M1
    # C12 -= M1? No.
    # C11 = M1 + M4 - M2 - M7
    # C12 = M3 + M5
    # C21 = M6 + M2
    # C22 = M1 - M3 + M5 - M6
    
    # So M1 contributes to C11 (+1) and C22 (+1).
    # M2 contributes to C21 (+1) and C22 (-1).
    # M3 contributes to C12 (+1) and C22 (-1).
    # M4 contributes to C11 (+1) and C21 (+1).
    # M5 contributes to C12 (+1) and C22 (+1).
    # M6 contributes to C21 (+1) and C22 (-1).
    # M7 contributes to C11 (-1).
    
    # Let's re-map the W coefficients based on this.
    
    strassen_final = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, -1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(2, 1), (3, -1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, -1)]),
    ]
    
    # Blocks: (a_rows, a_cols, b_rows, b_cols)
    # A is 2x4, B is 4x6.
    # Split A cols into 2+2, B rows into 2+2.
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    
    blocks = [
        ([0, 1], [0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [0, 1], [2, 3]),
        ([0, 1], [0, 1], [0, 1], [4, 5]),
        ([0, 1], [2, 3], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3], [2, 3]),
        ([0, 1], [2, 3], [2, 3], [4, 5]),
    ]
    
    rank = len(blocks) * len(strassen_final)  # 6 * 7 = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for a_rows, a_cols, b_rows, b_cols in blocks:
        if time.time() > deadline:
            break
        
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1]
        }
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1]
        }
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_final:
            if time.time() > deadline:
                break
            for local_idx, coeff in u_coeffs:
                U[a_map[local_idx], col] += coeff
            for local_idx, coeff in v_coeffs:
                V[b_map[local_idx], col] += coeff
            for local_idx, coeff in w_coeffs:
                W[w_map[local_idx], col] += coeff
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
