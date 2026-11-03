import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 42 decomposition
    # Structure: 6 blocks of 2x2x2, each using Strassen (rank 7)
    # Total rank 42
    
    U = np.zeros((n * m, 42), dtype=float)
    V = np.zeros((m * p, 42), dtype=float)
    W = np.zeros((p * n, 42), dtype=float)
    
    # Strassen coefficients for 2x2x2
    # M1: (A11+A22)(B11+B22) -> Z11, Z22
    # M2: (A21+A22)B11 -> Z21, -Z22
    # M3: A11(B21-B22) -> Z11, Z12
    # M4: (A11-A21)(B11+B12) -> Z21, Z22
    # M5: (A12+A22)(B21+B22) -> -Z11, Z12
    # M6: (A11-A21)(B12-B22) -> Z21
    # M7: (A12-A22)(B11+B12) -> Z12
    
    # We define the blocks
    # A is 2x4. B is 4x6.
    # A_L: rows 0-1, cols 0-1. A_R: rows 0-1, cols 2-3.
    # B_T: rows 0-1. B_B: rows 2-3.
    # B_T is split into 3 2x2 blocks: cols 0-1, 2-3, 4-5.
    # B_B is split into 3 2x2 blocks: cols 0-1, 2-3, 4-5.
    
    blocks = [
        # (a_rows, a_cols, b_rows, b_cols)
        ([0, 1], [0, 1], [0, 1], [0, 1]), # A_L * B_T1
        ([0, 1], [0, 1], [0, 1], [2, 3]), # A_L * B_T2
        ([0, 1], [0, 1], [0, 1], [4, 5]), # A_L * B_T3
        ([0, 1], [2, 3], [2, 3], [0, 1]), # A_R * B_B1
        ([0, 1], [2, 3], [2, 3], [2, 3]), # A_R * B_B2
        ([0, 1], [2, 3], [2, 3], [4, 5]), # A_R * B_B3
    ]
    
    # Strassen terms: (u_coeffs, v_coeffs, w_coeffs)
    # Local indices:
    # A: 0:00, 1:01, 2:10, 3:11
    # B: 0:00, 1:01, 2:10, 3:11
    # Z: 0:00, 1:01, 2:10, 3:11
    
    strassen_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    col = 0
    for block in blocks:
        a_rows, a_cols, b_rows, b_cols = block
        
        # Mapping local indices to global
        # A global: i*m + j
        # B global: j*p + k
        # W global: k*n + i
        
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
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for local_idx, coeff in u_coeffs:
                g_idx = a_map[local_idx]
                U[g_idx, col] += coeff
            
            for local_idx, coeff in v_coeffs:
                g_idx = b_map[local_idx]
                V[g_idx, col] += coeff
            
            for local_idx, coeff in w_coeffs:
                g_idx = w_map[local_idx]
                W[g_idx, col] += coeff
            
            col += 1
            
            # Time check
            if time.time() - start_time > 100:
                break
        if time.time() - start_time > 100:
            break
            
    # Convert to lists of lists as required by interface?
    # The interface says: factor matrices of shapes ... with integer/half-integer entries
    # Usually numpy arrays are accepted, but let's check the seed.
    # Seed used lists of lists.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
