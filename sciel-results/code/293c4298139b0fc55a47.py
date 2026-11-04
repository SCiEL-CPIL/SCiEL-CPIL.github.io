"""Matrix multiplication 2x4x4 with rank-28 decomposition using Strassen."""
import time

DIMENSIONS = [2, 4, 4]


def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Uses Strassen's algorithm on four 2x2 * 2x2 sub-blocks.
    Total rank: 4 * 7 = 28.
    
    Interface:
      U shape: (n*m, rank) = (8, 28)
      V shape: (m*p, rank) = (16, 28)
      W shape: (p*n, rank) = (8, 28)
    """
    t0 = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    
    col = 0
    
    # Strassen formulas for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C = [[M1-M4+M5+M7, M3+M5], [M2+M4, M1-M3+M6]]
    
    # We need to map local A indices (0,1) and B indices (0,1) to global indices.
    # Local A: [[a, b], [c, d]] corresponds to A[i, j]
    # Local B: [[e, f], [g, h]] corresponds to B[j, k]
    # Local C: [[c00, c01], [c10, c11]] corresponds to C[i, k]
    
    # Strassen steps definition:
    # Each step is (coeffs for A, coeffs for B, coeffs for C)
    # coeffs are dicts or lists of (local_index, value)
    
    # M1: (a+d)(e+h)
    # A terms: (0,0):1, (1,1):1
    # B terms: (0,0):1, (0,1):1, (1,0):1, (1,1):1  -- Wait, (e+h) is B[0,0]+B[1,1]? No.
    # B = [[e, f], [g, h]]. e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1].
    # e+h = B[0,0] + B[1,1].
    # So B terms: (0,0):1, (1,1):1.
    # C terms: C[0,0]:1, C[1,1]:1.
    
    # Let's define the 7 steps explicitly.
    # Step 1: M1
    # A: (0,0):1, (1,1):1
    # B: (0,0):1, (1,1):1
    # C: (0,0):1, (1,1):1
    
    # Step 2: M2 = (c+d)e
    # A: (1,0):1, (1,1):1
    # B: (0,0):1
    # C: (1,0):1
    
    # Step 3: M3 = a(f-h)
    # A: (0,0):1
    # B: (0,1):1, (1,1):-1
    # C: (0,1):1
    
    # Step 4: M4 = d(g-e)
    # A: (1,1):1
    # B: (1,0):1, (0,0):-1
    # C: (0,0):-1, (1,0):1
    
    # Step 5: M5 = (a+b)h
    # A: (0,0):1, (0,1):1
    # B: (1,1):1
    # C: (0,0):1, (0,1):1
    
    # Step 6: M6 = (c-a)(e+f)
    # A: (1,0):1, (0,0):-1
    # B: (0,0):1, (0,1):1
    # C: (1,1):1
    
    # Step 7: M7 = (b-d)(g+h)
    # A: (0,1):1, (1,1):-1
    # B: (1,0):1, (1,1):1
    # C: (0,0):1, (0,1):-1
    
    steps = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: list of (local_i, local_j, val)
        # B_coeffs: list of (local_j, local_k, val)
        # C_coeffs: list of (local_i, local_k, val)
        [
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,-1), (1,0,1)]),
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,1), (0,1,1)]),
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (0,1,-1)]),
        ]
    ]
    
    # Define the 4 blocks
    # Block 0: A cols 0-1, B rows 0-1, B cols 0-1
    # Block 1: A cols 0-1, B rows 0-1, B cols 2-3
    # Block 2: A cols 2-3, B rows 2-3, B cols 0-1
    # Block 3: A cols 2-3, B rows 2-3, B cols 2-3
    
    blocks = [
        {'a_col_offset': 0, 'b_row_offset': 0, 'b_col_offset': 0},
        {'a_col_offset': 0, 'b_row_offset': 0, 'b_col_offset': 2},
        {'a_col_offset': 2, 'b_row_offset': 2, 'b_col_offset': 0},
        {'a_col_offset': 2, 'b_row_offset': 2, 'b_col_offset': 2},
    ]
    
    for block in blocks:
        a_off = block['a_col_offset']
        b_row_off = block['b_row_offset']
        b_col_off = block['b_col_offset']
        
        for a_coeffs, b_coeffs, c_coeffs in steps[0]:
            if time.time() - t0 > deadline - 5:
                return {"U": U, "V": W, "W": W} # Placeholder, need correct return
            
            # Process A
            for li, lj, val in a_coeffs:
                gi = li # i is 0 or 1
                gj = lj + a_off # j is 0..3
                u_idx = gi * 4 + gj
                U[u_idx][col] += val
                
            # Process B
            for lj, lk, val in b_coeffs:
                gj = lj + b_row_off # j is 0..3
                gk = lk + b_col_off # k is 0..3
                v_idx = gj * 4 + gk
                V[v_idx][col] += val
                
            # Process C
            for li, lk, val in c_coeffs:
                gi = li # i is 0 or 1
                gk = lk + b_col_off # k is 0..3
                # W index is k*n + i = gk*2 + gi
                w_idx = gk * 2 + gi
                W[w_idx][col] += val
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
