import numpy as np
import time

def construct():
    """
    Returns a rank-56 tensor decomposition for 2x4 by 4x8 matrix multiplication.
    This exploits the block structure where the multiplication can be split into
    two independent 2x2 by 2x8 multiplications, each of which can be further
    split into four 2x2 by 2x2 multiplications.
    Rank = 2 * 4 * 7 = 56.
    """
    n, m, p = 2, 4, 8
    rank = 56
    
    # Initialize U, V, W
    # U shape: (n*m, rank) = (8, 56)
    # V shape: (m*p, rank) = (32, 56)
    # W shape: (p*n, rank) = (16, 56)
    
    # Using lists of lists as required
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen multipliers for 2x2 * 2x2
    # Indices in 2x2 block:
    # A: 0,1 (top row), 2,3 (bottom row) -> actually 0,1,2,3 flattened
    # Let's use local 0..3 for A (2x2)
    # B: 0..3 (2x2)
    # C: 0..3 (2x2)
    
    # Standard Strassen 7 multiplications:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(h)  -- wait, standard is a(f-h) or (a-b)h? 
    # Let's stick to the previous working one to ensure correctness, 
    # but re-encode it.
    
    # Previous successful pattern:
    # 1. (a+d)(e+h)
    # 2. (c+d)e
    # 3. a(f-h)
    # 4. d(g-e)
    # 5. (a+b)h
    # 6. (c-a)(e+f)
    # 7. (b-d)(g+h)
    
    # Let's map these to local indices 0,1,2,3 for 2x2 matrices.
    # A: [[0, 1], [2, 3]] -> a=0, b=1, c=2, d=3
    # B: [[0, 1], [2, 3]] -> e=0, f=1, g=2, h=3
    # C: [[0, 1], [2, 3]] -> out00=0, out01=1, out10=2, out11=3
    
    # U_local, V_local, W_local are lists of (index, coeff)
    
    strassen_steps = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # Each coeff is (local_idx, val)
        (
            [(0, 1), (3, 1)],  # a+d
            [(0, 1), (3, 1)],  # e+h
            [(0, 1), (3, 1)]   # c11 + c22
        ),
        (
            [(2, 1), (3, 1)],  # c+d
            [(0, 1)],          # e
            [(2, 1), (3, -1)]  # c21 - c22
        ),
        (
            [(0, 1)],          # a
            [(1, 1), (3, -1)], # f-h
            [(1, 1), (3, 1)]   # c12 + c22
        ),
        (
            [(3, 1)],          # d
            [(2, 1), (0, -1)], # g-e
            [(0, 1), (2, 1)]   # c11 + c21
        ),
        (
            [(0, 1), (1, 1)],  # a+b
            [(3, 1)],          # h
            [(0, -1), (1, 1)]  # -c11 + c12
        ),
        (
            [(2, 1), (0, -1)], # c-a
            [(0, 1), (1, 1)],  # e+f
            [(3, 1)]           # c22
        ),
        (
            [(1, 1), (3, -1)], # b-d
            [(2, 1), (3, 1)],  # g+h
            [(0, 1)]           # c11
        )
    ]
    
    col = 0
    # Block b: 0, 1 (columns of A, rows of B)
    # Block c: 0, 1, 2, 3 (columns of B)
    
    for b in range(2):
        for c in range(4):
            # Global indices for A (2x4)
            # A is 2 rows, 4 cols.
            # Block b covers cols 2b, 2b+1.
            # Local 0 -> A[0, 2b]
            # Local 1 -> A[0, 2b+1]
            # Local 2 -> A[1, 2b]
            # Local 3 -> A[1, 2b+1]
            
            a_idx = [0*m + 2*b, 0*m + 2*b + 1, 1*m + 2*b, 1*m + 2*b + 1]
            
            # Global indices for B (4x8)
            # B is 4 rows, 8 cols.
            # Block b covers rows 2b, 2b+1.
            # Block c covers cols 2c, 2c+1.
            # Local 0 -> B[2b, 2c]
            # Local 1 -> B[2b, 2c+1]
            # Local 2 -> B[2b+1, 2c]
            # Local 3 -> B[2b+1, 2c+1]
            
            b_idx = [2*b*p + 2*c, 2*b*p + 2*c + 1, (2*b+1)*p + 2*c, (2*b+1)*p + 2*c + 1]
            
            # Global indices for C (2x8)
            # C is 2 rows, 8 cols.
            # Block c covers cols 2c, 2c+1.
            # Local 0 -> C[0, 2c]
            # Local 1 -> C[0, 2c+1]
            # Local 2 -> C[1, 2c]
            # Local 3 -> C[1, 2c+1]
            # Note: W indexing in task is W[k*n+i, r] which is C[i, k] if k is row?
            # Task: "output factor follows the official trace convention W[k*n+i,r]"
            # Usually C[i, j] is index i*p + j? No, W is (p*n, rank).
            # p=8, n=2.
            # W index = k*n + i.
            # If C is 2x8, indices are 0..15.
            # C[i, j] corresponds to W[j*n + i]?
            # Let's check the previous code's W indexing.
            # Previous: idx_c11 = 2*c*n + 0.
            # n=2. So 4c.
            # C[0, 2c] is row 0, col 2c.
            # If W index is col*n + row, then C[0, 2c] -> 2c*2 + 0 = 4c.
            # This matches idx_c11 = 2*c*n + 0.
            # So W_idx = col * n + row.
            
            c_idx = [2*c*n + 0, 2*c*n + 1, (2*c+1)*n + 0, (2*c+1)*n + 1]
            # c_idx[0] = C[0, 2c]
            # c_idx[1] = C[0, 2c+1]
            # c_idx[2] = C[1, 2c]
            # c_idx[3] = C[1, 2c+1]
            
            for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
                for loc_idx, val in u_coeffs:
                    U[a_idx[loc_idx]][col] += val
                for loc_idx, val in v_coeffs:
                    V[b_idx[loc_idx]][col] += val
                for loc_idx, val in w_coeffs:
                    W[c_idx[loc_idx]][col] += val
                col += 1
                
    return {"U": U, "V": V, "W": W}
