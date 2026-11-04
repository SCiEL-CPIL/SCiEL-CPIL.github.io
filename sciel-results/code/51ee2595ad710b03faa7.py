import time
import sys

# Entry point: returns dict with 'U', 'V', 'W'
def construct():
    """
    Rank-84 exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Partition A (4x4) into 2x2 grid of 2x2 blocks,
    B (4x6) into 2x3 grid of 2x2 blocks.
    For each of the 2*3*2 = 12 block pairs (A_ik, B_kj), apply rank-7 Strassen.
    Total rank = 12 * 7 = 84.
    
    Index conventions:
      U: (n*m) x rank = 4*4 = 16 rows, 84 cols
        Row index: i*m + j where i is A row (0..3), j is A col (0..3)
      V: (m*p) x rank = 4*6 = 24 rows, 84 cols
        Row index: j*p + k where j is B row (0..3), k is B col (0..5)
      W: (p*n) x rank = 6*4 = 24 rows, 84 cols
        Row index: k*n + i where k is C row (0..5), i is C col (0..3)
    
    Strassen for 2x2 * 2x2:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B01-B11)
      M4 = A11(B10-B00)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B01)
      M7 = (A01-A11)(B10+B11)
      
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120  # 120 second safety margin
    
    # Strassen coefficients for each of the 7 intermediate products
    # Each entry: (a_coeff, b_coeff, c_coeff) for the output element
    # For a given (a_local_idx, b_local_idx) -> contributes to c_local_idx
    # 
    # We define for each Strassen term r (0..6):
    #   U_contribution: which A elements are summed and with what coefficient
    #   V_contribution: which B elements are summed and with what coefficient
    #   W_contribution: which C elements receive the product and with what coefficient
    #
    # A local indices: (u, v) where u is row (0,1), v is col (0,1)
    # B local indices: (x, y) where x is row (0,1), y is col (0,1)
    # C local indices: (z, w) where z is row (0,1), w is col (0,1)
    #
    # For each Strassen term, we need:
    #   A_mask: dict from (u,v) to coefficient
    #   B_mask: dict from (x,y) to coefficient
    #   C_mask: dict from (z,w) to coefficient
    #
    # But since Strassen uses specific linear combinations, we decompose each term
    # into rank-1 contributions. Each Strassen term is itself a product of two
    # linear combinations, so each contributes rank 1 to the local tensor.
    #
    # For term r:
    #   A combination: sum over (u,v) of a_r[u,v] * A[u,v]
    #   B combination: sum over (x,y) of b_r[x,y] * B[x,y]
    #   This product contributes to C: sum over (z,w) of c_r[z,w] * C[z,w]
    
    strassen_A = [
        {(0,0): 1, (1,1): 1},          # M1: A00 + A11
        {(1,0): 1, (1,1): 1},          # M2: A10 + A11
        {(0,0): 1},                     # M3: A00
        {(1,1): 1},                     # M4: A11
        {(0,0): 1, (0,1): 1},          # M5: A00 + A01
        {(1,0): 1, (0,0): -1},         # M6: A10 - A00
        {(0,1): 1, (1,1): -1},         # M7: A01 - A11
    ]
    
    strassen_B = [
        {(0,0): 1, (1,1): 1},          # M1: B00 + B11
        {(0,0): 1},                     # M2: B00
        {(0,1): 1, (1,1): -1},         # M3: B01 - B11
        {(1,0): 1, (0,0): -1},         # M4: B10 - B00
        {(1,1): 1},                     # M5: B11
        {(0,0): 1, (0,1): 1},          # M6: B00 + B01
        {(1,0): 1, (1,1): 1},          # M7: B10 + B11
    ]
    
    # C contributions for each Strassen term
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    strassen_C = [
        {(0,0): 1, (1,1): 1},          # M1 -> C00: +1, C11: +1
        {(1,0): 1, (1,1): -1},         # M2 -> C10: +1, C11: -1
        {(0,1): 1, (1,1): 1},          # M3 -> C01: +1, C11: +1
        {(0,0): 1, (1,0): 1},          # M4 -> C00: +1, C10: +1
        {(0,0): -1, (0,1): 1},         # M5 -> C00: -1, C01: +1
        {(1,1): 1},                    # M6 -> C11: +1
        {(0,0): 1},                    # M7 -> C00: +1
    ]
    
    n, m, p = 4, 4, 6
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    # i_block: row block of A (0,1)
    # k_block: col block of A / row block of B (0,1) 
    # j_block: col block of B (0,1,2)
    for i_block in range(2):
        for k_block in range(2):
            for j_block in range(3):
                for r in range(7):
                    if time.time() > deadline - 5:
                        # Return what we have so far (should not happen for this small problem)
                        break
                    # A global row: i = i_block*2 + u, col in A: j_a = k_block*2 + v
                    # B global row: i_b = k_block*2 + x, col in B: j_b = j_block*2 + y
                    # C global row: i_c = j_block*2 + z, col in C: j_c = i_block*2 + w
                    
                    # Fill U
                    for (u, v), a_coeff in strassen_A[r].items():
                        u_global = i_block * 2 + u
                        v_global = k_block * 2 + v
                        U[u_global * m + v_global][col] = a_coeff
                    
                    # Fill V
                    for (x, y), b_coeff in strassen_B[r].items():
                        x_global = k_block * 2 + x
                        y_global = j_block * 2 + y
                        V[x_global * p + y_global][col] = b_coeff
                    
                    # Fill W
                    for (z, w), c_coeff in strassen_C[r].items():
                        z_global = j_block * 2 + z
                        w_global = i_block * 2 + w
                        W[z_global * n + w_global][col] = c_coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
