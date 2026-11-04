"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 8 groups * 7 Strassen products = 56
    rank = 56
    
    # Initialize U, V, W
    # U: (n*m, rank) = (8, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (16, 56)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen definitions for 2x2 matrices
    # X entries: 00, 01, 10, 11
    # Y entries: 00, 01, 10, 11
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y01 - Y11)
    # M4 = X11(Y10 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 - X11)(Y10 + Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For each group g (0..7)
    col = 0
    for g in range(8):
        b_j = g % 2
        b_k = g // 2
        
        # Indices for X (from A)
        # X00: A[0, 2*b_j]
        # X01: A[0, 2*b_j+1]
        # X10: A[1, 2*b_j]
        # X11: A[1, 2*b_j+1]
        
        # Indices for Y (from B)
        # Y00: B[2*b_j, 2*b_k]
        # Y01: B[2*b_j, 2*b_k+1]
        # Y10: B[2*b_j+1, 2*b_k]
        # Y11: B[2*b_j+1, 2*b_k+1]
        
        # Indices for Z (to C)
        # Z00: C[0, 2*b_k] -> W[2*b_k * 2 + 0]
        # Z01: C[0, 2*b_k+1] -> W[(2*b_k+1) * 2 + 0]
        # Z10: C[1, 2*b_k] -> W[2*b_k * 2 + 1]
        # Z11: C[1, 2*b_k+1] -> W[(2*b_k+1) * 2 + 1]
        
        # Helper to get U index for A[i, j]
        def u_idx(i, j):
            return i * m + j
            
        # Helper to get V index for B[j, k]
        def v_idx(j, k):
            return j * p + k
            
        # Helper to get W index for C[i, k]
        def w_idx(i, k):
            return k * n + i
            
        # Define the 7 Strassen products
        # Each product is a tuple (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: list of (idx, coeff)
        # V_coeffs: list of (idx, coeff)
        # W_coeffs: list of (idx, coeff)
        
        products = []
        
        # M1 = (X00 + X11)(Y00 + Y11)
        # U: X00, X11
        # V: Y00, Y11
        # W: Z00 (+1), Z11 (+1)
        products.append((
            [(u_idx(0, 2*b_j), 1), (u_idx(1, 2*b_j+1), 1)],
            [(v_idx(2*b_j, 2*b_k), 1), (v_idx(2*b_j+1, 2*b_k+1), 1)],
            [(w_idx(0, 2*b_k), 1), (w_idx(1, 2*b_k+1), 1)]
        ))
        
        # M2 = (X10 + X11)Y00
        # U: X10, X11
        # V: Y00
        # W: Z10 (+1), Z11 (-1)
        products.append((
            [(u_idx(1, 2*b_j), 1), (u_idx(1, 2*b_j+1), 1)],
            [(v_idx(2*b_j, 2*b_k), 1)],
            [(w_idx(1, 2*b_k), 1), (w_idx(1, 2*b_k+1), -1)]
        ))
        
        # M3 = X00(Y01 - Y11)
        # U: X00
        # V: Y01, Y11
        # W: Z01 (+1), Z11 (+1)
        products.append((
            [(u_idx(0, 2*b_j), 1)],
            [(v_idx(2*b_j, 2*b_k+1), 1), (v_idx(2*b_j+1, 2*b_k+1), -1)],
            [(w_idx(0, 2*b_k+1), 1), (w_idx(1, 2*b_k+1), 1)]
        ))
        
        # M4 = X11(Y10 - Y00)
        # U: X11
        # V: Y10, Y00
        # W: Z00 (+1), Z10 (+1)
        products.append((
            [(u_idx(1, 2*b_j+1), 1)],
            [(v_idx(2*b_j+1, 2*b_k), 1), (v_idx(2*b_j, 2*b_k), -1)],
            [(w_idx(0, 2*b_k), 1), (w_idx(1, 2*b_k), 1)]
        ))
        
        # M5 = (X00 + X01)Y11
        # U: X00, X01
        # V: Y11
        # W: Z00 (-1), Z01 (+1)
        products.append((
            [(u_idx(0, 2*b_j), 1), (u_idx(0, 2*b_j+1), 1)],
            [(v_idx(2*b_j+1, 2*b_k+1), 1)],
            [(w_idx(0, 2*b_k), -1), (w_idx(0, 2*b_k+1), 1)]
        ))
        
        # M6 = (X10 - X00)(Y00 + Y01)
        # U: X10, X00
        # V: Y00, Y01
        # W: Z11 (+1)
        products.append((
            [(u_idx(1, 2*b_j), 1), (u_idx(0, 2*b_j), -1)],
            [(v_idx(2*b_j, 2*b_k), 1), (v_idx(2*b_j, 2*b_k+1), 1)],
            [(w_idx(1, 2*b_k+1), 1)]
        ))
        
        # M7 = (X01 - X11)(Y10 + Y11)
        # U: X01, X11
        # V: Y10, Y11
        # W: Z00 (+1)
        products.append((
            [(u_idx(0, 2*b_j+1), 1), (u_idx(1, 2*b_j+1), -1)],
            [(v_idx(2*b_j+1, 2*b_k), 1), (v_idx(2*b_j+1, 2*b_k+1), 1)],
            [(w_idx(0, 2*b_k), 1)]
        ))
        
        for s in range(7):
            u_coeffs, v_coeffs, w_coeffs = products[s]
            
            for idx, coeff in u_coeffs:
                U[idx][col] = coeff
            for idx, coeff in v_coeffs:
                V[idx][col] = coeff
            for idx, coeff in w_coeffs:
                W[idx][col] = coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
