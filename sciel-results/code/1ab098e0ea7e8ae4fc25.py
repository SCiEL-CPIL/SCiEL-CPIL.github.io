"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 14 (2 * Strassen's 7)
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 multiplications for 2x2
    # A indices: 0:00, 1:01, 2:10, 3:11 (i*m+j)
    # B local indices: 0:00, 1:01, 2:10, 3:11 (j*2+k) -> mapped to global
    # C local indices: 0:00, 1:01, 2:10, 3:11 (k*2+i) -> mapped to global
    
    # Strassen terms: (U_coeff, V_coeff, W_coeff)
    # U is 4-dim
    # V, W are 4-dim local
    
    strassen_terms = [
        # M1 = A00 * (B00 + B11)
        ([1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 1]), # C00 +=, C11 +=
        # M2 = (A00 + A10) * B11
        ([1, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 1]), # C00 -=, C11 +=
        # M3 = A11 * (B01 - B10)
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, -1]), # C00 +=, C11 -=
        # M4 = A10 * (B00 + B01)
        ([0, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0]), # C01 +=, C10 +=
        # M5 = (A01 + A11) * B10
        ([0, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]), # C00 +=
        # M6 = (A00 - A11) * (B01 + B11)
        ([1, 0, 0, -1], [0, 1, 0, 1], [0, 1, 0, 0]), # C01 +=
        # M7 = (A01 - A10) * (B00 - B10)
        ([0, 1, -1, 0], [1, 0, -1, 0], [0, 1, 0, 0]), # C10 +=
    ]
    
    # Fill Block 1 (k=0,1)
    for r in range(7):
        u_coeffs, v_coeffs, w_coeffs = strassen_terms[r]
        col = r
        
        # U
        for i in range(4):
            U[i][col] = u_coeffs[i]
            
        # V (global indices for k=0,1)
        # Local j,k -> global j*4+k
        # 0:00->0, 1:01->1, 2:10->4, 3:11->5
        v_map = [0, 1, 4, 5]
        for idx, val in enumerate(v_coeffs):
            if val != 0:
                V[v_map[idx]][col] = val
                
        # W (global indices for k=0,1)
        # Local k,i -> global k*2+i
        # 0:00->0, 1:01->1, 2:10->2, 3:11->3
        w_map = [0, 1, 2, 3]
        for idx, val in enumerate(w_coeffs):
            if val != 0:
                W[w_map[idx]][col] = val
                
    # Fill Block 2 (k=2,3)
    for r in range(7):
        u_coeffs, v_coeffs, w_coeffs = strassen_terms[r]
        col = 7 + r
        
        # U (same)
        for i in range(4):
            U[i][col] = u_coeffs[i]
            
        # V (global indices for k=2,3)
        # 0:00->2, 1:01->3, 2:10->6, 3:11->7
        v_map = [2, 3, 6, 7]
        for idx, val in enumerate(v_coeffs):
            if val != 0:
                V[v_map[idx]][col] = val
                
        # W (global indices for k=2,3)
        # 0:00->4, 1:01->5, 2:10->6, 3:11->7
        w_map = [4, 5, 6, 7]
        for idx, val in enumerate(w_coeffs):
            if val != 0:
                W[w_map[idx]][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
