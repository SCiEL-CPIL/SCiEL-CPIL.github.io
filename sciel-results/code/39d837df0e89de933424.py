"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 blocks * 7 Strassen terms = 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions for a 2x2 block
    # Each term is a tuple: (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
    # Indices are local to the 2x2 block for U, and relative to the block start for V/W
    
    # U indices: 00=0, 01=1, 10=2, 11=3
    # V indices (local): 00=0, 01=1, 10=2, 11=3 (mapped to global j*p+k later)
    # W indices (local): 00=0, 01=1, 10=2, 11=3 (mapped to global k*n+i later)
    
    # P1: (A00+A11)(B00+B11)
    # U: 0:+1, 3:+1
    # V: 0:+1, 3:+1
    # W: 0:+1, 3:+1
    terms = [
        # (U_idx, U_val, V_idx, V_val, W_idx, W_val)
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]), # P1
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),       # P2: (A10+A11)B00 -> C10:+1, C11:-1
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),       # P3: A00(B01-B11) -> C01:+1, C11:+1
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),       # P4: A11(B10-B00) -> C00:+1, C10:+1
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),       # P5: (A00+A01)B11 -> C00:-1, C01:+1
        ([0, 2], [-1, 1], [0, 1], [1, 1], [3], [1]),       # P6: (A10-A00)(B00+B01) -> C11:+1
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),       # P7: (A01-A11)(B10+B11) -> C00:+1
    ]
    
    num_blocks = p // 2 # 4 blocks
    
    for l in range(num_blocks):
        # Global column offset for V (B matrix)
        # B indices are j*p + k. 
        # For block l, k is 2l and 2l+1.
        # V local index 0 -> B[0, 2l] -> global 0*p + 2l = 2l
        # V local index 1 -> B[0, 2l+1] -> global 0*p + 2l+1 = 2l+1
        # V local index 2 -> B[1, 2l] -> global 1*p + 2l = 8+2l
        # V local index 3 -> B[1, 2l+1] -> global 1*p + 2l+1 = 9+2l
        v_offsets = [2*l, 2*l+1, 8+2*l, 9+2*l]
        
        # Global row offset for W (C matrix)
        # C indices are k*n + i.
        # For block l, k is 2l and 2l+1.
        # W local index 0 -> C[0, 2l] -> global 2l*n + 0 = 4l
        # W local index 1 -> C[0, 2l+1] -> global (2l+1)*n + 0 = 4l+2
        # W local index 2 -> C[1, 2l] -> global 2l*n + 1 = 4l+1
        # W local index 3 -> C[1, 2l+1] -> global (2l+1)*n + 1 = 4l+3
        w_offsets = [4*l, 4*l+2, 4*l+1, 4*l+3]
        
        for t_idx, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
            col = l * 7 + t_idx
            
            for idx, val in zip(u_idx, u_val):
                U[idx][col] += val
                
            for idx, val in zip(v_idx, v_val):
                V[v_offsets[idx]][col] += val
                
            for idx, val in zip(w_idx, w_val):
                W[w_offsets[idx]][col] += val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
