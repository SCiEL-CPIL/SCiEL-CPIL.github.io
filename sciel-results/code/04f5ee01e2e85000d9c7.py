"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Strassen factors for 2x2x2 block
    # Format: list of 7 tuples (u_vec, v_vec, w_vec)
    # u_vec: length 4 (indices 0..3 for A)
    # v_vec: length 4 (indices 0..3 for B block)
    # w_vec: length 4 (indices 0..3 for C block)
    
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    def idx(i, j): return i * m + j
    
    strassen_factors = [
        # M1 = (a00+a11)(b00+b11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (a10+a11)b00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # C10+=1, C11-=-1? 
        # Wait, let's re-verify the Strassen factors for 2x2x2
        # C00 = M1 - M2 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M6 - M7
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Let's map these to vectors.
        # A vec: [A00, A01, A10, A11]
        # B vec: [B00, B01, B10, B11]
        # C vec: [C00, C01, C10, C11]
        
        # M1:
        # A: [1, 0, 0, 1]
        # B: [1, 0, 0, 1]
        # C: [1, 0, 0, 1] (Contributes to C00 and C11)
        
        # M2:
        # A: [0, 0, 1, 1]
        # B: [1, 0, 0, 0]
        # C: [0, 0, 1, -1] (Contributes to C10 and C11)
        
        # M3:
        # A: [1, 0, 0, 0]
        # B: [0, 1, 0, -1]
        # C: [0, 1, 0, 1] (Contributes to C01 and C11)
        
        # M4:
        # A: [0, 0, 0, 1]
        # B: [0, 0, 1, -1]
        # C: [1, 0, 1, 0] (Contributes to C00 and C10)
        
        # M5:
        # A: [1, 1, 0, 0]
        # B: [0, 0, 0, 1]
        # C: [-1, 1, 0, 0] (Contributes to C00 and C01)
        
        # M6:
        # A: [0, 0, 1, -1]
        # B: [1, 1, 0, 0]
        # C: [0, 0, 0, 1] (Contributes to C11)
        
        # M7:
        # A: [0, 1, 0, -1]
        # B: [0, 0, 1, 1]
        # C: [1, 0, 0, 0] (Contributes to C00)
    ]
    
    # Verify M6: (A10-A00)(B00+B01)
    # A: [0, 0, 1, -1] -> A10 - A00. Correct.
    # B: [1, 1, 0, 0] -> B00 + B01. Correct.
    # C: Should contribute to C11?
    # C11 = M1 - M3 + M6 - M7.
    # M6 contributes +1 to C11.
    # So C vec for M6 should be [0, 0, 0, 1]. Correct.
    
    # Verify M7: (A01-A11)(B10+B11)
    # A: [0, 1, 0, -1] -> A01 - A11. Correct.
    # B: [0, 0, 1, 1] -> B10 + B11. Correct.
    # C: Should contribute to C00?
    # C00 = M1 - M2 + M4 - M5 + M7.
    # M7 contributes +1 to C00.
    # So C vec for M7 should be [1, 0, 0, 0]. Correct.

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_vec, v_vec, w_vec = strassen_factors[s]
            
            # Fill U
            for i in range(4):
                U[i][col] = u_vec[i]
                
            # Fill V
            # V rows are j*p + k.
            # For block t, k is 2t or 2t+1.
            # v_vec indices 0..3 correspond to (j=0,k_local=0), (j=0,k_local=1), (j=1,k_local=0), (j=1,k_local=1)
            # So v_vec[0] -> j=0, k=2t
            # v_vec[1] -> j=0, k=2t+1
            # v_vec[2] -> j=1, k=2t
            # v_vec[3] -> j=1, k=2t+1
            
            V[0 * p + (2 * t)][col] = v_vec[0]
            V[0 * p + (2 * t + 1)][col] = v_vec[1]
            V[1 * p + (2 * t)][col] = v_vec[2]
            V[1 * p + (2 * t + 1)][col] = v_vec[3]
            
            # Fill W
            # W rows are k*n + i.
            # w_vec indices 0..3 correspond to (i=0,k_local=0), (i=0,k_local=1), (i=1,k_local=0), (i=1,k_local=1)
            # So w_vec[0] -> i=0, k=2t
            # w_vec[1] -> i=0, k=2t+1
            # w_vec[2] -> i=1, k=2t
            # w_vec[3] -> i=1, k=2t+1
            
            W[(2 * t) * n + 0][col] = w_vec[0]
            W[(2 * t + 1) * n + 0][col] = w_vec[1]
            W[(2 * t) * n + 1][col] = w_vec[2]
            W[(2 * t + 1) * n + 1][col] = w_vec[3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
