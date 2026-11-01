"""Rewritten program using Strassen-like decomposition for 2x2 * 2x10 matrix multiplication."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # Rank is 5 blocks of 2x2 multiplication, each rank 7 (Strassen)
    rank = 35
    
    # U: (n*m, rank) = (4, 35)
    # V: (m*p, rank) = (20, 35)
    # W: (p*n, rank) = (20, 35)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps for 2x2 multiplication
    # u is vector of length 4 (A00, A01, A10, A11)
    # v_coeffs: list of (index_in_B, coeff)
    # w_coeffs: list of (index_in_C, coeff)
    
    # Indices for A (flattened row-major 2x2):
    # A00: 0, A01: 1, A10: 2, A11: 3
    
    # For a block b (0 to 4), columns are 2b and 2b+1.
    # B indices (flattened 2x10 row-major):
    # B0, 2b: 0*10 + 2b = 2b
    # B0, 2b+1: 0*10 + 2b + 1 = 2b + 1
    # B1, 2b: 1*10 + 2b = 10 + 2b
    # B1, 2b+1: 1*10 + 2b + 1 = 10 + 2b + 1
    
    # C indices (flattened 2x10 row-major):
    # C0, 2b: 2b
    # C0, 2b+1: 2b + 1
    # C1, 2b: 10 + 2b
    # C1, 2b+1: 10 + 2b + 1
    
    steps = [
        # (u, v_indices, w_indices)
        # Step 1: M1 = (A00+A11)(B00+B11)
        ([1, 0, 0, 1], 
         [(0, 1), (1, 1)],  # B00, B11 relative to block
         [(0, 1), (1, -1)]), # C00, C11 relative to block
         
        # Step 2: M2 = (A10+A11)B00
        ([0, 0, 1, 1], 
         [(0, 1)], 
         [(2, 1), (3, -1)]), # C10, C11
        
        # Step 3: M3 = A00(B01-B11)
        ([1, 0, 0, 0], 
         [(1, 1), (3, -1)], # B01, B11
         [(1, 1), (3, 1)]), # C01, C11
        
        # Step 4: M4 = A11(B10-B00)
        ([0, 0, 0, 1], 
         [(2, 1), (0, -1)], # B10, B00
         [(0, 1), (2, 1)]), # C00, C10
        
        # Step 5: M5 = (A00+A01)B11
        ([1, 1, 0, 0], 
         [(3, 1)], 
         [(0, -1), (1, 1)]), # C00, C01
        
        # Step 6: M6 = (A10-A00)(B00+B01)
        ([-1, 0, 1, 0], 
         [(0, 1), (1, 1)], # B00, B01
         [(3, 1)]), # C11
        
        # Step 7: M7 = (A01-A11)(B10+B11)
        ([0, 1, 0, -1], 
         [(2, 1), (3, 1)], # B10, B11
         [(0, 1)]) # C00
    ]
    
    # Relative indices in 2x2 block:
    # B: 0->B00, 1->B01, 2->B10, 3->B11
    # C: 0->C00, 1->C01, 2->C10, 3->C11
    
    for b in range(5):
        base_col = 2 * b
        for s in range(7):
            r = b * 7 + s
            u, v_rel, w_rel = steps[s]
            
            # Fill U
            for i in range(4):
                U[i][r] = u[i]
                
            # Fill V
            # Map relative B index to global
            # rel 0: 2b
            # rel 1: 2b+1
            # rel 2: 10+2b
            # rel 3: 10+2b+1
            for rel_idx, coeff in v_rel:
                if rel_idx == 0:
                    global_idx = 2 * b
                elif rel_idx == 1:
                    global_idx = 2 * b + 1
                elif rel_idx == 2:
                    global_idx = 10 + 2 * b
                elif rel_idx == 3:
                    global_idx = 10 + 2 * b + 1
                V[global_idx][r] = coeff
                
            # Fill W
            # Map relative C index to global
            # rel 0: 2b
            # rel 1: 2b+1
            # rel 2: 10+2b
            # rel 3: 10+2b+1
            for rel_idx, coeff in w_rel:
                if rel_idx == 0:
                    global_idx = 2 * b
                elif rel_idx == 1:
                    global_idx = 2 * b + 1
                elif rel_idx == 2:
                    global_idx = 10 + 2 * b
                elif rel_idx == 3:
                    global_idx = 10 + 2 * b + 1
                W[global_idx][r] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
