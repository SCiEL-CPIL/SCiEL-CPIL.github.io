import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block definitions
    # A blocks: 1x2
    # B blocks: 2x2
    # C blocks: 1x2
    
    # Strassen terms
    # Each term has a U pattern (2x4) and V pattern (4x4)
    # and contributes to C blocks with signs.
    
    # M1: (A00+A11)(B00+B11)
    # U1: A00(1,1), A11(1,1)
    # V1: B00(1s), B11(1s)
    # C: C00(+1), C11(+1)
    
    # M2: (A10+A11)B00
    # U2: A10(1,1), A11(1,1)
    # V2: B00(1s)
    # C: C10(+1), C11(-1)
    
    # M3: A00(B01-B11)
    # U3: A00(1,1)
    # V3: B01(1s), B11(-1s)
    # C: C01(+1), C11(+1)
    
    # M4: A11(B10-B00)
    # U4: A11(1,1)
    # V4: B10(1s), B00(-1s)
    # C: C00(+1), C10(+1)
    
    # M5: (A00+A01)B11
    # U5: A00(1,1), A01(1,1)
    # V5: B11(1s)
    # C: C00(-1), C01(+1)
    
    # M6: (A10-A00)(B00+B01)
    # U6: A10(1,1), A00(-1,-1)
    # V6: B00(1s), B01(1s)
    # C: C11(+1)
    
    # M7: (A01-A11)(B10+B11)
    # U7: A01(1,1), A11(-1,-1)
    # V7: B10(1s), B11(1s)
    # C: C00(+1)
    
    # For each term, we decompose the 1x2 x 2x2 product into 4 rank-1 tensors.
    # The product res_k = sum_j u_j v_jk.
    # The 4 rank-1 tensors are for (j=0, k=0), (j=0, k=1), (j=1, k=0), (j=1, k=1).
    
    terms = [
        # (U_coeffs, V_coeffs, C_contribs)
        # U_coeffs: list of (row, col, coeff) for the 1x2 block parts
        # V_coeffs: list of (row, col, coeff) for the 2x2 block parts
        # C_contribs: list of (c_row, c_col, sign)
        
        # M1
        ([ (0,0,1), (0,1,1), (1,2,1), (1,3,1) ],
         [ (0,0,1), (0,1,1), (1,0,1), (1,1,1), (2,2,1), (2,3,1), (3,2,1), (3,3,1) ],
         [ (0,0,1), (1,2,1) ]), # C00 is row 0 col 0-1, C11 is row 1 col 2-3
        
        # M2
        ([ (1,0,1), (1,1,1), (1,2,1), (1,3,1) ],
         [ (0,0,1), (0,1,1), (1,0,1), (1,1,1) ],
         [ (1,0,1), (1,2,-1) ]),
        
        # M3
        ([ (0,0,1), (0,1,1) ],
         [ (0,2,1), (0,3,1), (1,2,1), (1,3,1), (2,2,-1), (2,3,-1), (3,2,-1), (3,3,-1) ],
         [ (0,2,1), (1,2,1) ]),
        
        # M4
        ([ (1,2,1), (1,3,1) ],
         [ (2,0,1), (2,1,1), (3,0,1), (3,1,1), (0,0,-1), (0,1,-1), (1,0,-1), (1,1,-1) ],
         [ (0,0,1), (1,0,1) ]),
        
        # M5
        ([ (0,0,1), (0,1,1), (0,2,1), (0,3,1) ],
         [ (2,2,1), (2,3,1), (3,2,1), (3,3,1) ],
         [ (0,0,-1), (0,2,1) ]),
        
        # M6
        ([ (1,0,1), (1,1,1), (0,0,-1), (0,1,-1) ],
         [ (0,0,1), (0,1,1), (1,0,1), (1,1,1), (0,2,1), (0,3,1), (1,2,1), (1,3,1) ],
         [ (1,2,1) ]),
        
        # M7
        ([ (0,2,1), (0,3,1), (1,2,-1), (1,3,-1) ],
         [ (2,0,1), (2,1,1), (3,0,1), (3,1,1), (2,2,1), (2,3,1), (3,2,1), (3,3,1) ],
         [ (0,0,1) ])
    ]
    
    col = 0
    for u_coeffs, v_coeffs, c_contribs in terms:
        # Decompose into 4 rank-1 tensors
        # j in 0,1 (internal index), k in 0,1 (output col of block)
        for j in range(2):
            for k in range(2):
                # U part: select u_j
                # The u_coeffs are for the 1x2 blocks.
                # We need to map (block_row, block_col, coeff) to global (row, col).
                # And pick the component j of the 1x2 vector.
                # The 1x2 vector has indices 0,1.
                # So if block_col is 0, it's j=0. If block_col is 1, it's j=1.
                # Wait, the blocks are 1x2.
                # A00 is cols 0,1. A01 is cols 2,3.
                # So for a block at col_offset, the local j is 0 or 1.
                
                u_part = []
                for r, c, coeff in u_coeffs:
                    # c is the global column index? No, in my definitions above I used global.
                    # Let's assume the tuples in u_coeffs are (row, global_col, coeff).
                    # And we need to check if global_col corresponds to local j.
                    # This is messy.
                    
                    pass
