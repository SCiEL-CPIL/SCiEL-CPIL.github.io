import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    col = 0
    
    # Strassen 2x2x2 definitions
    # Each M_k: (list of (row, col, coeff) for A, list of (row, col, coeff) for B, list of (row, col, coeff) for C)
    # For 2x2x2 Strassen:
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C01+=0, C10+=0, C11+=M1
    # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
    # M3 = A00(B01-B11) -> C01+=M3, C11+=M3
    # M4 = A11(B10-B00) -> C00+=M4, C10+=M4
    # M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
    # M6 = (A10-A00)(B00+B01) -> C11+=M6
    # M7 = (A01-A11)(B10+B11) -> C00+=M7
    
    strassen_2x2 = [
        # (A_entries, B_entries, C_entries)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # Outer Strassen for 4x4x4:
    # View 4x4 as 2x2 blocks of 2x2.
    # A blocks: A00 at (0,0), A01 at (0,2), A10 at (2,0), A11 at (2,2)
    # B blocks: B00 at (0,0), B01 at (0,2), B10 at (2,0), B11 at (2,2)
    # C blocks: C00 at (0,0), C01 at (0,2), C10 at (2,0), C11 at (2,2)
    
    # Outer Strassen combinations (same as inner, but on blocks):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each outer M_k is a 2x2x2 multiplication of 2x2 blocks.
    # Done with inner Strassen (7 rank-1 terms per outer M_k).
    
    # For outer M_k, the X and Y are 2x2 block matrices.
    # X = sum of A-blocks with coefficients, Y = sum of B-blocks with coefficients.
    # Inner Strassen on X*Y:
    #   For each inner M_j, the A-entries are from X and B-entries are from Y.
    
    # Let me define outer combinations:
    # outer[k] = (list of (block_row, block_col, coeff) for A, list of (block_row, block_col, coeff) for B)
    outer_defs = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),  # M1
        ([(1,0,1),(1,1,1)], [(0,0,1)]),           # M2
        ([(0,0,1)], [(0,1,1),(1,1,-1)]),          # M3
        ([(1,1,1)], [(1,0,1),(0,0,-1)]),          # M4
        ([(0,0,1),(0,1,1)], [(1,1,1)]),           # M5
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),  # M6
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),  # M7
    ]
    
    # C block contributions (same as Strassen):
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    c_contribs = [
        [(0,1),(3,1),(4,-1),(6,1)],   # C00
        [(2,1),(4,1)],                 # C01
        [(1,1),(3,1)],                 # C10
        [(0,1),(1,-1),(2,1),(5,1)],   # C11
    ]
    
    # For each outer M_k, do inner Strassen.
    # The inner Strassen operates on the 2x2 block matrices X and Y.
    # X is a 2x2 matrix where X[i][j] is a 2x2 block (linear combo of A-blocks).
    # Y is a 2x2 matrix where Y[i][j] is a 2x2 block (linear combo of B-blocks).
    
    # For inner Strassen M_j:
    #   X_inner entries: combinations of X[i][j] for various i,j
    #   Y_inner entries: combinations of Y[i][j] for various i,j
    #   C_inner entries: combinations of C[i][j] for various i,j
    
    # X[i][j] = sum over outer A-terms where block position is (i,j)
    # Y[i][j] = sum over outer B-terms where block position is (i,j)
    
    # For inner Strassen term j:
    #   A_entries: for each (ri, ci, coeff) in inner A-def, the entry is X[ri][ci]
    #     = sum of (block_row, block_col, coeff_outer) in outer A-defs where block_row=ri, block_col=ci
    #     Each such A-block at (br, bc) has 2x2 entries at global positions (br*2+r, bc*2+c) for r,c in 0,1
    #   Similarly for B and C.
    
    # This is getting very nested. Let me just build it step by step.
    
    for k in range(7):
        outer_a, outer_b = outer_defs[k]
        # outer_a: list of (block_r, block_c, coeff) for A
        # outer_b: list of (block_r, block_c, coeff) for B
        
        # Build X (2x2 block matrix) and Y (2x2 block matrix)
        # X[i][j] = sum of coeff for (br,bc,coeff) in outer_a where br=i, bc=j
        # Y[i][j] = sum of coeff for (br,bc,coeff) in outer_b where br=i, bc=j
        
        # For inner Strassen, each term j has:
        #   A part: for each (ri, ci, coeff_inner) in strassen_2x2[j][0]:
        #     X[ri][ci] = sum of outer A-terms at block (ri, ci)
        #     Each outer A-term at block (br, bc) with coeff co gives A entries at (br*2+r, bc*2+c) for r,c in 0,1
        #   B part: similar
        #   C part: for each (ri, ci, coeff_inner) in strassen_2x2[j][2]:
        #     C[ri][ci] contributes to outer C blocks based on c_contribs
        
        for j in range(7):
            inner_a_def, inner_b_def, inner_c_def = strassen_2x2[j]
            
            U_entries = []
            V_entries = []
            W_entries = []
            
            # A part
            for (ri, ci, coeff_inner) in inner_a_def:
                # X[ri][ci] = sum of outer A-terms at block (ri, ci)
                for (br, bc, coeff_outer) in outer_a:
                    if br == ri and bc == ci:
                        coeff = coeff_inner * coeff_outer
                        # A-block at (br, bc) has entries at (br*2+r, bc*2+c)
                        for r in range(2):
                            for c in range(2):
                                U_entries.append(((br*2+r)*4 + (bc*2+c), coeff))
            
            # B part
            for (ri, ci, coeff_inner) in inner_b_def:
                # Y[ri][ci] = sum of outer B-terms at block (ri, ci)
                for (br, bc, coeff_outer) in outer_b:
                    if br == ri and bc == ci:
                        coeff = coeff_inner * coeff_outer
                        # B-block at (br, bc) has entries at (br*2+r, bc*2+c)
                        for r in range(2):
                            for c in range(2):
                                V_entries.append(((br*2+r)*5 + (bc*2+c), coeff))
            
            # C part
            for (ri, ci, coeff_inner) in inner_c_def:
                # C[ri][ci] within the 2x2 block result
                # This contributes to outer C blocks
                for (ci_outer, cj_outer) in range(4):
                    for (mk, sign) in c_contribs[ci_outer]:
                        if mk == k:
                            # C block ci_outer at position (ri, ci) within the block
                            # Global C position: (ci_outer*2+ri, cj_outer*2+ci)
                            # W index: col*4 + row = (cj_outer*2+ci)*4 + (ci_outer*2+ri)
                            coeff = coeff_inner * sign
                            W_entries.append(((cj_outer*2+ci)*4 + (ci_outer*2+ri), coeff))
            
            for (idx, coeff) in U_entries:
                U[idx][col] += coeff
            for (idx, coeff) in V_entries:
                V[idx][col] += coeff
            for (idx, coeff) in W_entries:
                W[idx][col] += coeff
            col += 1
    
    # Part 2: 4x4x1 (rank 16)
    for i in range(4):
        for j in range(4):
            U[i*4 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
