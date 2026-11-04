"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition: rank 42 for 2x4 * 4x6
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # 8 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 12 x 42
    
    # Split A (2x4) into A1 = A[:,0:2], A2 = A[:,2:4]  (each 2x2)
    # Split B (4x6) into B blocks: for each output column pair, 
    #   B_{b,1} = B[0:2, 2b:2b+2], B_{b,2} = B[2:4, 2b:2b+2]  (each 2x2)
    # Output C (2x6): C_b = C[:, 2b:2b+2] for b=0,1,2
    
    # Strassen multipliers for 2x2 * 2x2:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y11)
    # M4 = X11(Y01-Y10)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y10)
    # M7 = (X11-X01)(Y01+Y11)
    # Results:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Each Strassen term r for block b contributes:
    # U part: coefficients on A entries (A1 or A2)
    # V part: coefficients on B entries (B_{b,1} or B_{b,2})
    # W part: coefficient on C entry
    
    # Indexing:
    # U rows: i*m + j where i=0..1 (row of A), j=0..3 (col of A)
    #   A1: j=0,1 -> U[0*4+0]=A00, U[0*4+1]=A01, U[1*4+0]=A10, U[1*4+1]=A11
    #   A2: j=2,3 -> U[0*4+2]=A02, U[0*4+3]=A03, U[1*4+2]=A12, U[1*4+3]=A13
    # V rows: j*p + k where j=0..3 (row of B), k=0..5 (col of B)
    #   B_{b,1}: rows 0,1 of B, cols 2b, 2b+1 -> V[0*6+2b], V[1*6+2b], V[0*6+2b+1], V[1*6+2b+1]
    #   B_{b,2}: rows 2,3 of B, cols 2b, 2b+1 -> V[2*6+2b], V[3*6+2b], V[2*6+2b+1], V[3*6+2b+1]
    # W rows: k*n + i where k=0..5 (col of C), i=0..1 (row of C)
    #   C_b: cols 2b, 2b+1 of C -> W[2b*2+0]=C0, W[2b*2+1]=C1 for row 0; W[2b*2+2]=C0, W[2b*2+3]=C1 for row 1
    #   Actually W[k*n+i]: k is output col index, i is output row index
    #   For block b, output cols are 2b and 2b+1
    #   C[0, 2b] = W[2b*2 + 0], C[1, 2b] = W[2b*2 + 1]
    #   C[0, 2b+1] = W[(2b+1)*2 + 0], C[1, 2b+1] = W[(2b+1)*2 + 1]
    
    # Strassen term definitions: (name, U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict from (i,j) to coeff, applied to A1 or A2
    # V_coeffs: dict from (j,k) to coeff, applied to B_{b,1} or B_{b,2}
    # W_coeffs: dict from (i_out, col_offset) to coeff, col_offset is 0 or 1 within block
    
    def strassen_terms():
        """Return list of (u_part, v_part, w_part) for each of 7 Strassen multipliers.
        u_part: dict {(i,j): coeff} for the 2x2 matrix X (A1 or A2)
        v_part: dict {(j,k): coeff} for the 2x2 matrix Y (B_{b,1} or B_{b,2})
        w_part: dict {(i_out, c_off): coeff} for output Z
        """
        terms = []
        
        # M1 = (X00+X11)(Y00+Y11)
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}  # Z00 += M1, Z11 += M1
        ))
        
        # M2 = (X10+X11)Y00
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}  # Z10 += M2, Z11 -= M2
        ))
        
        # M3 = X00(Y10-Y11)
        terms.append((
            {(0,0): 1},
            {(1,0): 1, (1,1): -1},
            {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}  # Z01 += M3, Z11 += M3
        ))
        
        # M4 = X11(Y01-Y10)
        terms.append((
            {(1,1): 1},
            {(0,1): 1, (1,0): -1},
            {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}  # Z00 += M4, Z10 += M4
        ))
        
        # M5 = (X00+X01)Y11
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}  # Z00 -= M5, Z01 += M5
        ))
        
        # M6 = (X10-X00)(Y00+Y10)
        terms.append((
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1},
            {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}  # Z11 += M6
        ))
        
        # M7 = (X11-X01)(Y01+Y11)
        terms.append((
            {(1,1): 1, (0,1): -1},
            {(0,1): 1, (1,1): 1},
            {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}  # Z00 += M7
        ))
        
        return terms
    
    terms = strassen_terms()
    
    col = 0
    for b in range(3):  # output column block
        for s in range(7):  # Strassen step
            u_part, v_part, w_part = terms[s]
            
            # A1: rows 0-1 of A, cols 0-1 -> U indices i*4+j
            # A2: rows 0-1 of A, cols 2-3 -> U indices i*4+j
            # For A1: local (i,j) -> global U[i*4 + j]
            # For A2: local (i,j) -> global U[i*4 + (j+2)]
            
            # Which A matrix does this term use?
            # Each Strassen term uses one A matrix (A1 or A2) and one B sub-block.
            # Wait - I need to think about this more carefully.
            # C_b = A1 * B_{b,1} + A2 * B_{b,2}
            # We can apply Strassen separately to A1*B_{b,1} and A2*B_{b,2}, 
            # but that would give 14 terms per block.
            # Alternatively, we can combine: the product A1*B_{b,1} + A2*B_{b,2}
            # is just a 2x4 * 4x2 = 2x2 multiplication with larger entries.
            # Actually, A1 is 2x2, B_{b,1} is 2x2, so A1*B_{b,1} is 2x2.
            # A2 is 2x2, B_{b,2} is 2x2, so A2*B_{b,2} is 2x2.
            # Their sum is 2x2.
            # But we can't directly apply Strassen to a sum of two 2x2*2x2 products 
            # to get fewer than 14 terms in general.
            # So we use 7 terms for A1*B_{b,1} and 7 terms for A2*B_{b,2}.
            
            pass
    
    # Let me redo: 7 terms per (A_block, B_subblock) pair, 3 B blocks, 2 A blocks = 42 terms
    
    col = 0
    for b in range(3):  # output column block
        for a_blk in range(2):  # A block: 0=A1, 1=A2
            for s in range(7):  # Strassen step
                u_part, v_part, w_part = terms[s]
                
                # Add U contributions
                for (i_local, j_local), coeff in u_part.items():
                    # A1 (a_blk=0): global col = j_local (0,1)
                    # A2 (a_blk=1): global col = j_local + 2 (2,3)
                    j_global = j_local + a_blk * 2
                    U[i_local * m + j_global][col] += coeff
                
                # Add V contributions
                # B_{b,1} (a_blk=0): rows 0,1 of B, cols 2b, 2b+1
                # B_{b,2} (a_blk=1): rows 2,3 of B, cols 2b, 2b+1
                for (j_local, k_local), coeff in v_part.items():
                    # j_local=0 or 1 within the 2x2 sub-block
                    # For a_blk=0: global row = j_local, global col = 2*b + k_local
                    # For a_blk=1: global row = j_local + 2, global col = 2*b + k_local
                    j_global = j_local + a_blk * 2
                    k_global = 2 * b + k_local
                    V[j_global * p + k_global][col] += coeff
                
                # Add W contributions
                for (i_out, c_off), coeff in w_part.items():
                    if coeff != 0:
                        # c_off=0 -> col 2b, c_off=1 -> col 2b+1
                        k_global = 2 * b + c_off
                        W[k_global * n + i_out][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
