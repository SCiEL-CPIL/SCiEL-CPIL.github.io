"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """Strassen-based rank-28 decomposition for 2x4 by 4x4 matrix multiplication."""
    n, m, p = 2, 4, 4
    # Split: A is 2x4 -> two 2x2 blocks A0 (cols 0,1), A1 (cols 2,3)
    # B is 4x4 -> four 2x2 blocks B00 (rows 0,1 cols 0,1), B01 (rows 0,1 cols 2,3),
    #                        B10 (rows 2,3 cols 0,1), B11 (rows 2,3 cols 2,3)
    # C is 2x4 -> two 2x2 blocks C0 (cols 0,1), C1 (cols 2,3)
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    # Each 2x2*2x2 uses Strassen rank 7, total rank = 4*7 = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen 2x2*2x2: given P (2x2) and Q (2x2), compute R = P*Q
    # Index convention: P[a,b], Q[b,c], R[a,c] with a,b,c in {0,1}
    # Strassen:
    # M1 = (P00+P11)*(Q00+Q11)
    # M2 = (P10+P11)*Q00
    # M3 = P00*(Q01-Q11)
    # M4 = P11*(Q10-Q00)
    # M5 = (P00+P01)*Q11
    # M6 = (P10-P00)*(Q00+Q01)
    # M7 = (P01-P11)*(Q10+Q11)
    # R00 = M1+M4-M5+M7
    # R01 = M3+M5
    # R10 = M2+M4
    # R11 = M1-M2+M3+M6
    
    # For each of the 4 terms (A_block * B_block), we need to map:
    # A_block indices: (ai, aj) where ai in {0,1} (row of A), aj in {0,1} (local col)
    #   Global A index: ai*m + (block_col_offset + aj)
    # B_block indices: (bj, bk) where bj in {0,1} (local row), bk in {0,1} (local col)
    #   Global B index: (block_row_offset + bj)*p + (block_col_offset + bk)
    # C_block indices: (ci, ck) where ci in {0,1} (row), ck in {0,1} (local col)
    #   Global C index: ck*n + ci  (trace convention W[k*n+i])
    
    # Strassen intermediate patterns for 2x2*2x2:
    # Each intermediate Mx involves products of sums/differences of P and Q entries.
    # We need to express each Mx as a rank-1 tensor: sum of U_val * V_val * W_val
    # But Strassen intermediates are products of linear combinations, so each Mx is
    # itself a product of two linear forms. In the tensor decomposition, each Mx
    # contributes one rank-1 term where U and V encode the linear combos.
    
    # Actually, for exact tensor decomposition, each "multiplication" in Strassen
    # corresponds to one rank-1 component. The linear combinations are absorbed
    # into the U and V matrices (they can have entries -1, 0, 1).
    
    # Strassen rank-1 components for 2x2*2x2 (P*Q=R):
    # Component 1: M1 = (P00+P11)*(Q00+Q11)
    #   U: P00->1, P11->1; V: Q00->1, Q11->1; W: R00+=1, R11+=1
    # Component 2: M2 = (P10+P11)*Q00
    #   U: P10->1, P11->1; V: Q00->1; W: R10+=1, R11-=1
    # Component 3: M3 = P00*(Q01-Q11)
    #   U: P00->1; V: Q01->1, Q11->-1; W: R01+=1, R11+=1
    # Component 4: M4 = P11*(Q10-Q00)
    #   U: P11->1; V: Q10->1, Q00->-1; W: R00+=1, R10+=1
    # Component 5: M5 = (P00+P01)*Q11
    #   U: P00->1, P01->1; V: Q11->1; W: R00-=1, R01+=1
    # Component 6: M6 = (P10-P00)*(Q00+Q01)
    #   U: P10->1, P00->-1; V: Q00->1, Q01->1; W: R11+=1
    # Component 7: M7 = (P01-P11)*(Q10+Q11)
    #   U: P01->1, P11->-1; V: Q10->1, Q11->1; W: R00+=1
    
    strassen_components = [
        # (U_entries: {P_idx: val}, V_entries: {Q_idx: val}, W_entries: {R_idx: val})
        # P_idx: (row, col) in 2x2, Q_idx: (row, col) in 2x2, R_idx: (row, col) in 2x2
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # The 4 terms: (A_block, B_block, C_block)
    # A_block 0: cols 0,1 -> global col offset 0
    # A_block 1: cols 2,3 -> global col offset 2
    # B_block 00: rows 0,1 cols 0,1 -> row offset 0, col offset 0
    # B_block 01: rows 0,1 cols 2,3 -> row offset 0, col offset 2
    # B_block 10: rows 2,3 cols 0,1 -> row offset 2, col offset 0
    # B_block 11: rows 2,3 cols 2,3 -> row offset 2, col offset 2
    # C_block 0: cols 0,1 -> col offset 0
    # C_block 1: cols 2,3 -> col offset 2
    
    terms = [
        # (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
        (0, 0, 0, 0),  # A0 * B00 -> C0
        (2, 2, 0, 0),  # A1 * B10 -> C0
        (0, 0, 2, 2),  # A0 * B01 -> C1
        (2, 2, 2, 2),  # A1 * B11 -> C1
    ]
    
    col = 0
    for (a_co, b_ro, b_co, c_co) in terms:
        for (u_pat, v_pat, w_pat) in strassen_components:
            for (pi, pj), val in u_pat.items():
                # A global index: row=pi, col=a_co+pj
                U[pi * m + (a_co + pj)][col] += val
            for (qi, qj), val in v_pat.items():
                # B global index: row=b_ro+qi, col=b_co+qj
                V[(b_ro + qi) * p + (b_co + qj)][col] += val
            for (ri, rj), val in w_pat.items():
                # C global index: W[k*n+i] where k=c_co+rj, i=ri
                W[(c_co + rj) * n + ri][col] += val
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
