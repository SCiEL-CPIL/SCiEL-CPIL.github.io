"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen terms for multiplying Q (2x2) by R (2x2)
    # Q is A[qi:qi+2, qj:qj+2]
    # R is B[ri:ri+2, rj:rj+2]
    # Result P is C[pi:pi+2, pj:pj+2]
    # Strassen indices:
    # M1: (Q11+Q22)(R11+R22)
    # M2: (Q21+Q22)R11
    # M3: Q11(R12-R22)
    # M4: Q22(R21-R11)
    # M5: (Q11+Q12)R22
    # M6: (Q21-Q11)(R11+R12)
    # M7: (Q12-Q22)(R21+R22)
    
    def add_strassen(qi, qj, ri, rj, pi, pj, start_col):
        # Map local 2x2 indices (0,1) to global indices
        # Q indices: i in {0,1} -> qi+i, j in {0,1} -> qj+j
        # R indices: i in {0,1} -> ri+i, j in {0,1} -> rj+j
        # P indices: i in {0,1} -> pi+i, j in {0,1} -> pj+j
        
        # Term 1: (Q11+Q22)(R11+R22)
        # U: Q11 (qi, qj), Q22 (qi+1, qj+1)
        # V: R11 (ri, rj), R22 (ri+1, rj+1)
        # W: P11 (pi, pj), P12 (pi, pj+1), P21 (pi+1, pj), P22 (pi+1, pj+1)
        # P11 += M1, P12 += 0, P21 += 0, P22 += M1
        # Wait, Strassen P formulas:
        # P11 = M1 + M4 - M5 + M7
        # P12 = M3 + M5
        # P21 = M2 + M4
        # P22 = M1 - M2 + M3 + M6
        
        # M1 contributes to P11 and P22 with coeff 1
        U[qi*m + qj][start_col] += 1
        U[(qi+1)*m + (qj+1)][start_col] += 1
        V[ri*p + rj][start_col] += 1
        V[(ri+1)*p + (rj+1)][start_col] += 1
        W[rj*n + pi][start_col] += 1 # P11 is at row pi, col pj. W index is k*n+i. Here k=pj, i=pi.
        W[(rj+1)*n + (pi+1)][start_col] += 1 # P22 is at row pi+1, col pj+1. W index is (pj+1)*n + (pi+1).
        
        # M2: (Q21+Q22)R11
        # U: Q21 (qi+1, qj), Q22 (qi+1, qj+1)
        # V: R11 (ri, rj)
        # W: P21 (pi+1, pj) with coeff 1
        U[(qi+1)*m + qj][start_col+1] += 1
        U[(qi+1)*m + (qj+1)][start_col+1] += 1
        V[ri*p + rj][start_col+1] += 1
        W[rj*n + (pi+1)][start_col+1] += 1 # P21: row pi+1, col pj. k=pj, i=pi+1.
        
        # M3: Q11(R12-R22)
        # U: Q11 (qi, qj)
        # V: R12 (ri, rj+1) with 1, R22 (ri+1, rj+1) with -1
        # W: P12 (pi, pj+1) with 1, P22 (pi+1, pj+1) with 1
        U[qi*m + qj][start_col+2] += 1
        V[ri*p + (rj+1)][start_col+2] += 1
        V[(ri+1)*p + (rj+1)][start_col+2] -= 1
        W[(rj+1)*n + pi][start_col+2] += 1 # P12: row pi, col pj+1. k=pj+1, i=pi.
        W[(rj+1)*n + (pi+1)][start_col+2] += 1 # P22: row pi+1, col pj+1. k=pj+1, i=pi+1.
        
        # M4: Q22(R21-R11)
        # U: Q22 (qi+1, qj+1)
        # V: R21 (ri+1, rj) with 1, R11 (ri, rj) with -1
        # W: P11 (pi, pj) with 1, P21 (pi+1, pj) with 1
        U[(qi+1)*m + (qj+1)][start_col+3] += 1
        V[(ri+1)*p + rj][start_col+3] += 1
        V[ri*p + rj][start_col+3] -= 1
        W[rj*n + pi][start_col+3] += 1 # P11: row pi, col pj. k=pj, i=pi.
        W[rj*n + (pi+1)][start_col+3] += 1 # P21: row pi+1, col pj. k=pj, i=pi+1.
        
        # M5: (Q11+Q12)R22
        # U: Q11 (qi, qj), Q12 (qi, qj+1)
        # V: R22 (ri+1, rj+1)
        # W: P11 (pi, pj) with -1, P12 (pi, pj+1) with 1
        U[qi*m + qj][start_col+4] += 1
        U[qi*m + (qj+1)][start_col+4] += 1
        V[(ri+1)*p + (rj+1)][start_col+4] += 1
        W[rj*n + pi][start_col+4] -= 1 # P11: row pi, col pj. k=pj, i=pi.
        W[(rj+1)*n + pi][start_col+4] += 1 # P12: row pi, col pj+1. k=pj+1, i=pi.
        
        # M6: (Q21-Q11)(R11+R12)
        # U: Q21 (qi+1, qj) with 1, Q11 (qi, qj) with -1
        # V: R11 (ri, rj) with 1, R12 (ri, rj+1) with 1
        # W: P22 (pi+1, pj+1) with 1
        U[(qi+1)*m + qj][start_col+5] += 1
        U[qi*m + qj][start_col+5] -= 1
        V[ri*p + rj][start_col+5] += 1
        V[ri*p + (rj+1)][start_col+5] += 1
        W[(rj+1)*n + (pi+1)][start_col+5] += 1 # P22: row pi+1, col pj+1. k=pj+1, i=pi+1.
        
        # M7: (Q12-Q22)(R21+R22)
        # U: Q12 (qi, qj+1) with 1, Q22 (qi+1, qj+1) with -1
        # V: R21 (ri+1, rj) with 1, R22 (ri+1, rj+1) with 1
        # W: P11 (pi, pj) with 1
        U[qi*m + (qj+1)][start_col+6] += 1
        U[(qi+1)*m + (qj+1)][start_col+6] -= 1
        V[(ri+1)*p + rj][start_col+6] += 1
        V[(ri+1)*p + (rj+1)][start_col+6] += 1
        W[rj*n + pi][start_col+6] += 1 # P11: row pi, col pj. k=pj, i=pi.

    # 1. X = A_L B_TL -> C_L
    # A_L: qi=0, qj=0. B_TL: ri=0, rj=0. C_L: pi=0, pj=0.
    add_strassen(0, 0, 0, 0, 0, 0, 0)
    
    # 2. Y = A_R B_BL -> C_L
    # A_R: qi=0, qj=2. B_BL: ri=2, rj=0. C_L: pi=0, pj=0.
    add_strassen(0, 2, 2, 0, 0, 0, 7)
    
    # 3. Z = A_L B_TR -> C_R
    # A_L: qi=0, qj=0. B_TR: ri=0, rj=2. C_R: pi=0, pj=2.
    add_strassen(0, 0, 0, 2, 0, 2, 14)
    
    # 4. W = A_R B_BR -> C_R
    # A_R: qi=0, qj=2. B_BR: ri=2, rj=2. C_R: pi=0, pj=2.
    add_strassen(0, 2, 2, 2, 0, 2, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
