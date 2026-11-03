import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    """
    Construct a rank-42 tensor decomposition for 2x4 * 4x6 matrix multiplication.
    
    Strategy: 
    1. Partition A (2x4) into A_left (2x2) and A_right (2x2).
    2. Partition B (4x6) into B_top (2x6) and B_bot (2x6).
    3. Partition B_top and B_bot into three 2x2 column blocks each.
    4. This yields 6 independent 2x2 * 2x2 multiplications.
    5. Apply Strassen's algorithm (rank 7) to each.
    6. Total rank = 6 * 7 = 42.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    rank = 42
    U = np.zeros((n * m, rank))  # (8, 42)
    V = np.zeros((m * p, rank))  # (24, 42)
    W = np.zeros((p * n, rank))  # (12, 42)
    
    # Strassen terms for 2x2 * 2x2:
    # M1 = (P00+P11)*(Q00+Q11)
    # M2 = (P10+P11)*Q00
    # M3 = P00*(Q01-Q11)
    # M4 = P11*(Q10-Q00)
    # M5 = (P00+P01)*Q11
    # M6 = (P10-P00)*(Q00+Q01)
    # M7 = (P01-P11)*(Q10+Q11)
    #
    # Format: (U_factors, V_factors, W_factors)
    # U_factors: [( (row, col), coeff ), ...] for P
    # V_factors: [( (row, col), coeff ), ...] for Q
    # W_factors: [( (row, col), coeff ), ...] for R
    
    strassen_terms = [
        # M1
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
        # M2
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        # M3
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        # M4
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        # M5
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        # M6
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        # M7
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]
    
    col_idx = 0
    for source in range(2):  # 0=A_left (top B), 1=A_right (bot B)
        for c in range(3):  # column block 0, 1, 2
            if time.time() > deadline:
                break
            for t in range(7):
                if time.time() > deadline:
                    break
                u_factors, v_factors, w_factors = strassen_terms[t]
                
                for (pi, pj), coeff in u_factors:
                    # U index: i*m + j
                    # P is A[pi, pj + 2*source]
                    u_row = pi * m + (pj + 2 * source)
                    U[u_row, col_idx] += coeff
                
                for (qi, qj), coeff in v_factors:
                    # V index: j*p + k
                    # Q is B[qi + 2*source, 2*c + qj]
                    v_row = (qi + 2 * source) * p + (2 * c + qj)
                    V[v_row, col_idx] += coeff
                
                for (ri, rj), coeff in w_factors:
                    # W index: k*n + i
                    # R is C[ri, 2*c + rj]
                    w_row = (2 * c + rj) * n + ri
                    W[w_row, col_idx] += coeff
                
                col_idx += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
