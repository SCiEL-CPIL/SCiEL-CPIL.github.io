"""Rewritten program for 2x4 * 4x6 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-42 tensor decomposition for 2x4 * 4x6 matrix multiplication.
    
    Strategy: Partition A (2x4) into two 2x2 blocks A_left, A_right.
    Partition B (4x6) into two 2x6 blocks B_top, B_bot.
    C = A_left * B_top + A_right * B_bot
    
    Each 2x6 block is split into three 2x2 column blocks.
    So we have 6 independent 2x2 * 2x2 multiplications.
    Each uses Strassen's rank-7 algorithm.
    Total rank = 6 * 7 = 42.
    
    Index conventions:
    - U has shape (8, rank): indices (i, j) where i in {0,1}, j in {0..3}
      U[i*4+j, t]
    - V has shape (24, rank): indices (j, k) where j in {0..3}, k in {0..5}
      V[j*6+k, t]
    - W has shape (12, rank): indices (k*n+i, r) = (k*2+i, k*2+r) where i,r in {0,1}, k in {0..5}
      W[(k*2+i)*6 + (k*2+r), t] -- wait, let me re-read.
    
    Actually, the output factor follows W[k*n+i, r]. With n=2, p=6:
    W has shape (p*n, rank) = (12, rank).
    The index is k*n + i where k in {0..5}, i in {0,1}.
    So W[k*2+i, t] corresponds to output element C[i, k].
    
    Wait, let me re-read: "W[k*n+i,r]". This means the row index of W is k*n+i.
    With n=2, k ranges over 0..5 (p values), i ranges over 0..1 (n values).
    So W has 12 rows, indexed by k*2+i.
    
    The tensor decomposition is:
    C[i,k] = sum_t U[i,j,t] * V[j,k,t] * W[i,k,t]
    
    But W is indexed by (k*n+i), so W[k*2+i, t] is the coefficient for C[i,k].
    
    Let me verify: the product sum_t U[i*4+j, t] * V[j*6+k, t] * W[k*2+i, t] should equal
    the standard matrix product sum_j A[i,j]*B[j,k].
    
    For Strassen, we need to be careful about how the W factors combine.
    
    Let me redefine more carefully.
    
    For a 2x2 * 2x2 Strassen decomposition with P (2x2) and Q (2x2) giving R (2x2):
    
    The 7 Strassen products are:
    M1 = (P00+P11)(Q00+Q11)
    M2 = (P10+P11)(Q00)
    M3 = (P00)(Q01-Q11)
    M4 = (P11)(Q10-Q00)
    M5 = (P00+P01)(Q11)
    M6 = (P10-P00)(Q00+Q01)
    M7 = (P01-P11)(Q10+Q11)
    
    Result:
    R00 = M1 - M5 + M4 + M7
    R01 = M3 + M5
    R10 = M2 + M4
    R11 = M1 - M2 + M3 + M6
    
    For each Strassen term Mt, we have:
    - U factors: which P entries with what coefficients
    - V factors: which Q entries with what coefficients  
    - W factors: which R entries (i.e., C entries) with what coefficients
    
    Let me define the Strassen terms more carefully.
    
    For term t, the contribution to C is:
    C[i, r] += coeff_W(i, r) * (sum over P entries: coeff_U * P) * (sum over Q entries: coeff_V * Q)
    
    So W[i, r, t] = coeff_W(i, r) for the relevant i, r.
    
    Let me tabulate:
    
    M1: U = {P00:1, P11:1}, V = {Q00:1, Q11:1}, W = {R00:1, R11:1}
    M2: U = {P10:1, P11:1}, V = {Q00:1}, W = {R10:1, R11:-1}
    M3: U = {P00:1}, V = {Q01:1, Q11:-1}, W = {R01:1, R11:1}
    M4: U = {P11:1}, V = {Q10:1, Q00:-1}, W = {R00:1, R10:1}
    M5: U = {P00:1, P01:1}, V = {Q11:1}, W = {R00:-1, R01:1}
    M6: U = {P10:1, P00:-1}, V = {Q00:1, Q01:1}, W = {R11:1}
    M7: U = {P01:1, P11:-1}, V = {Q10:1, Q11:1}, W = {R00:1}
    
    Now, for our 2x4 * 4x6 problem:
    
    A[i, j] for i in {0,1}, j in {0,1,2,3}
    B[j, k] for j in {0,1,2,3}, k in {0,1,2,3,4,5}
    
    Split:
    A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    B_top = B[0:2, :] (2x6), B_bot = B[2:4, :] (2x6)
    
    C = A_left * B_top + A_right * B_bot
    
    B_top = [B_top_col0 | B_top_col1 | B_top_col2] where each is 2x2
    B_bot = [B_bot_col0 | B_bot_col1 | B_bot_col2] where each is 2x2
    
    So:
    C[:, 0:2] = A_left * B_top_col0 + A_right * B_bot_col0
    C[:, 2:4] = A_left * B_top_col1 + A_right * B_bot_col1
    C[:, 4:6] = A_left * B_top_col2 + A_right * B_bot_col2
    
    For each column block c in {0,1,2}:
    C[:, 2c:2c+2] = A_left * B_top[:, 2c:2c+2] + A_right * B_bot[:, 2c:2c+2]
    
    This is a sum of two 2x2*2x2 products. We can apply Strassen to each.
    
    For source s in {0, 1} (0=A_left, 1=A_right):
    For column block c in {0, 1, 2}:
      For Strassen term t in {0..6}:
        U factors: P entries from A_s
        V factors: Q entries from B_s_col_c
        W factors: R entries -> C entries
    
    Let me map indices:
    
    For A_s (s=0 or 1):
    A_s[pi, pj] = A[pi, pj + 2*s]
    So U row index = pi * 4 + (pj + 2*s)
    
    For B_s_col_c:
    B_s_col_c[qi, qj] = B[qi + 2*s, 2*c + qj]
    So V row index = (qi + 2*s) * 6 + (2*c + qj)
    
    For C output:
    C[ri, rj] where rj is in {2c, 2c+1}
    W row index = (2*c + rj) * 2 + ri  (since W[k*n+i] with k=2c+rj, n=2, i=ri)
    
    Let me now build this.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    rank = 42
    U = np.zeros((n * m, rank))  # (8, 42)
    V = np.zeros((m * p, rank))  # (24, 42)
    W = np.zeros((p * n, rank))  # (12, 42)
    
    # Strassen terms: (u_factors, v_factors, w_factors)
    # Each factor is a list of ((local_row, local_col), coeff)
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
    for source in range(2):  # 0=A_left, 1=A_right
        for c in range(3):  # column block
            for t in range(7):
                u_factors, v_factors, w_factors = strassen_terms[t]
                
                for (pi, pj), coeff in u_factors:
                    u_row = pi * m + (pj + 2 * source)
                    U[u_row, col_idx] += coeff
                
                for (qi, qj), coeff in v_factors:
                    v_row = (qi + 2 * source) * p + (2 * c + qj)
                    V[v_row, col_idx] += coeff
                
                for (ri, rj), coeff in w_factors:
                    w_row = (2 * c + rj) * n + ri
                    W[w_row, col_idx] += coeff
                
                col_idx += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
