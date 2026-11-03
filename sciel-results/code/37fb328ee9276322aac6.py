import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6

    # We use a Strassen-style block decomposition.
    # A is 2x4, B is 4x6.
    # Split A into 2 blocks of size 2x2: A = [A0, A1] where A0, A1 are 2x2
    # Split B into 3 blocks of size 4x2: B = [B0, B1, B2] where Bj is 4x2
    # Each Bj can be split into 2 blocks of 2x2: Bj = [Bj_top, Bj_bot]
    #
    # Actually, let's think more carefully.
    # A: 2x4 = 1x2 blocks of 2x2
    # B: 4x6 = 2x3 blocks of 2x2
    # C: 2x6 = 1x3 blocks of 2x2
    #
    # Standard approach: treat this as multiplying a row vector of 1x(2x2) by a column vector of (2x2)x1... 
    # but that's just the schoolbook.
    #
    # Better: Use the fact that 2x4 * 4x6 can be seen as:
    # A (2x4) * B (4x6)
    # Partition A into [A_left, A_right] each 2x2
    # Partition B into 3 columns of 4x2, each split top/bottom into 2x2
    #
    # Let me use a different framing. The tensor for 2x4*4x6 has indices:
    # T[i, j, k] where i in {0,1}, j in {0,1,2,3}, k in {0,...,5}
    # 
    # Strassen insight for 2x2*2x2: rank 7 instead of 8.
    # 
    # For 2x4 * 4x6:
    # Think of it as: A is 2x4, B is 4x6.
    # We can write B as a 2x3 array of 2x2 blocks: B[2,3] where each is 2x2.
    # A is a 1x2 array of 2x2 blocks: A[1,2] where each is 2x2.
    # Result C is a 1x3 array of 2x2 blocks.
    #
    # So we need to compute: for each column c in 0..2:
    #   C[:, 2c:2c+2] = A_left * B_top[:, 2c:2c+2] + A_right * B_bot[:, 2c:2c+2]
    #
    # That's 3 independent 2x2 * 2x2 multiplications (for the top part) 
    # plus 3 independent 2x2 * 2x2 multiplications (for the bottom part)
    # = 6 Strassen multiplications of rank 7 each = 42.
    #
    # Wait, that's not right either. Let me re-examine.
    #
    # A (2x4) * B (4x6):
    # A = [A0 | A1] where A0, A1 are 2x2
    # B rows 0,1 multiplied by A0, rows 2,3 multiplied by A1.
    # B (4x6) split into top 2x6 and bottom 2x6.
    # C = A0 * B_top + A1 * B_bot
    # B_top is 2x6, B_bot is 2x6.
    # So C = A0 * B_top + A1 * B_bot
    # This is: (2x2)*(2x6) + (2x2)*(2x6)
    # Each (2x2)*(2x6) can be done as 3 Strassen (2x2)*(2x2) products.
    # So total = 3 + 3 = 6 Strassen products, each rank 7 = 42.
    #
    # Let me construct this explicitly.
    #
    # Strassen for 2x2 * 2x2:
    # Given P (2x2) and Q (2x2):
    # M1 = (P00+P11)*(Q00+Q11)
    # M2 = (P10+P11)*Q00
    # M3 = P00*(Q01-Q11)
    # M4 = P11*(Q10-Q00)
    # M5 = (P00+P01)*Q11
    # M6 = (P10-P00)*(Q00+Q01)
    # M7 = (P01-P11)*(Q10+Q11)
    # Result:
    # R00 = M1 + M4 - M5 + M7
    # R01 = M3 + M5
    # R10 = M2 + M4
    # R11 = M1 - M2 + M3 + M6
    #
    # For (2x2) * (2x6) = (2x2) * [Q0 | Q1 | Q2] where Qj are 2x2:
    # We do 3 separate Strassen multiplications, one for each Qj.
    #
    # Total rank = 6 * 7 = 42.
    #
    # Now let me build the factor matrices.
    #
    # U has shape (n*m, rank) = (8, 42)
    # V has shape (m*p, rank) = (24, 42)  
    # W has shape (p*n, rank) = (12, 42)
    #
    # Indexing:
    # U[i*m+j, c] where i in 0..1, j in 0..3
    # V[j*p+k, c] where j in 0..3, k in 0..5
    # W[k*n+i, c] where k in 0..5, i in 0..1
    #
    # The 6 Strassen multiplications:
    # For col c in 0..2 (corresponding to output columns 2c, 2c+1):
    #   Top: A0 (2x2) * B_top[:, 2c:2c+2] (2x2)
    #   Bot: A1 (2x2) * B_bot[:, 2c:2c+2] (2x2)
    #
    # A0 = A[0:2, 0:2] -> indices (i, j) for i=0,1; j=0,1
    # A1 = A[0:2, 2:4] -> indices (i, j) for i=0,1; j=2,3
    # B_top[:, 2c:2c+2] = B[0:2, 2c:2c+2] -> indices (j, k) for j=0,1; k=2c,2c+1
    # B_bot[:, 2c:2c+2] = B[2:4, 2c:2c+2] -> indices (j, k) for j=2,3; k=2c,2c+1
    #
    # Output C[i, 2c:2c+2] = A0 * B_top_col + A1 * B_bot_col
    #
    # For each of the 6 Strassen multiplications, we have 7 intermediate values.
    # Each intermediate value is a bilinear form in the inputs.
    #
    # Let me define the 6 Strassen multiplications:
    # For each (source, col) where source in {top, bot}, col in {0,1,2}:
    #   P = A_source (2x2), Q = B_part[:, 2*col:2*col+2] (2x2)
    #   7 Strassen intermediates
    #
    # Total 42 intermediates.
    #
    # Let me build U, V, W column by column.
    
    rank = 42
    U = np.zeros((8, rank))
    V = np.zeros((24, rank))
    W = np.zeros((12, rank))
    
    # Strassen coefficients for 2x2*2x2:
    # M1 = (P00+P11)*(Q00+Q11)
    # M2 = (P10+P11)*Q00
    # M3 = P00*(Q01-Q11)
    # M4 = P11*(Q10-Q00)
    # M5 = (P00+P01)*Q11
    # M6 = (P10-P00)*(Q00+Q01)
    # M7 = (P01-P11)*(Q10+Q11)
    #
    # For each Strassen multiplication, we need to express:
    # U factors (from P), V factors (from Q), W factors (to C)
    #
    # Let's define helper: for a Strassen multiplication of P (2x2) * Q (2x2) -> R (2x2)
    # P indices: P00, P01, P10, P11
    # Q indices: Q00, Q01, Q10, Q11
    # R indices: R00, R01, R10, R11
    #
    # The 7 terms and their contributions:
    strassen_terms = [
        # (U_factors, V_factors, W_factors)
        # U_factors: list of (p_idx, coeff) for P
        # V_factors: list of (q_idx, coeff) for Q
        # W_factors: list of (r_idx, coeff) for R
        # M1 = (P00+P11)*(Q00+Q11)
        ([((0,0), 1), ((1,1), 1)],  # P00 + P11
         [((0,0), 1), ((1,1), 1)],  # Q00 + Q11
         [((0,0), 1), ((0,1), 0), ((1,0), 0), ((1,1), 1)]),  # R00 + R11
        # M2 = (P10+P11)*Q00
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),  # R10 - R11
        # M3 = P00*(Q01-Q11)
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),  # R01 + R11
        # M4 = P11*(Q10-Q00)
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),  # R00 + R10
        # M5 = (P00+P01)*Q11
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),  # -R00 + R01
        # M6 = (P10-P00)*(Q00+Q01)
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),  # R11
        # M7 = (P01-P11)*(Q10+Q11)
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),  # R00
    ]
    
    # Now, for each of the 6 Strassen multiplications:
    # source=0 (top): P = A0 = A[i, 0:2], Q = B[0:2, 2c:2c+2]
    # source=1 (bot): P = A1 = A[i, 2:4], Q = B[2:4, 2c:2c+2]
    
    col_idx = 0
    for source in range(2):  # 0=top, 1=bot
        for c in range(3):  # column block 0,1,2
            for t in range(7):  # 7 Strassen terms
                u_factors, v_factors, w_factors = strassen_terms[t]
                
                # Map P indices to U indices:
                # P is 2x2, P[pi, pj]
                # If source=0: P[pi,pj] = A[pi, pj] -> U[pi*4 + pj, col_idx]
                # If source=1: P[pi,pj] = A[pi, pj+2] -> U[pi*4 + (pj+2), col_idx]
                for (pi, pj), coeff in u_factors:
                    u_row = pi * 4 + (pj + 2*source)
                    U[u_row, col_idx] += coeff
                
                # Map Q indices to V indices:
                # Q is 2x2, Q[qi, qj]
                # Q[qi,qj] = B[qi + 2*source, 2*c + qj]
                # V index: (qi + 2*source) * 6 + (2*c + qj)
                for (qi, qj), coeff in v_factors:
                    v_row = (qi + 2*source) * 6 + (2*c + qj)
                    V[v_row, col_idx] += coeff
                
                # Map R indices to W indices:
                # R is 2x2, R[ri, rj]
                # R[ri,rj] contributes to C[ri, 2*c + rj]
                # W index: (2*c + rj) * 2 + ri
                for (ri, rj), coeff in w_factors:
                    if coeff != 0:
                        w_row = (2*c + rj) * 2 + ri
                        W[w_row, col_idx] += coeff
                
                col_idx += 1

    # Verification
    if time.time() < DEADLINE - 10:
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        A_flat = A.flatten() # shape (n*m)
        B_flat = B.flatten() # shape (m*p)
        
        u_vals = U @ A_flat # shape (rank)
        v_vals = V @ B_flat # shape (rank)
        
        W_reshaped = W.reshape(p, n, rank).transpose(1, 0, 2) # (n, p, rank)
        
        prod_vals = u_vals * v_vals # shape (rank)
        C_decomp = np.sum(W_reshaped * prod_vals[None, None, :], axis=2)
        
        if not np.allclose(C_true, C_decomp):
            # Fallback to trivial if Strassen fails
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, r] = 1.0
                        V[j * p + k, r] = 1.0
                        W[k * n + i, r] = 1.0
                        r += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))
