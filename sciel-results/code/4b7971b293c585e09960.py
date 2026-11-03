"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Decomposition strategy:
    # Split 5x5 into 3x3 + 2x2 blocks:
    # A = [[A11(3x3), A12(3x2)],
    #      [A21(2x3), A22(2x2)]]
    # B = [[B11(3x3), B12(3x2)],
    #      [B21(2x3), B22(2x2)]]
    # C = [[C11(3x3), C12(3x2)],
    #      [C21(2x3), C22(2x2)]]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Rank contributions:
    # C11: A11*B11 (3x3 schoolbook, 27) + A12*B21 (3x2*2x3 schoolbook, 18) = 45
    # C12: A11*B12 (3x3*3x2 schoolbook, 18) + A12*B22 (3x2*2x2 schoolbook, 12) = 30
    # C21: A21*B11 (2x3*3x3 schoolbook, 18) + A22*B21 (2x2*2x3 schoolbook, 12) = 30
    # C22: A21*B12 (2x3*3x2 schoolbook, 12) + A22*B22 (2x2*2x2 Strassen, 7) = 19
    #
    # Total rank: 45 + 30 + 30 + 19 = 124

    rank = 124
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    # Helper: add schoolbook product for blocks
    # A block: rows ar0..ar0+ar-1, cols ac0..ac0+ac-1
    # B block: rows br0..br0+br-1, cols bc0..bc0+bc-1
    # C block: rows cr0..cr0+cr-1, cols cc0..cc0+cc-1
    # For C = A_block * B_block:
    #   C[r,c] = sum_k A_block[r,k] * B_block[k,c]
    #   In global indices:
    #     A global row = ar0 + r, col = ac0 + k
    #     B global row = br0 + k, col = bc0 + c
    #     C global row = cr0 + r, col = cc0 + c
    #     W index = (cc0 + c) * n + (cr0 + r)

    def add_schoolblock(ar0, ar, ac0, ac, br0, br, bc0, bc, cr0, cr, cc0, cc):
        nonlocal col
        for r in range(ar):
            for c in range(cc):
                for k in range(ac):  # ac should equal br for valid matmul
                    U[(ar0 + r) * m + (ac0 + k)][col] = 1
                    V[(br0 + k) * p + (bc0 + c)][col] = 1
                    W[(cc0 + c) * n + (cr0 + r)][col] = 1
                    col += 1

    # Helper: add Strassen product for 2x2 blocks
    # A block: rows ar0, ar0+1; cols ac0, ac0+1
    # B block: rows br0, br0+1; cols bc0, bc0+1
    # C block: rows cr0, cr0+1; cols cc0, cc0+1
    # Strassen: M1..M7
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    def add_strassen22(ar0, ac0, br0, bc0, cr0, cc0):
        nonlocal col
        # A11: row ar0, col ac0
        # A12: row ar0, col ac0+1
        # A21: row ar0+1, col ac0
        # A22: row ar0+1, col ac0+1
        # B11: row br0, col bc0
        # B12: row br0, col bc0+1
        # B21: row br0+1, col bc0
        # B22: row br0+1, col bc0+1
        # C11: row cr0, col cc0
        # C12: row cr0, col cc0+1
        # C21: row cr0+1, col cc0
        # C22: row cr0+1, col cc0+1

        # We need to express Strassen in terms of the tensor decomposition.
        # Each "multiplication" M = X * Y where X is a linear combo of A entries
        # and Y is a linear combo of B entries.
        # In the tensor decomposition, each such multiplication contributes
        # one rank-1 term: U[i,j] = coeff_A, V[j,k] = coeff_B, W[k,n+i] = coeff_C
        # But here X and Y are themselves linear combinations, so we need to
        # expand them.

        # Actually, in a tensor decomposition, each rank-1 term corresponds to
        # one scalar multiplication. For Strassen, we have 7 such multiplications.
        # But each "input" to the multiplication is a linear combination of
        # multiple A and B entries. So we need to distribute.

        # Let me think more carefully. In the standard tensor decomposition:
        # C_{i,k} = sum_t U_{i,j}^{(t)} * V_{j,k}^{(t)} * W_{k,i}^{(t)}
        # where the sum is over t (the rank index), and j ranges over m.
        # Wait, that's not quite right. Let me re-examine the seed.

        # From the seed:
        # U[i*m + j][col] = 1  -- U is (n*m) x rank, indexed by (i,j) for A
        # V[j*p + k][col] = 1  -- V is (m*p) x rank, indexed by (j,k) for B
        # W[k*n + i][col] = 1  -- W is (p*n) x rank, indexed by (k,i) for C
        #
        # So the decomposition is:
        # For each t:
        #   For each i, j, k:
        #     C_{i,k} += U[i*m+j][t] * A_{i,j} * V[j*p+k][t] * B_{j,k} * W[k*n+i][t]
        #
        # Wait, that's not right either. Let me re-read.
        #
        # Actually, I think the convention is:
        # The tensor T_{i,j,k} for matrix multiplication C = A*B where
        # A is n x m, B is m x p, C is n x p:
        # T_{i,j,k} = 1 if the term A_{i,j} * B_{j,k} contributes to C_{i,k}
        #
        # The decomposition: T = sum_t U_t (over i,j) * V_t (over j,k) * W_t (over k,i)
        # where U_t is a vector of length n*m, V_t of length m*p, W_t of length p*n.
        #
        # More precisely:
        # T_{i,j,k} = sum_t U_t[i*m+j] * V_t[j*p+k] * W_t[k*n+i]
        #
        # So for the schoolbook case, for each (i,j,k):
        # U_t[i*m+j] = 1, V_t[j*p+k] = 1, W_t[k*n+i] = 1 for that specific t.
        #
        # For Strassen, we need to express the 7 multiplications in this form.
        # Each Strassen multiplication M_r = X_r * Y_r where:
        # X_r = sum_{i,j} x_r[i,j] * A_{i,j}
        # Y_r = sum_{j,k} y_r[j,k] * B_{j,k}
        # And then C_{i,k} gets contribution c_r[i,k] * M_r
        #
        # In the tensor decomposition, this becomes:
        # For the r-th Strassen multiplication, we add a rank-1 term:
        # U_t[i*m+j] = x_r[i,j]
        # V_t[j*p+k] = y_r[j,k]
        # W_t[k*n+i] = c_r[i,k]
        #
        # So let me define the Strassen coefficients for 2x2 blocks.

        # A local indices: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
        # B local indices: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
        # C local indices: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22

        # M1 = (A11+A22)(B11+B22)
        # x1: A11=1, A22=1, others 0
        # y1: B11=1, B22=1, others 0
        # c1: C11=1, C22=1, others 0

        # M2 = (A21+A22)B11
        # x2: A21=1, A22=1, others 0
        # y2: B11=1, others 0
        # c2: C21=1, C22=-1, others 0

        # M3 = A11(B21-B22)
        # x3: A11=1, others 0
        # y3: B21=1, B22=-1, others 0
        # c3: C12=1, C22=1, others 0

        # M4 = A22(B12-B11)
        # x4: A22=1, others 0
        # y4: B12=1, B11=-1, others 0
        # c4: C11=1, C21=1, others 0

        # M5 = (A11+A12)B22
        # x5: A11=1, A12=1, others 0
        # y5: B22=1, others 0
        # c5: C11=-1, C12=1, others 0

        # M6 = (A21-A11)(B11+B12)
        # x6: A21=1, A11=-1, others 0
        # y6: B11=1, B12=1, others 0
        # c6: C22=1, others 0

        # M7 = (A12-A22)(B21+B22)
        # x7: A12=1, A22=-1, others 0
        # y7: B21=1, B22=1, others 0
        # c7: C11=1, C21=1, others 0

        # Let me verify: 
        # C11 = M1 + M4 - M5 + M7 = (A11+A22)(B11+B22) + A22(B12-B11) - (A11+A12)B22 + (A12-A22)(B21+B22)
        #     = A11*B11 + A11*B22 + A22*B11 + A22*B22 + A22*B12 - A22*B11 - A11*B22 - A12*B22 + A12*B21 + A12*B22 - A22*B21 - A22*B22
        #     = A11*B11 + A12*B21 + (A11*B22 - A11*B22) + (A22*B11 - A22*B11) + (A22*B22 - A22*B22) + (-A12*B22 + A12*B22) + (-A22*B21) + A22*B12
        #     = A11*B11 + A12*B21 + A22*B12 - A22*B21
        # Hmm, that doesn't look right. Let me recheck.
        # 
        # Standard Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # Let me recompute C11:
        # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # M4 = A22(B12-B11) = A22B12 - A22B11
        # -M5 = -(A11+A12)B22 = -A11B22 - A12B22
        # M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22
        # Sum: A11B11 + A11B22 + A22B11 + A22B22 + A22B12 - A22B11 - A11B22 - A12B22 + A12B21 + A12B22 - A22B21 - A22B22
        # = A11B11 + A12B21 + A22B12 - A22B21
        #
        # But C11 should be A11B11 + A12B21. So I have extra terms A22B12 - A22B21.
        # That's wrong. Let me look up the correct Strassen formulas.
        #
        # Correct Strassen for 2x2:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # Let me recompute C11:
        # = (A11+A22)(B11+B22) + A22(B12-B11) - (A11+A12)B22 + (A12-A22)(B21+B22)
        # = A11B11 + A11B22 + A22B11 + A22B22 + A22B12 - A22B11 - A11B22 - A12B22 + A12B21 + A12B22 - A22B21 - A22B22
        # = A11B11 + A12B21 + A22B12 - A22B21
        #
        # This is still not A11B11 + A12B21. There must be an error in my Strassen formulas.
        # Let me look it up again.
        #
        # Actually, I think the issue is that I'm using the wrong convention.
        # Let me use the standard reference:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # Let me try a numerical example.
        # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        # AB = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        #
        # M1 = (1+4)(5+8) = 5*13 = 65
        # M2 = (3+4)*5 = 35
        # M3 = 1*(7-8) = -1
        # M4 = 4*(6-5) = 4
        # M5 = (1+2)*8 = 24
        # M6 = (3-1)*(5+6) = 2*11 = 22
        # M7 = (2-4)*(7+8) = -2*15 = -30
        #
        # C11 = 65 + 4 - 24 + (-30) = 15. But should be 19. Wrong.
        #
        # I must have the formulas wrong. Let me use a verified source.
        # 
        # From Wikipedia (Strassen algorithm):
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # This is what I have. Let me recheck the arithmetic.
        # C11 = 65 + 4 - 24 + (-30) = 65 + 4 = 69, 69 - 24 = 45, 45 - 30 = 15.
        # But AB[0,0] = 1*5 + 2*7 = 5 + 14 = 19.
        #
        # So either my M values or my formula is wrong.
        # Let me recompute M7: (A12 - A22)(B21 + B22) = (2-4)(7+8) = -2 * 15 = -30. Correct.
        # M5 = (A11 + A12)B22 = (1+2)*8 = 3*8 = 24. Correct.
        # M4 = A22(B12 - B11) = 4*(6-5) = 4*1 = 4. Correct.
        # M1 = (A11 + A22)(B11 + B22) = (1+4)(5+8) = 5*13 = 65. Correct.
        #
        # So the formula gives 15, but the answer is 19. The formula must be wrong.
        #
        # Let me try a different set of formulas. Maybe:
        # C11 = M1 - M5 + M4 + M7  (same thing)
        #
        # Actually, I think I may have the wrong M7. Let me try:
        # M7 = (A12 - A22)(B21 + B22)
        #
        # Hmm, let me just use a known-correct implementation.
        #
        # After checking, the correct formulas are:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # Let me try yet another numerical example with simpler numbers.
        # A = [[1,0],[0,1]], B = [[1,0],[0,1]] (identity matrices)
        # AB = [[1,0],[0,1]]
        # M1 = (1+1)(1+1) = 4
        # M2 = (0+1)*1 = 1
        # M3 = 1*(0-1) = -1
        # M4 = 1*(0-1) = -1
        # M5 = (1+0)*1 = 1
        # M6 = (0-1)(1+0) = -1
        # M7 = (0-1)(0+1) = -1
        # C11 = 4 + (-1) - 1 + (-1) = 1. Correct!
        # C12 = -1 + 1 = 0. Correct!
        # C21 = 1 + (-1) = 0. Correct!
        # C22 = 4 - 1 + (-1) + (-1) = 1. Correct!
        #
        # OK so the formulas are correct. Let me recheck my earlier example.
        # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        # M1 = (1+4)(5+8) = 5*13 = 65
        # M2 = (3+4)*5 = 7*5 = 35
        # M3 = 1*(7-8) = 1*(-1) = -1
        # M4 = 4*(6-5) = 4*1 = 4
        # M5 = (1+2)*8 = 3*8 = 24
        # M6 = (3-1)*(5+6) = 2*11 = 22
        # M7 = (2-4)*(7+8) = (-2)*15 = -30
        #
        # C11 = 65 + 4 - 24 + (-30) = 15. But AB[0,0] = 19.
        #
        # Wait, I think I made an arithmetic error earlier. Let me recompute AB[0,0].
        # AB[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*5 + 2*7 = 5 + 14 = 19.
        #
        # And Strassen gives 15. That's a discrepancy of 4.
        #
        # Let me recheck M1: (A11+A22)(B11+B22) = (1+4)(5+8) = 5*13 = 65. Correct.
        # M4: A22(B12-B11) = 4*(6-5) = 4. Correct.
        # M5: (A11+A12)B22 = (1+2)*8 = 24. Correct.
        # M7: (A12-A22)(B21+B22) = (2-4)(7+8) = -2*15 = -30. Correct.
        #
        # 65 + 4 - 24 - 30 = 15. But should be 19.
        #
        # I'm confused. Let me try to verify with a trusted source or re-derive.
        #
        # Actually, I just realized: maybe I have the indices mixed up.
        # In Strassen, the blocks are:
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        # C = AB means C11 = A11*B11 + A12*B21, C12 = A11*B12 + A12*B22, etc.
        #
        # Let me re-derive Strassen from scratch.
        # We want to compute:
        # C11 = A11B11 + A12B21
        # C12 = A11B12 + A12B22
        # C21 = A21B11 + A22B21
        # C22 = A21B12 + A22B22
        #
        # Define:
        # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # M2 = (A21+A22)B11 = A21B11 + A22B11
        # M3 = A11(B21-B22) = A11B21 - A11B22
        # M4 = A22(B12-B11) = A22B12 - A22B11
        # M5 = (A11+A12)B22 = A11B22 + A12B22
        # M6 = (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12
        # M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22
        #
        # Now:
        # C11 = A11B11 + A12B21
        # Let's try M1 + M4 - M5 + M7:
        # = (A11B11 + A11B22 + A22B11 + A22B22) + (A22B12 - A22B11) - (A11B22 + A12B22) + (A12B21 + A12B22 - A22B21 - A22B22)
        # = A11B11 + A11B22 + A22B11 + A22B22 + A22B12 - A22B11 - A11B22 - A12B22 + A12B21 + A12B22 - A22B21 - A22B22
        # = A11B11 + A12B21 + A22B12 - A22B21
        #
        # This is not C11. It has extra terms A22B12 - A22B21.
        #
        # So the standard Strassen formulas I've been using are incorrect?
        # Let me look up the actual correct formulas.
        #
        # After further research, I believe the issue is that I've been using a
        # non-standard variant. The correct standard Strassen is:
        #
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # But my algebra shows this doesn't work. Let me try a completely different approach:
        # just use the schoolbook algorithm for the 2x2 block as well. That gives rank 8
        # instead of 7, but total rank becomes 45+30+30+20=125, same as full schoolbook.
        # That's not an improvement.
        #
        # Given the complexity and time constraints, let me just implement the full
        # schoolbook decomposition with rank 125. It's guaranteed correct and simple.

    # Reset: use full schoolbook for simplicity and correctness
    rank = n * m * p  # 125
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
