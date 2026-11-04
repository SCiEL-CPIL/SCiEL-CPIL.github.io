"""Tensor-train / hierarchical compression for 4x4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Strategy: Split k-index (0..6) into two groups:
    #   Group A: k in {0,1,2,3} (4 values) -> use Strassen for 4x4x4
    #   Group B: k in {4,5,6} (3 values)   -> use schoolbook for 4x4x3
    #
    # For 4x4x4 via block Strassen:
    # View 4x4x4 as 2x2 blocks of 2x2 matrices.
    # Use 7 Strassen multiplications, each a 2x2x2 product.
    # Each 2x2x2 product uses schoolbook (rank 4).
    # Total rank for 4x4x4 part = 7 * 4 = 28.
    #
    # For 4x4x3 via schoolbook: rank = 4*4*3 = 48.
    #
    # Total rank = 28 + 48 = 76.
    
    # Strassen decomposition for 2x2x2 (rank 7):
    # Each Strassen term t has:
    #   a_t[I,J] for I,J in {0,1}
    #   b_t[J,K] for J,K in {0,1}
    #   c_t[I,K] for I,K in {0,1}
    #
    # M0 = (A00+A11)(B00+B11)
    #   a = [[1,0],[0,1]], b = [[1,0],[0,1]], c = [[1,0],[0,1]] (but c is for output)
    #   Actually: C00 += M0, C11 += M0
    #   So c = [[1,0],[0,1]]
    #
    # Let me define the Strassen terms properly.
    # For each term t, we have:
    #   A_comb[I,jp] = sum_{Ia} a_t[Ia,J] * A[2*Ia+ip, 2*J+jp]  -- wait, this is getting confusing
    #
    # Let me use a cleaner formulation.
    #
    # The 7 Strassen terms for 2x2x2 are:
    # t=0: A0 = A00+A11, B0 = B00+B11, C00 += A0*B0, C11 += A0*B0
    # t=1: A1 = A10+A11, B1 = B00, C10 += A1*B1
    # t=2: A2 = A00, B2 = B10-B11, C01 += A2*B2
    # t=3: A3 = A11, B3 = B01-B00, C00 += A3*B3
    # t=4: A4 = A00+A01, B4 = B11, C01 += A4*B4
    # t=5: A5 = A10-A00, B5 = B00+B01, C11 += A5*B5
    # t=6: A6 = A01-A11, B6 = B10+B11, C00 += A6*B6
    #
    # C00 = M0 + M3 + M6 - M4  (wait, let me recheck)
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    #
    # So the c coefficients are:
    # t=0: c = [[1,0],[0,1]]  (M0 contributes to C00 and C11)
    # t=1: c = [[0,0],[1,0]]  (M1 contributes to C10)
    # t=2: c = [[0,1],[0,0]]  (M2 contributes to C01)
    # t=3: c = [[1,0],[1,0]]  (M3 contributes to C00 and C10)
    # t=4: c = [[0,1],[0,0]]  (M4 contributes to C01)
    # t=5: c = [[0,0],[0,1]]  (M5 contributes to C11)
    # t=6: c = [[1,0],[0,0]]  (M6 contributes to C00)
    #
    # Wait, that's not right. Let me be more careful.
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    #
    # So:
    # t=0 (M0): contributes +1 to C00, +1 to C11 -> c = [[1,0],[0,1]]
    # t=1 (M1): contributes +1 to C10 -> c = [[0,0],[1,0]]
    # t=2 (M2): contributes +1 to C01, +1 to C11 -> c = [[0,1],[0,1]]
    # t=3 (M3): contributes +1 to C00, +1 to C10 -> c = [[1,0],[1,0]]
    # t=4 (M4): contributes -1 to C00, +1 to C01 -> c = [[-1,1],[0,0]]
    # t=5 (M5): contributes +1 to C11 -> c = [[0,0],[0,1]]
    # t=6 (M6): contributes +1 to C00 -> c = [[1,0],[0,0]]
    
    # a coefficients: a_t[Ia, J] where Ia is the row block index, J is the col block index
    # b coefficients: b_t[J, Kb] where J is the row block index, Kb is the col block index
    #
    # t=0: A00+A11, B00+B11
    #   a = [[1,0],[1,0]]  (a[Ia,J]: Ia=0,J=0 -> 1; Ia=1,J=0 -> 1)
    #   Wait, A00 is block (0,0), A11 is block (1,1)
    #   So a[0,0]=1, a[1,1]=1, others 0
    #   a = [[1,0],[0,1]]
    #   b = [[1,0],[0,1]]
    #
    # t=1: A10+A11, B00
    #   A10 is block (1,0), A11 is block (1,1)
    #   a[1,0]=1, a[1,1]=1
    #   a = [[0,0],[1,1]]
    #   b[0,0]=1
    #   b = [[1,0],[0,0]]
    #
    # t=2: A00, B10-B11
    #   a[0,0]=1
    #   a = [[1,0],[0,0]]
    #   b[1,0]=1, b[1,1]=-1
    #   b = [[0,0],[1,-1]]
    #
    # t=3: A11, B01-B00
    #   a[1,1]=1
    #   a = [[0,0],[0,1]]
    #   b[0,1]=1, b[0,0]=-1
    #   b = [[-1,1],[0,0]]
    #
    # t=4: A00+A01, B11
    #   a[0,0]=1, a[0,1]=1
    #   a = [[1,1],[0,0]]
    #   b[1,1]=1
    #   b = [[0,0],[0,1]]
    #
    # t=5: A10-A00, B00+B01
    #   a[1,0]=1, a[0,0]=-1
    #   a = [[-1,0],[1,0]]
    #   b[0,0]=1, b[0,1]=1
    #   b = [[1,1],[0,0]]
    #
    # t=6: A01-A11, B10+B11
    #   a[0,1]=1, a[1,1]=-1
    #   a = [[0,1],[-1,0]]  -- wait, a[0,1]=1 means Ia=0,J=1
    #   Actually: A01 is block (0,1), A11 is block (1,1)
    #   a[0,1]=1, a[1,1]=-1
    #   a = [[0,1],[-1,0]]  -- no, a[Ia,J]: a[0,1]=1, a[1,1]=-1
    #   So a = [[0,1],[-1,0]] is wrong. a[0,1]=1, a[1,1]=-1
    #   a = [[0,1],[-1,0]] would mean a[0,0]=0, a[0,1]=1, a[1,0]=-1, a[1,1]=0
    #   But we want a[0,1]=1, a[1,1]=-1
    #   So a = [[0,1],[0,-1]]
    #   b[1,0]=1, b[1,1]=1
    #   b = [[0,0],[1,1]]
    
    # Let me redefine clearly:
    # a_t is a 2x2 matrix where a_t[Ia, J] is the coefficient of A_block[Ia, J]
    # b_t is a 2x2 matrix where b_t[J, Kb] is the coefficient of B_block[J, Kb]
    # c_t is a 2x2 matrix where c_t[Ic, Kc] is the coefficient for C_block[Ic, Kc]
    
    a_coeffs = [
        [[1, 0], [0, 1]],  # t=0: A00+A11
        [[0, 0], [1, 1]],  # t=1: A10+A11
        [[1, 0], [0, 0]],  # t=2: A00
        [[0, 0], [0, 1]],  # t=3: A11
        [[1, 1], [0, 0]],  # t=4: A00+A01
        [[-1, 0], [1, 0]], # t=5: A10-A00
        [[0, 1], [0, -1]], # t=6: A01-A11
    ]
    
    b_coeffs = [
        [[1, 0], [0, 1]],  # t=0: B00+B11
        [[1, 0], [0, 0]],  # t=1: B00
        [[0, 0], [1, -1]], # t=2: B10-B11
        [[-1, 1], [0, 0]], # t=3: B01-B00
        [[0, 0], [0, 1]],  # t=4: B11
        [[1, 1], [0, 0]],  # t=5: B00+B01
        [[0, 0], [1, 1]],  # t=6: B10+B11
    ]
    
    c_coeffs = [
        [[1, 0], [0, 1]],  # t=0: C00+=M0, C11+=M0
        [[0, 0], [1, 0]],  # t=1: C10+=M1
        [[0, 1], [0, 1]],  # t=2: C01+=M2, C11+=M2
        [[1, 0], [1, 0]],  # t=3: C00+=M3, C10+=M3
        [[-1, 1], [0, 0]], # t=4: C00-=M4, C01+=M4
        [[0, 0], [0, 1]],  # t=5: C11+=M5
        [[1, 0], [0, 0]],  # t=6: C00+=M6
    ]
    
    # Now construct the rank-1 terms.
    # For each Strassen term t and each (ip, jp, kp) in {0,1}^3:
    # We create a rank-1 term where:
    #   U[2*Ia+ip, 2*J+jp] = a_t[Ia, J] for all Ia, J
    #   V[2*J+jp, 2*Kb+kp] = b_t[J, Kb] for all J, Kb
    #   W[2*Kc+kp, 2*Ic+ip] = c_t[Ic, Kc] for all Ic, Kc
    #
    # But wait, this doesn't work because U, V, W are fixed vectors, not matrices.
    #
    # The correct approach: For each (t, ip, jp, kp), we need to verify that
    # the product U * V * W gives the correct contribution.
    #
    # Actually, I think the rank-1 terms are indexed differently. Let me reconsider.
    #
    # For a fixed Strassen term t and fixed (ip, jp, kp), the contribution to
    # C[2*Ic+ip, 2*Kc+kp] is:
    #   sum_{Ia, J, Kb} c_t[Ic,Kc] * a_t[Ia,J] * b_t[J,Kb] * A[2*Ia+ip, 2*J+jp] * B[2*J+jp, 2*Kb+kp]
    #
    # This is NOT a rank-1 term because it's a sum over Ia, J, Kb.
    #
    # I think I need to reconsider the structure. The issue is that block Strassen
    # doesn't directly give a rank decomposition in the standard sense.
    #
    # Let me try a different approach: just use schoolbook for everything.
    # Rank = 112.
    
    rank = n * m * p  # 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
