"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook decomposition of rank n*m*p
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_school

    # Try to improve using Strassen-like decomposition for the 2x2 part
    # The idea: decompose 2x2 x 2x13 as (2x2 x 2x2) + (2x2 x 2x11)
    # Use rank-7 for the 2x2x2 part and schoolbook for the rest
    # Total rank = 7 + 4*(p-2) = 7 + 44 = 51 < 52

    if p >= 2:
        strassen_rank = 7
        schoolbook_cols = p - 2
        schoolbook_rank = 4 * schoolbook_cols
        total_rank = strassen_rank + schoolbook_rank

        if total_rank < best_rank:
            U2 = [[0] * total_rank for _ in range(n * m)]  # 4 rows
            V2 = [[0] * total_rank for _ in range(m * p)]  # 26 rows
            W2 = [[0] * total_rank for _ in range(p * n)]  # 26 rows

            col = 0
            # Strassen decomposition for columns 0,1 of B
            # A = [[a,b],[c,d]], B_cols01 = [[e,f],[g,h]], C_cols01 = [[ae+bg, af+bh],[ce+dg, cf+dh]]
            #
            # V index mapping for cols 0,1: V[j*p+k]
            #   V[0*13+0]=e (j=0,k=0), V[1*13+0]=g (j=1,k=0)
            #   V[0*13+1]=f (j=0,k=1), V[1*13+1]=h (j=1,k=1)
            #
            # W index mapping: W[k*n+i]
            #   W[0*2+0]=C00 (k=0,i=0), W[1*2+0]=C10 (k=1,i=0)
            #   W[0*2+1]=C01 (k=0,i=1), W[1*2+1]=C11 (k=1,i=1)
            #
            # U index: 0=a, 1=b, 2=c, 3=d

            # Standard Strassen terms for 2x2 x 2x2:
            # We need 7 rank-1 tensors (u, v, w) such that
            # sum_t u_t[i,j] * v_t[j',k] * w_t[k',i'] = delta_{i,k'} * delta_{j,j'}
            #
            # Using the standard Strassen decomposition:
            # M1 = (a+b)(e+f)
            # M2 = (c+d)e
            # M3 = a(f-h)  -- wait, let me use a verified set
            #
            # Let me use a known correct Strassen decomposition:
            # The 7 products are:
            # p1 = (a+b)(e+f) = ae+af+be+bf
            # p2 = (c+d)e = ce+de
            # p3 = a(h-f) = ah-bf
            # p4 = d(g-e) = dg-eg
            # p5 = (a+d)(e+h) = ae+ah+de+dh
            # p6 = (b+c)(g-e) = bg-be+cg-ce
            # p7 = (b+d)(f-h) = bf-bh+df-dh
            #
            # C00 = p1 - p3 - p4 + p7
            # C01 = p3 + p5
            # C10 = p2 + p4
            # C11 = p1 - p2 + p6 - p7
            #
            # Let's verify:
            # C00 = (ae+af+be+bf) - (ah-bf) - (dg-eg) + (bf-bh+df-dh)
            #      = ae+af+be+bf - ah+bf - dg+eg + bf-bh+df-dh
            #      = ae + af + be + 3bf - ah - dg + eg - bh + df - dh
            # That's not right. Let me use a different verified decomposition.

            # Actually, let me just verify the standard Strassen formulas directly.
            # The standard Strassen for C = AB where A,B are 2x2:
            # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
            # C = [[c11, c12], [c21, c22]]
            #
            # M1 = (a11 + a22)(b11 + b22)
            # M2 = (a21 + a22)b11
            # M3 = a11(b12 - b22)
            # M4 = a22(b21 - b11)
            # M5 = (a11 + a12)b22
            # M6 = (a21 - a11)(b11 + b12)
            # M7 = (a12 - a22)(b21 + b22)
            #
            # c11 = M1 + M4 - M5 + M7
            # c12 = M3 + M5
            # c21 = M2 + M4
            # c22 = M1 - M2 + M3 - M6
            #
            # Let me verify c11:
            # M1 = (a11+a22)(b11+b22) = a11b11 + a11b22 + a22b11 + a22b22
            # M4 = a22(b21-b11) = a22b21 - a22b11
            # M5 = (a11+a12)b22 = a11b22 + a12b22
            # M7 = (a12-a22)(b21+b22) = a12b21 + a12b22 - a22b21 - a22b22
            # c11 = M1 + M4 - M5 + M7
            #      = a11b11 + a11b22 + a22b11 + a22b22 + a22b21 - a22b11 - a11b22 - a12b22 + a12b21 + a12b22 - a22b21 - a22b22
            #      = a11b11 + (a11b22 - a11b22) + (a22b11 - a22b11) + (a22b22 - a22b22) + (a22b21 - a22b21) + (-a12b22 + a12b22) + a12b21
            #      = a11b11 + a12b21
            # That's correct! c11 = a11b11 + a12b21

            # Let me verify c12:
            # c12 = M3 + M5
            # M3 = a11(b12-b22) = a11b12 - a11b22
            # M5 = (a11+a12)b22 = a11b22 + a12b22
            # c12 = a11b12 - a11b22 + a11b22 + a12b22 = a11b12 + a12b22
            # Correct!

            # Let me verify c21:
            # c21 = M2 + M4
            # M2 = (a21+a22)b11 = a21b11 + a22b11
            # M4 = a22(b21-b11) = a22b21 - a22b11
            # c21 = a21b11 + a22b11 + a22b21 - a22b11 = a21b11 + a22b21
            # Correct!

            # Let me verify c22:
            # c22 = M1 - M2 + M3 - M6
            # M1 = a11b11 + a11b22 + a22b11 + a22b22
            # M2 = a21b11 + a22b11
            # M3 = a11b12 - a11b22
            # M6 = (a21-a11)(b11+b12) = a21b11 + a21b12 - a11b11 - a11b12
            # c22 = a11b11 + a11b22 + a22b11 + a22b22 - a21b11 - a22b11 + a11b12 - a11b22 - a21b11 - a21b12 + a11b11 + a11b12
            #     = 2a11b11 + a11b22 - a11b22 + a22b11 - a22b11 + a22b22 - 2a21b11 + 2a11b12 - a21b12
            #     = 2a11b11 + a22b22 - 2a21b11 + 2a11b12 - a21b12
            # That doesn't look right. Let me recheck.
            #
            # c22 should be a21b12 + a22b22
            #
            # M1 - M2 + M3 - M6:
            # = (a11b11 + a11b22 + a22b11 + a22b22) - (a21b11 + a22b11) + (a11b12 - a11b22) - (a21b11 + a21b12 - a11b11 - a11b12)
            # = a11b11 + a11b22 + a22b11 + a22b22 - a21b11 - a22b11 + a11b12 - a11b22 - a21b11 - a21b12 + a11b11 + a11b12
            # = 2a11b11 + a11b22 - a11b22 + a22b11 - a22b11 + a22b22 - 2a21b11 + 2a11b12 - a21b12
            # = 2a11b11 + a22b22 - 2a21b11 + 2a11b12 - a21b12
            #
            # This is not equal to a21b12 + a22b22. So my Strassen formulas must be wrong.

            # Let me look up the correct Strassen formulas.
            # The standard Strassen algorithm:
            # M1 = (a11 + a22)(b11 + b22)
            # M2 = (a21 + a22)(b11)
            # M3 = (a11)(b12 - b22)
            # M4 = (a22)(b21 - b11)
            # M5 = (a11 + a12)(b22)
            # M6 = (a21 - a11)(b11 + b12)
            # M7 = (a12 - a22)(b21 + b22)
            #
            # c11 = M1 + M4 - M5 + M7
            # c12 = M3 + M5
            # c21 = M2 + M4
            # c22 = M1 - M2 + M3 - M6
            #
            # I already verified c11, c12, c21 are correct. Let me recheck c22 more carefully.
            #
            # c22 = M1 - M2 + M3 - M6
            # M1 = a11b11 + a11b22 + a22b11 + a22b22
            # M2 = a21b11 + a22b11
            # M3 = a11b12 - a11b22
            # M6 = a21b11 + a21b12 - a11b11 - a11b12
            #
            # c22 = (a11b11 + a11b22 + a22b11 + a22b22) - (a21b11 + a22b11) + (a11b12 - a11b22) - (a21b11 + a21b12 - a11b11 - a11b12)
            #
            # Let me group terms:
            # a11b11: +1 - 0 + 0 - (-1) = +1 + 1 = +2... wait
            # M6 = a21b11 + a21b12 - a11b11 - a11b12, so -M6 = -a21b11 - a21b12 + a11b11 + a11b12
            #
            # c22 = a11b11 + a11b22 + a22b11 + a22b22 - a21b11 - a22b11 + a11b12 - a11b22 - a21b11 - a21b12 + a11b11 + a11b12
            #
            # a11b11: 1 + 1 = 2
            # a11b22: 1 - 1 = 0
            # a22b11: 1 - 1 = 0
            # a22b22: 1
            # a21b11: -1 - 1 = -2
            # a11b12: 1 + 1 = 2
            # a21b12: -1
            #
            # c22 = 2a11b11 + a22b22 - 2a21b11 + 2a11b12 - a21b12
            #
            # This is definitely not a21b12 + a22b22. So the standard Strassen formulas I'm using are incorrect.

            # Let me use a different approach. I'll just use the fact that the tensor rank of 2x2 x 2x2 is 7
            # and construct the decomposition directly by solving for it.

            # Actually, for safety, let me just return the schoolbook decomposition.
            # The Strassen approach is complex and error-prone.

            pass

    return best_result
# EVOLVE-BLOCK-END
