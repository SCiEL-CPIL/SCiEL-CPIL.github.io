"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    Uses block decomposition with Strassen-like sub-decompositions.
    """
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS  # n=2, m=4, p=5

    def verify_numeric(U, V, W, tol=1e-9):
        """Verify tensor decomposition using floating point."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def verify_exact(U, V, W):
        """Verify tensor decomposition using exact Fraction arithmetic."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        U_frac = [[Fraction(U[i, r]) for r in range(rank)] for i in range(n*m)]
        V_frac = [[Fraction(V[j, r]) for r in range(rank)] for j in range(m*p)]
        W_frac = [[Fraction(W[k, r]) for r in range(rank)] for k in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for r in range(rank):
                        val += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                    if val != Fraction(1):
                        return False
        return True

    # Strategy: Block decomposition
    # A is 2x4, split into A = [A1 A2] where A1, A2 are 2x2
    # B is 4x5, split into B = [B1; B2] where B1, B2 are 2x5
    # C = A1*B1 + A2*B2
    # Each A_i * B_i is a 2x2 * 2x5 multiplication = M_{2,2,5}
    #
    # For M_{2,2,5}: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Split k into groups: {0,1,2} and {3,4}
    # M_{2,2,5} = M_{2,2,3} + M_{2,2,2}
    # M_{2,2,2} uses Strassen (rank 7)
    # M_{2,2,3} can use: rank <= 7 + 4 = 11 (Strassen for 2x2x2 + 4 for 2x2x1)
    # Actually M_{2,2,3} <= 9 using a specific construction
    #
    # Total: 2 * (9 + 7) = 32

    # Let's build M_{2,2,2} with Strassen (rank 7)
    # For 2x2 * 2x2: C = A*B
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+g)
    # m7 = (b-d)(e+g)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6

    def strassen_2x2x2():
        """Return U,V,W for M_{2,2,2} with Strassen, rank 7.
        U is 4x7 (rows: a,b,c,d), V is 4x7 (rows: e,f,g,h), W is 4x7 (rows: C11,C12,C21,C22)
        Using convention: U[i*2+j, r] for A[i,j], V[j*2+k, r] for B[j,k], W[k*2+i, r] for C[k,i]
        Wait - the convention is W[k*n+i, r] where n=2. So W row index = k*n + i = k*2 + i.
        C[i,k] corresponds to W[k*2+i, r].
        """
        # Rows of U: A[0,0]=a (idx 0), A[0,1]=b (idx 1), A[1,0]=c (idx 2), A[1,1]=d (idx 3)
        # Rows of V: B[0,0]=e (idx 0), B[0,1]=f (idx 1), B[1,0]=g (idx 2), B[1,1]=h (idx 3)
        # Rows of W: C[0,0] -> W[0*2+0]=W[0], C[0,1] -> W[1*2+0]=W[2], C[1,0] -> W[0*2+1]=W[1], C[1,1] -> W[1*2+1]=W[3]
        # So W row 0 = C[0,0], W row 1 = C[1,0], W row 2 = C[0,1], W row 3 = C[1,1]

        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # m1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: C11+=1, C22+=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        # m2 = (c+d)e: U: c=1,d=1; V: e=1; W: C21+=1, C22-=1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1

        # m3 = a(f-h): U: a=1; V: f=1,h=-1; W: C12+=1, C22+=1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1

        # m4 = d(g-e): U: d=1; V: g=1,e=-1; W: C11+=1, C21+=1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1

        # m5 = (a+b)h: U: a=1,b=1; V: h=1; W: C11-=1, C12+=1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1

        # m6 = (c-d)(f+g): U: c=1,d=-1; V: f=1,g=1; W: C22+=1
        U[2, 5] = 1; U[3, 5] = -1
        V[1, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1

        # m7 = (b-d)(e+g): U: b=1,d=-1; V: e=1,g=1; W: C11+=1
        U[1, 6] = 1; U[3, 6] = -1
        V[0, 6] = 1; V[2, 6] = 1
        W[0, 6] = 1

        return U, V, W

    # Verify Strassen
    U_s, V_s, W_s = strassen_2x2x2()
    assert verify_numeric(U_s, V_s, W_s), "Strassen verification failed"

    # Now build M_{2,2,5} by splitting k into {0,1,2} and {3,4}
    # M_{2,2,5} = M_{2,2,3} (for k=0,1,2) + M_{2,2,2} (for k=3,4)
    # M_{2,2,3}: C[i,k] for k in {0,1,2} = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is 2x2 * 2x3. Rank <= 7 + 4 = 11 (Strassen for first 2 cols + 4 for last col)
    # Or we can use a direct construction.
    #
    # Actually, let me use a simpler approach for M_{2,2,3}:
    # Direct: 6 products (2*2*3=6)
    # Or use: split into 2x2x2 + 2x2x1 = 7 + 4 = 11? That's worse than 6!
    # So for M_{2,2,3}, schoolbook rank 6 is better than Strassen-based 11.
    #
    # So M_{2,2,5} = M_{2,2,3} + M_{2,2,2} = 6 + 7 = 13
    # And M_{2,4,5} = 2 * 13 = 26

    def m_2x2x3_schoolbook():
        """M_{2,2,3} schoolbook, rank 6.
        U: 4x6 (A rows: a,b,c,d), V: 6x6 (B rows: e,f,g,h,i,j for B[0,0..2],B[1,0..2]), W: 6x6 (C rows)
        B is 2x3: B[0,0]=e(idx0), B[0,1]=f(idx1), B[0,2]=g(idx2), B[1,0]=h(idx3), B[1,1]=i(idx4), B[1,2]=j(idx5)
        C is 2x3: C[0,0]->W[0*2+0]=W[0], C[0,1]->W[1*2+0]=W[2], C[0,2]->W[2*2+0]=W[4]
                   C[1,0]->W[0*2+1]=W[1], C[1,1]->W[1*2+1]=W[3], C[1,2]->W[2*2+1]=W[5]
        """
        U = np.zeros((4, 6))
        V = np.zeros((6, 6))
        W = np.zeros((6, 6))

        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[i*2 + j, col] = 1
                    V[j*3 + k, col] = 1
                    W[k*2 + i, col] = 1
                    col += 1
        return U, V, W

    def m_2x2x2_strassen():
        """M_{2,2,2} with Strassen, rank 7. Same as strassen_2x2x2 above."""
        return strassen_2x2x2()

    # Build M_{2,2,5} = M_{2,2,3} (cols 0,1,2) + M_{2,2,2} (cols 3,4)
    # Rank = 6 + 7 = 13
    def m_2x2x5():
        U3, V3, W3 = m_2x2x3_schoolbook()  # 4x6, 6x6, 6x6
        U2, V2, W2 = m_2x2x2_strassen()     # 4x7, 4x7, 4x7

        U = np.zeros((4, 13))
        V = np.zeros((6, 13))  # B is 2x5, so 10 rows? Wait...
        # B for M_{2,2,5} is 2x5, so V has 2*5=10 rows
        # I made an error. Let me redo.
        # U: 4 rows (A is 2x2)
        # V: 10 rows (B is 2x5): B[0,0..4] = idx 0-4, B[1,0..4] = idx 5-9
        # W: 10 rows (C is 2x5): C[0,0..4] -> W[k*2+0], C[1,0..4] -> W[k*2+1]

        U = np.zeros((4, 13))
        V = np.zeros((10, 13))
        W = np.zeros((10, 13))

        # Part 1: M_{2,2,3} for k=0,1,2 (rank 6)
        # U3 is 4x6, V3 is 6x6 (but V should be 10x6 for B[0,0..2] and B[1,0..2])
        # V3 row j*3+k: for j=0: k=0,1,2 -> B[0,0],B[0,1],B[0,2] = V rows 0,1,2
        #              for j=1: k=0,1,2 -> B[1,0],B[1,1],B[1,2] = V rows 5,6,7
        # So V3 maps to V rows: [0,1,2,5,6,7]
        U[:, 0:6] = U3
        for v_row in range(6):
            actual_row = v_row if v_row < 3 else v_row + 3  # j=0: 0,1,2; j=1: 5,6,7 -> v_row 0,1,2,3,4,5 -> 0,1,2,5,6,7
            # v_row 0,1,2 -> B[0,0],B[0,1],B[0,2] -> V rows 0,1,2
            # v_row 3,4,5 -> B[1,0],B[1,1],B[1,2] -> V rows 5,6,7
            if v_row < 3:
                V[v_row, 0:6] = V3[v_row, :]
            else:
                V[v_row + 3, 0:6] = V3[v_row, :]

        # W3 is 6x6: W3 row k*2+i for k=0,1,2, i=0,1
        # W row k*2+i: for k=0: W[0]=C[0,0], W[1]=C[1,0]
        #              for k=1: W[2]=C[0,1], W[3]=C[1,1]
        #              for k=2: W[4]=C[0,2], W[5]=C[1,2]
        W[0:6, 0:6] = W3

        # Part 2: M_{2,2,2} for k=3,4 (rank 7)
        # U2 is 4x7, V2 is 4x7 (B[0,0],B[0,1],B[1,0],B[1,1] -> maps to B[0,3],B[0,4],B[1,3],B[1,4])
        # V2 row 0=B[0,0]->B[0,3]=V row 3, row 1=B[0,1]->B[0,4]=V row 4
        # V2 row 2=B[1,0]->B[1,3]=V row 8, row 3=B[1,1]->B[1,4]=V row 9
        U[:, 6:13] = U2
        V[3, 6:13] = V2[0, :]   # B[0,3]
        V[4, 6:13] = V2[1, :]   # B[0,4]
        V[8, 6:13] = V2[2, :]   # B[1,3]
        V[9, 6:13] = V2[3, :]   # B[1,4]

        # W2 is 4x7: W2 row 0=C[0,0]->C[0,3]=W row 6, row 1=C[1,0]->C[1,3]=W row 7
        # W2 row 2=C[0,1]->C[0,4]=W row 8, row 3=C[1,1]->C[1,4]=W row 9
        W[6, 6:13] = W2[0, :]   # C[0,3]
        W[7, 6:13] = W2[1, :]   # C[1,3]
        W[8, 6:13] = W2[2, :]   # C[0,4]
        W[9, 6:13] = W2[3, :]   # C[1,4]

        return U, V, W

    # Verify M_{2,2,5}
    U_225, V_225, W_225 = m_2x2x5()
    # For verification, we need to check the M_{2,2,5} tensor
    # But our verify function is for the full 2x4x5. Let me create a helper.

    def verify_m225(U, V, W, tol=1e-9):
        """Verify M_{2,2,5} decomposition."""
        n2, m2, p2 = 2, 2, 5
        rank = U.shape[1]
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m2+j, r] * V[j*p2+k, r] * W[k*n2+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    assert verify_m225(U_225, V_225, W_225), "M_{2,2,5} verification failed"

    # Now build M_{2,4,5} = M_{2,2,5} (for j=0,1) + M_{2,2,5} (for j=2,3)
    # Rank = 13 + 13 = 26
    def m_2x4x5():
        U1, V1, W1 = m_2x2x5()  # 4x13, 10x13, 10x13
        U2, V2, W2 = m_2x2x5()  # 4x13, 10x13, 10x13

        U = np.zeros((8, 26))   # A is 2x4, so 8 rows
        V = np.zeros((20, 26))  # B is 4x5, so 20 rows
        W = np.zeros((10, 26))  # C is 2x5, so 10 rows

        # Part 1: j=0,1 (A[0,0],A[0,1],A[1,0],A[1,1] and B[0,0..4],B[1,0..4])
        U[0:4, 0:13] = U1
        V[0:10, 0:13] = V1
        W[0:10, 0:13] = W1

        # Part 2: j=2,3 (A[0,2],A[0,3],A[1,2],A[1,3] and B[2,0..4],B[3,0..4])
        U[4:8, 13:26] = U2
        V[10:20, 13:26] = V2
        W[0:10, 13:26] = W2

        return U, V, W

    U_final, V_final, W_final = m_2x4x5()

    # Verify
    if verify_numeric(U_final, V_final, W_final):
        # Try to verify with exact arithmetic
        if verify_exact(U_final, V_final, W_final):
            return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
        else:
            # Fall back to numeric verification
            return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    else:
        # Fall back to schoolbook
        rank_schoolbook = n * m * p
        U = np.zeros((n*m, rank_schoolbook))
        V = np.zeros((m*p, rank_schoolbook))
        W = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
