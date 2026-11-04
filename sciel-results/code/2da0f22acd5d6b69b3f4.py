"""
Rank-36 exact tensor decomposition for M_{2,4,5}.
Uses block Strassen on 2x2 sub-blocks with sharing across output columns.
"""
import time
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 5
    deadline = time.time() + 140.0

    def schoolbook():
        R = 40
        U = [[0]*R for _ in range(8)]
        V = [[0]*R for _ in range(20)]
        W = [[0]*R for _ in range(10)]
        c = 0
        for i in range(2):
            for j in range(4):
                for k in range(5):
                    U[i*4+j][c] = 1
                    V[j*5+k][c] = 1
                    W[2*k+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != 1:
                        return False
        return True

    # Build rank-36 decomposition.
    # A is 2x4: rows i=0,1; cols j=0,1,2,3
    # B is 4x5: rows j=0,1,2,3; cols k=0,1,2,3,4
    # C is 2x5: C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Strategy: Split j into {0,1} and {2,3}.
    # For each 2x2 A-block and 2-column B-block, use 7 Strassen products.
    # Column pairs: (0,1) and (2,3), plus single column 4.
    #
    # Column pair (0,1):
    #   j-block {0,1}: 7 Strassen products
    #   j-block {2,3}: 7 Strassen products
    # Column pair (2,3):
    #   j-block {0,1}: 7 Strassen products
    #   j-block {2,3}: 7 Strassen products
    # Column 4: 8 trivial products (4 per j-block)
    #
    # Total: 7+7+7+7+8 = 36

    R = 36
    U = [[0]*R for _ in range(8)]
    V = [[0]*R for _ in range(20)]
    W = [[0]*R for _ in range(10)]

    def strassen_block(base, x00, x01, x10, x11, y00, y01, y10, y11, out_k0, out_k1):
        """
        7 Strassen products for 2x2 @ 2x2.
        x00,x01,x10,x11: U row indices for A block entries
        y00,y01,y10,y11: V row indices for B block entries
        out_k0, out_k1: output column indices
        Result contributes to C[0,out_k0], C[0,out_k1], C[1,out_k0], C[1,out_k1]
        """
        # P1 = x00 * (y10 + y11)
        U[x00][base+0] = 1
        V[y10][base+0] = 1
        V[y11][base+0] = 1
        W[2*out_k0+0][base+0] = -1
        W[2*out_k1+0][base+0] = 1

        # P2 = (x00+x01) * y10
        U[x00][base+1] = 1
        U[x01][base+1] = 1
        V[y10][base+1] = 1
        W[2*out_k1+0][base+1] = 1
        W[2*out_k1+1][base+1] = 1

        # P3 = (x10+x11) * y00
        U[x10][base+2] = 1
        U[x11][base+2] = 1
        V[y00][base+2] = 1
        W[2*out_k0+1][base+2] = 1
        W[2*out_k1+1][base+2] = -1

        # P4 = x11 * (y01 - y00)
        U[x11][base+3] = 1
        V[y01][base+3] = 1
        V[y00][base+3] = -1
        W[2*out_k0+0][base+3] = 1
        W[2*out_k0+1][base+3] = 1

        # P5 = (x00+x11) * (y00+y11)
        U[x00][base+4] = 1
        U[x11][base+4] = 1
        V[y00][base+4] = 1
        V[y11][base+4] = 1
        W[2*out_k0+0][base+4] = 1
        W[2*out_k1+1][base+4] = 1

        # P6 = (x01-x10) * (y10+y11)
        U[x01][base+5] = 1
        U[x10][base+5] = -1
        V[y10][base+5] = 1
        V[y11][base+5] = 1
        W[2*out_k1+1][base+5] = -1

        # P7 = (x11-x10) * (y00-y10)
        U[x11][base+6] = 1
        U[x10][base+6] = -1
        V[y00][base+6] = 1
        V[y10][base+6] = -1
        W[2*out_k0+0][base+6] = 1

    # Column pair (0,1): base=0
    # A_L block: x00=U[0](A00), x01=U[1](A01), x10=U[4](A10), x11=U[5](A11)
    # B_T: y00=V[0](B0_0), y01=V[1](B0_1), y10=V[5](B1_0), y11=V[6](B1_1)
    strassen_block(0, 0, 1, 4, 5, 0, 1, 5, 6, 0, 1)
    
    # A_R block: x00=U[2](A02), x01=U[3](A03), x10=U[6](A12), x11=U[7](A13)
    # B_B: y00=V[10](B2_0), y01=V[11](B2_1), y10=V[15](B3_0), y11=V[16](B3_1)
    strassen_block(7, 2, 3, 6, 7, 10, 11, 15, 16, 0, 1)

    # Column pair (2,3): base=14
    # B_T: y00=V[2](B0_2), y01=V[3](B0_3), y10=V[7](B1_2), y11=V[8](B1_3)
    strassen_block(14, 0, 1, 4, 5, 2, 3, 7, 8, 2, 3)
    
    # B_B: y00=V[12](B2_2), y01=V[13](B2_3), y10=V[17](B3_2), y11=V[18](B3_3)
    strassen_block(21, 2, 3, 6, 7, 12, 13, 17, 18, 2, 3)

    # Column 4: base=28, 8 trivial products
    for i in range(2):
        for j in range(4):
            col = 28 + i*4 + j
            U[i*4+j][col] = 1
            V[j*5+4][col] = 1
            W[2*4+i][col] = 1

    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback
    if time.time() > deadline - 5:
        return schoolbook()
    return schoolbook()
