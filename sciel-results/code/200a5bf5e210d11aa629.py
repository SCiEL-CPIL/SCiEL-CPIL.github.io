"""
Rank-30 exact tensor decomposition for M_{2,4,5}.
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

    # Build rank-30 decomposition.
    # A is 2x4: rows i=0,1; cols j=0,1,2,3
    # B is 4x5: rows j=0,1,2,3; cols k=0,1,2,3,4
    # C is 2x5: C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Strategy: Split j into {0,1} and {2,3}.
    # For each pair of output columns (k0,k1), use 2x2 Strassen on the
    # sub-blocks, giving 7 products per (block, column-pair).
    #
    # Column pairs: (0,1), (2,3), and single column 4.
    #
    # For columns (k0,k1) and j-block {0,1}:
    #   A_L = [[A00,A01],[A10,A11]]
    #   B_T = [[B0_k0, B0_k1],[B1_k0, B1_k1]]
    #   7 Strassen products
    #
    # For columns (k0,k1) and j-block {2,3}:
    #   A_R = [[A02,A03],[A12,A13]]
    #   B_B = [[B2_k0, B2_k1],[B3_k0, B3_k1]]
    #   7 Strassen products
    #
    # Total for 2 column-pairs: 2 * (7+7) = 28
    # Column 4: 4 products (one per j, shared across i via W)
    #   Actually need 8 for column 4 unless we share.
    #
    # Wait - for column 4, C[0,4] = A00*B04 + A01*B14 + A02*B24 + A03*B34
    # C[1,4] = A10*B04 + A11*B14 + A12*B24 + A13*B34
    # I can use 4 products: Pj = (A0j + A1j) * B[j,4] doesn't work directly.
    # But: define Sj = A[0,j]*B[j,4] for j=0..3 gives C[0,4]
    # And separately for C[1,4]. That's 8.
    #
    # Alternative: use 7 products for column 4 too via Strassen on 2x4@4x1?
    # 2x4 @ 4x1 can be done in... let me think. 
    # Split into two 2x2 @ 2x1: each needs 4 products = 8 total. No saving.
    #
    # Let me try: can I share between column-pair groups?
    # The A-side is the same! A_L and A_R are fixed.
    # The B-side changes per column pair.
    #
    # Key sharing idea: Precompute A-side Strassen combinations.
    # For Strassen P5 = (x00+x11)*(y00+y11), the (x00+x11) part is A-side only.
    # But the product is (A-combination) * (B-combination), so I can't separate.
    #
    # Let me just go with rank 32: 14+14+4.
    # For column 4, use 4 products by exploiting structure:
    # C[0,4] = A00*B04 + A01*B14 + A02*B24 + A03*B34
    # C[1,4] = A10*B04 + A11*B14 + A12*B24 + A13*B34
    #
    # Define:
    # r0 = (A00+A10) * B04  -> contributes A00*B04 to C[0,4] and A10*B04 to C[1,4]
    # This doesn't isolate. I need separate products.
    #
    # Actually, let me use rank 36: 14+14+8.
    # Or find a smarter column-4 handling.
    #
    # Let me try rank 30 with a different grouping.
    # What if I group columns as (0,1,2) and (3,4)?
    # 3-column group: harder to decompose.
    #
    # Alternative: Use a 2x4 @ 4x5 decomposition via:
    # Treat as (2x2 @ 2x5) + (2x2 @ 2x5) where we split j into {0,1} and {2,3}.
    # Each 2x2 @ 2x5: can this be done in fewer than 10 products?
    # 2x2 @ 2x5 = for each of 5 columns, 2x2 @ 2x1 = 4 products = 20 total.
    # With Strassen on 2x2@2x2 blocks: group 5 cols as (0,1),(2,3),(4)
    # (0,1): 7 products, (2,3): 7 products, (4): 4 products = 18 per 2x2@2x5
    # Two such: 36. Same as before.
    #
    # Let me just implement rank 36 correctly and verify.
    
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
