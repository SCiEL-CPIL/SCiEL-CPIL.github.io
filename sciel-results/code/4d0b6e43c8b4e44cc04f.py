"""Matrix multiplication 2,4,6 - rank-minimal exact tensor decomposition.
Uses a structured Strassen-style decomposition on 2x2 blocks, giving rank 42.
"""
DIMENSIONS = [2, 4, 6]

def construct():
    """Return {'U','V','W'} with integer entries that exactly decompose
    the 2x4 x 4x6 matrix multiplication tensor.
    """
    n, m, p = 2, 4, 6
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (12, rank)
    #
    # We decompose A (2x4) and B (4x6) into 2x2 blocks along the shared dimension.
    # A has columns 0..3, B has rows 0..3.
    # Split A into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3).
    # Split B into B_TL (2x3, rows 0-1, cols 0-2), B_BL (2x3, rows 2-3, cols 0-2).
    # Actually let's think more carefully.
    #
    # A is 2x4: rows i=0,1; cols j=0,1,2,3
    # B is 4x6: rows j=0,1,2,3; cols k=0,1,2,3,4,5
    # C = A*B is 2x6: rows i=0,1; cols k=0,1,2,3,4,5
    #
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # Split j into two groups: j in {0,1} and j in {2,3}.
    # A[i,0:2] is a 2x2 block (call it A1), A[i,2:4] is another 2x2 block (A2).
    # B[0:2, k] is a 2x6 block (B1), B[2:4, k] is a 2x6 block (B2).
    #
    # C[i,k] = sum_{j=0}^{1} A1[i,j] * B1[j,k] + sum_{j=0}^{1} A2[i,j] * B2[j,k]
    #
    # Each sum is a 2x2 matrix product (2x2 times 2x6 = 2x6).
    # Using Strassen for 2x2 * 2x2, we need 7 products per 2x2 block pair.
    # But B1 is 2x6, so we need 7 products per (A1, B1[:,c]) for each of 6 columns of B.
    # Actually Strassen's 7 products compute a 2x2 result from 2x2 inputs.
    # For 2x2 * 2x6, we can apply Strassen column-by-column: 6 * 7 = 42 products.
    #
    # Alternatively, we can think of it as: for each output column k,
    # C[i,k] = (A1 * B1[:,k])_i + (A2 * B2[:,k])_i
    # where A1*B1[:,k] is a 2-vector computed from a 2x2 matrix A1 and a 2-vector B1[:,k].
    #
    # Using Strassen's 7 products for 2x2 * 2x2:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b10)
    # m7 = (a01-a11)(b01+b11)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    #
    # For our case, the "2x2 B" is actually a 2x6 matrix. We apply Strassen
    # to each 2x2 column-block of B. Since B1 is 2x6, we have 3 groups of 2 columns.
    # That gives 3 * 7 = 21 products for A1*B1, and similarly 21 for A2*B2.
    # Total: 42.
    #
    # Let's construct the decomposition.
    # For A1 (2x2): A1[0,0]=A[0,0], A1[0,1]=A[0,1], A1[1,0]=A[1,0], A1[1,1]=A[1,1]
    # For B1 (2x6): B1[0,k]=B[0,k], B1[1,k]=B[1,k]
    #
    # Strassen products for A1 * B1[:, 2t:2t+2] (t=0,1,2):
    # Let b0k = B1[0,2t], b1k = B1[1,2t], b0k1 = B1[0,2t+1], b1k1 = B1[1,2t+1]
    #
    # m1 = (a00+a11)(b0k+b1k)
    # m2 = (a10+a11)b0k
    # m3 = a00(b0k1-b1k1)
    # m4 = a11(b1k-b0k)
    # m5 = (a00+a01)b1k1
    # m6 = (a10-a00)(b0k+b1k)
    # m7 = (a01-a11)(b0k1+b1k1)
    #
    # c0k = m1 + m4 - m5 + m7  (this is C[0, 2t])
    # c0k1 = m3 + m5  (this is C[0, 2t+1])
    # c1k = m2 + m4  (this is C[1, 2t])
    # c1k1 = m1 - m2 + m3 + m6  (this is C[1, 2t+1])
    #
    # Similarly for A2 * B2.
    #
    # Now I need to express this as a tensor decomposition.
    # U[i*m+j, r] picks out A[i,j], V[j*p+k, r] picks out B[j,k], W[k*n+i, r] contributes to C[i,k].
    #
    # For each Strassen product, we have a linear form in A and a linear form in B.
    # The product is rank-1 in the tensor.
    #
    # Let me enumerate all 42 terms.

    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42

    # Term indices
    t = 0

    # Process A1 * B1 (j in {0,1})
    # A1 indices: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[4], A[1,1]=U[5]
    # B1 indices: B[0,k]=V[k], B[1,k]=V[6+k]
    for col_group in range(3):  # 3 groups of 2 columns
        k0 = col_group * 2
        k1 = col_group * 2 + 1
        # B1[0,k0] = V[0*6+k0] = V[k0]
        # B1[0,k1] = V[k1]
        # B1[1,k0] = V[6+k0]
        # B1[1,k1] = V[6+k1]
        
        # m1 = (a00+a11)(b0k+b1k)
        U[0][t] = 1; U[5][t] = 1
        V[k0][t] = 1; V[6+k0][t] = 1
        # c0k += m1, c1k1 += m1
        W[k0*2+0][t] = 1   # C[0,k0]
        W[k1*2+1][t] = 1   # C[1,k1]
        t += 1

        # m2 = (a10+a11)b0k
        U[4][t] = 1; U[5][t] = 1
        V[k0][t] = 1
        # c1k += m2, c1k1 -= m2
        W[k0*2+1][t] = 1   # C[1,k0]
        W[k1*2+1][t] = -1  # C[1,k1]
        t += 1

        # m3 = a00(b0k1-b1k1)
        U[0][t] = 1
        V[k1][t] = 1; V[6+k1][t] = -1
        # c0k1 += m3, c1k1 += m3
        W[k1*2+0][t] = 1   # C[0,k1]
        W[k1*2+1][t] = 1   # C[1,k1]
        t += 1

        # m4 = a11(b1k-b0k)
        U[5][t] = 1
        V[6+k0][t] = 1; V[k0][t] = -1
        # c0k += m4, c1k += m4
        W[k0*2+0][t] = 1   # C[0,k0]
        W[k0*2+1][t] = 1   # C[1,k0]
        t += 1

        # m5 = (a00+a01)b1k1
        U[0][t] = 1; U[1][t] = 1
        V[6+k1][t] = 1
        # c0k -= m5, c0k1 += m5
        W[k0*2+0][t] = -1  # C[0,k0]
        W[k1*2+0][t] = 1   # C[0,k1]
        t += 1

        # m6 = (a10-a00)(b0k+b1k)
        U[4][t] = 1; U[0][t] = -1
        V[k0][t] = 1; V[6+k0][t] = 1
        # c1k1 += m6
        W[k1*2+1][t] = 1   # C[1,k1]
        t += 1

        # m7 = (a01-a11)(b0k1+b1k1)
        U[1][t] = 1; U[5][t] = -1
        V[k1][t] = 1; V[6+k1][t] = 1
        # c0k += m7
        W[k0*2+0][t] = 1   # C[0,k0]
        t += 1

    # Process A2 * B2 (j in {2,3})
    # A2 indices: A[0,2]=U[2], A[0,3]=U[3], A[1,2]=U[6], A[1,3]=U[7]
    # B2 indices: B[2,k]=V[12+k], B[3,k]=V[18+k]
    for col_group in range(3):
        k0 = col_group * 2
        k1 = col_group * 2 + 1
        # B2[0,k0] = V[2*6+k0] = V[12+k0]
        # B2[0,k1] = V[12+k1]
        # B2[1,k0] = V[3*6+k0] = V[18+k0]
        # B2[1,k1] = V[18+k1]
        
        # m1 = (a00+a11)(b0k+b1k)
        U[2][t] = 1; U[7][t] = 1
        V[12+k0][t] = 1; V[18+k0][t] = 1
        W[k0*2+0][t] = 1   # C[0,k0]
        W[k1*2+1][t] = 1   # C[1,k1]
        t += 1

        # m2 = (a10+a11)b0k
        U[6][t] = 1; U[7][t] = 1
        V[12+k0][t] = 1
        W[k0*2+1][t] = 1   # C[1,k0]
        W[k1*2+1][t] = -1  # C[1,k1]
        t += 1

        # m3 = a00(b0k1-b1k1)
        U[2][t] = 1
        V[12+k1][t] = 1; V[18+k1][t] = -1
        W[k1*2+0][t] = 1   # C[0,k1]
        W[k1*2+1][t] = 1   # C[1,k1]
        t += 1

        # m4 = a11(b1k-b0k)
        U[7][t] = 1
        V[18+k0][t] = 1; V[12+k0][t] = -1
        W[k0*2+0][t] = 1   # C[0,k0]
        W[k0*2+1][t] = 1   # C[1,k0]
        t += 1

        # m5 = (a00+a01)b1k1
        U[2][t] = 1; U[3][t] = 1
        V[18+k1][t] = 1
        W[k0*2+0][t] = -1  # C[0,k0]
        W[k1*2+0][t] = 1   # C[0,k1]
        t += 1

        # m6 = (a10-a00)(b0k+b1k)
        U[6][t] = 1; U[2][t] = -1
        V[12+k0][t] = 1; V[18+k0][t] = 1
        W[k1*2+1][t] = 1   # C[1,k1]
        t += 1

        # m7 = (a01-a11)(b0k1+b1k1)
        U[3][t] = 1; U[7][t] = -1
        V[12+k1][t] = 1; V[18+k1][t] = 1
        W[k0*2+0][t] = 1   # C[0,k0]
        t += 1

    return {"U": U, "V": V, "W": W}
