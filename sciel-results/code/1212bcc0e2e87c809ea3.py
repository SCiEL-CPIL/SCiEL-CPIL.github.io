"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication
    using Strassen's algorithm applied to 2x2 blocks.
    
    The 4x4 matrices are viewed as 2x2 blocks of 2x2 matrices.
    Strassen's algorithm for 2x2 block multiplication requires 7 block products,
    each of which is a 2x2 matrix multiplication (rank 7 each).
    Total rank = 7 * 7 = 49.
    
    Output convention: W[k*n+i, r] where n=4, so output index = k*4 + i.
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 49
    
    # Initialize matrices
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Strassen's 7 products for 2x2 block multiplication
    # Each product M_s = X_s * Y_s where X_s, Y_s are linear combos of 2x2 blocks
    # For each s in 0..6, we need the 7 Strassen rank-1 terms for the 2x2 product
    
    # Strassen's 7 rank-1 terms for 2x2 matrix multiplication:
    # For multiplying P (2x2) * Q (2x2):
    # P = [[p0, p1], [p2, p3]], Q = [[q0, q1], [q2, q3]]
    # 
    # s0 = (p0 + p3)(q0 + q3)
    # s1 = (p2 + p3)q0
    # s2 = p0(q1 - q3)
    # s3 = p3(q2 - q0)
    # s4 = (p0 + p1)q3
    # s5 = (p2 - p0)(q0 + q1)
    # s6 = (p1 - p3)(q2 + q3)
    #
    # Result:
    # r00 = s0 + s3 - s4 + s6
    # r01 = s2 + s4
    # r10 = s1 + s3
    # r11 = s0 - s1 + s2 + s5
    
    # For the 4x4 case, each "2x2 block" is actually a 2x2 matrix of entries.
    # Let's define the 4x4 matrix A with entries A[i,j] for i,j in 0..3
    # Block structure:
    # A11 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]  -> indices (0,0),(0,1),(1,0),(1,1)
    # A12 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]  -> indices (0,2),(0,3),(1,2),(1,3)
    # A21 = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]]  -> indices (2,0),(2,1),(3,0),(3,1)
    # A22 = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]]  -> indices (2,2),(2,3),(3,2),(3,3)
    
    # Input index for U: i*m + j where i is row of A, j is column of A
    # So A[i,j] -> U[i*4 + j, col]
    
    # For B: V[j*p + k, col] where j is row of B, k is column of B
    # So B[j,k] -> V[j*4 + k, col]
    
    # Output: C[i,k] -> W[k*4 + i, col]
    
    # The 7 Strassen block products:
    # M0 = (A11 + A22)(B11 + B22)
    # M1 = (A21 + A22)B11
    # M2 = A11(B12 - B22)
    # M3 = A22(B21 - B11)
    # M4 = (A11 + A12)B22
    # M5 = (A21 - A11)(B11 + B12)
    # M6 = (A12 - A22)(B21 + B22)
    
    # For each block product M_s = X_s * Y_s, we apply Strassen to the 2x2 matrices X_s and Y_s.
    # X_s has entries X_s[a,b] for a,b in 0,1 (within the 2x2 block)
    # Y_s has entries Y_s[c,d] for c,d in 0,1
    
    # The 7 rank-1 terms for X_s * Y_s:
    # t0 = (X_s[0,0] + X_s[1,1])(Y_s[0,0] + Y_s[1,1])
    # t1 = (X_s[1,0] + X_s[1,1])Y_s[0,0]
    # t2 = X_s[0,0](Y_s[0,1] - Y_s[1,1])
    # t3 = X_s[1,1](Y_s[1,0] - Y_s[0,0])
    # t4 = (X_s[0,0] + X_s[0,1])Y_s[1,1]
    # t5 = (X_s[1,0] - X_s[0,0])(Y_s[0,0] + Y_s[0,1])
    # t6 = (X_s[0,1] - X_s[1,1])(Y_s[1,0] + Y_s[1,1])
    
    # Result of X_s * Y_s:
    # R[0,0] = t0 + t3 - t4 + t6
    # R[0,1] = t2 + t4
    # R[1,0] = t1 + t3
    # R[1,1] = t0 - t1 + t2 + t5
    
    # Now, X_s and Y_s are linear combinations of the 2x2 blocks of A and B.
    # Each entry of X_s is a linear combination of entries of A.
    # Each entry of Y_s is a linear combination of entries of B.
    
    # Let's define the 4 entries of each 2x2 block:
    # A11: A[0,0], A[0,1], A[1,0], A[1,1]
    # A12: A[0,2], A[0,3], A[1,2], A[1,3]
    # A21: A[2,0], A[2,1], A[3,0], A[3,1]
    # A22: A[2,2], A[2,3], A[3,2], A[3,3]
    
    # For a linear combination like A11 + A22, the entries are:
    # [A[0,0]+A[2,2], A[0,1]+A[2,3], A[1,0]+A[3,2], A[1,1]+A[3,3]]
    
    # Let's build the U and V matrices for each of the 49 terms.
    
    # For each of the 7 block products s (0-6), and each of the 7 Strassen terms t (0-6):
    # We have a rank-1 term.
    
    # First, define the linear combinations for X_s and Y_s.
    # X_s is a 2x2 matrix, Y_s is a 2x2 matrix.
    
    # Block product definitions (X_s, Y_s):
    # s=0: X = A11 + A22, Y = B11 + B22
    # s=1: X = A21 + A22, Y = B11
    # s=2: X = A11, Y = B12 - B22
    # s=3: X = A22, Y = B21 - B11
    # s=4: X = A11 + A12, Y = B22
    # s=5: X = A21 - A11, Y = B11 + B12
    # s=6: X = A12 - A22, Y = B21 + B22
    
    # For each block, we need to express the 4 entries as linear combinations of A/B entries.
    
    # Let's create a helper to get the linear combination coefficients.
    # A block is specified by which 2x2 submatrix and coefficients.
    
    # Define the 4 blocks of A:
    # Block (br, bc) where br in {0,1}, bc in {0,1}
    # A[br*2+a, bc*2+b] for a,b in {0,1}
    
    # For X_s, it's a linear combination of A blocks.
    # For Y_s, it's a linear combination of B blocks.
    
    # Let's define the coefficients for each X_s and Y_s.
    # X_s[a,b] = sum over blocks (br,bc) of coeff_X[s][br][bc] * A[br*2+a, bc*2+b]
    # Y_s[c,d] = sum over blocks (br,bc) of coeff_Y[s][br][bc] * B[br*2+c, bc*2+d]
    
    # Define coeff_X and coeff_Y as 4x4 arrays for each s (2x2 blocks, 2x2 positions within block)
    
    coeff_X = []  # coeff_X[s][a][b][br][bc]
    coeff_Y = []  # coeff_Y[s][c][d][br][bc]
    
    # s=0: X = A11 + A22
    # X[a,b] = A[0*2+a, 0*2+b] + A[1*2+a, 1*2+b]
    # So coeff_X[0][a][b][0][0] = 1, coeff_X[0][a][b][1][1] = 1
    cX0 = [[[[0,0],[0,0]], [[0,0],[0,0]]],
           [[[0,0],[0,0]], [[0,0],[0,0]]]]
    cX0[0][0][0][0] = 1; cX0[0][0][1][1] = 1
    cX0[0][1][0][0] = 1; cX0[0][1][1][1] = 1
    cX0[1][0][0][0] = 1; cX0[1][0][1][1] = 1
    cX0[1][1][0][0] = 1; cX0[1][1][1][1] = 1
    coeff_X.append(cX0)
    
    cY0 = [[[[0,0],[0,0]], [[0,0],[0,0]]],
           [[[0,0],[0,0]], [[0,0],[0,0]]]]
    cY0[0][0][0][0] = 1; cY0[0][0][1][1] = 1
    cY0[0][1][0][0] = 1; cY0[0][1][1][1] = 1
    cY0[1][0][0][0] = 1; cY0[1][0][1][1] = 1
    cY0[1][1][0][0] = 1; cY0[1][1][1][1] = 1
    coeff_Y.append(cY0)
    
    # s=1: X = A21 + A22, Y = B11
    cX1 = [[[[0,0],[0,0]], [[0,0],[0,0]]],
           [[[0,0],[0,0]], [[0,0],[0,0]]]]
    cX1[0][0][1][0] = 1; cX1[0][0][1][1] = 1
    cX1[0][1][1][0] = 1; cX1[0][1][1][1] = 1
    cX1[1][0][1][0] = 1; cX1[1][0][1][1] = 1
    cX1[1][1][1][0] = 1; cX1[1][1][1][1] = 1
    coeff_X.append(cX1)
    
    cY1 = [[[[0,0],[0,0]], [[0,0],[0,0]]],
           [[[0,0],[0,0]], [[0,0],[0,0]]]]
    cY1[0][0][0][0] = 1
    cY1[0][1][0][0] = 1
    cY1[1][0][0][0] = 1
    cY1[1][1][0][0] = 1
    coeff_Y.append(cY1)
    
    # s=2: X = A11, Y = B12 - B22
    cX2 = [[[[0,0],[0,0]], [[0,0],[0,0]]],
           [[[0,0],[0,0]], [[0,0],[0,0]]]]
    cX2[0][0][0][0] = 1
    cX2[0][1][0][0] = 1
    cX2[1][0][0][0] = 1
    cX2[1][1][0][0] = 1
    coeff_X.append(cX2)
    
    cY2 = [[[[0,0],[0,0]], [[0,0],[0,0]]],
           [[[0,0],[0,0]], [[0,0],[0,0]]]]
    cY2[0][0][0][1] = 1; cY2[0][0][1][1] = -1
    cY2[0][1][0][1] = 1; cY2[0][1][1][1] = -1
    cY2[1][0][0][1] = 1; cY2[1][0][1][1] = -1
    cY2[1][1][0][1] = 1; cY2[1][1][1][1] = -1
    coeff_Y.append(cY2)
    
    # s=3: X = A22, Y = B21 - B11
    cX3 = [[[[0,0],[0,0]], [[0,0],[0,0]]],
           [[[0,0],[0,0]], [[0,0],[0,0]]]]
    cX3[0][0][1][1] = 1
    cX3[0][1][1][1] = 1
    cX3[1][0][1][1] = 1
    cX3[1][1][1][1] = 1
    coeff_X.append(cX3)
    
    cY3 = [[[[0,0],[0,0]], [[0,0],[0,0]]],
           [[[0,0],[0,0]], [[0,0],[0,0]]]]
    cY3[0][0][1][0] = 1; cY3[0][0][0][0] = -1
    cY3[0][1][1][0] = 1; cY3[0][1][0][0] = -1
    cY3[1][0][1][0] = 1; cY3[1][0][0][0] = -1
    cY3[1][1][1][0] = 1; cY3[1][1][0][0] = -1
    coeff_Y.append(cY3)
    
    # s=4: X = A11 + A12, Y = B22
    cX4 = [[[[0,0],[0,0]], [[0,0],[0,0]]],
           [[[0,0],[0,0]], [[0,0],[0,0]]]]
    cX4[0][0][0][0] = 1; cX4[0][0][0][1] = 1
    cX4[0][1][0][0] = 1; cX4[0][1][0][1] = 1
    cX4[1][0][0][0] = 1; cX4[1][0][0][1] = 1
    cX4[1][1][0][0] = 1; cX4[1][1][0][1] = 1
    coeff_X.append(cX4)
    
    cY4 = [[[[0,0],[0,0]], [[0,0],[0,0]]],
           [[[0,0],[0,0]], [[0,0],[0,0]]]]
    cY4[0][0][1][1] = 1
    cY4[0][1][1][1] = 1
    cY4[1][0][1][1] = 1
    cY4[1][1][1][1] = 1
    coeff_Y.append(cY4)
    
    # s=5: X = A21 - A11, Y = B11 + B12
    cX5 = [[[[0,0],[0,0]], [[0,0],[0,0]]],
           [[[0,0],[0,0]], [[0,0],[0,0]]]]
    cX5[0][0][1][0] = 1; cX5[0][0][0][0] = -1
    cX5[0][1][1][0] = 1; cX5[0][1][0][0] = -1
    cX5[1][0][1][0] = 1; cX5[1][0][0][0] = -1
    cX5[1][1][1][0] = 1; cX5[1][1][0][0] = -1
    coeff_X.append(cX5)
    
    cY5 = [[[[0,0],[0,0]], [[0,0],[0,0]]],
           [[[0,0],[0,0]], [[0,0],[0,0]]]]
    cY5[0][0][0][0] = 1; cY5[0][0][0][1] = 1
    cY5[0][1][0][0] = 1; cY5[0][1][0][1] = 1
    cY5[1][0][0][0] = 1; cY5[1][0][0][1] = 1
    cY5[1][1][0][0] = 1; cY5[1][1][0][1] = 1
    coeff_Y.append(cY5)
    
    # s=6: X = A12 - A22, Y = B21 + B22
    cX6 = [[[[0,0],[0,0]], [[0,0],[0,0]]],
           [[[0,0],[0,0]], [[0,0],[0,0]]]]
    cX6[0][0][0][1] = 1; cX6[0][0][1][1] = -1
    cX6[0][1][0][1] = 1; cX6[0][1][1][1] = -1
    cX6[1][0][0][1] = 1; cX6[1][0][1][1] = -1
    cX6[1][1][0][1] = 1; cX6[1][1][1][1] = -1
    coeff_X.append(cX6)
    
    cY6 = [[[[0,0],[0,0]], [[0,0],[0,0]]],
           [[[0,0],[0,0]], [[0,0],[0,0]]]]
    cY6[0][0][1][0] = 1; cY6[0][0][1][1] = 1
    cY6[0][1][1][0] = 1; cY6[0][1][1][1] = 1
    cY6[1][0][1][0] = 1; cY6[1][0][1][1] = 1
    cY6[1][1][1][0] = 1; cY6[1][1][1][1] = 1
    coeff_Y.append(cY6)
    
    # Now, for each s (0-6) and t (0-6), we have a rank-1 term.
    # The Strassen terms for 2x2 multiplication:
    # t0 = (X[0,0] + X[1,1])(Y[0,0] + Y[1,1])
    # t1 = (X[1,0] + X[1,1])Y[0,0]
    # t2 = X[0,0](Y[0,1] - Y[1,1])
    # t3 = X[1,1](Y[1,0] - Y[0,0])
    # t4 = (X[0,0] + X[0,1])Y[1,1]
    # t5 = (X[1,0] - X[0,0])(Y[0,0] + Y[0,1])
    # t6 = (X[0,1] - X[1,1])(Y[1,0] + Y[1,1])
    
    # Each t is a product of two linear forms in the entries of X and Y.
    # We need to express each linear form as a linear combination of A/B entries.
    
    # For a given t, the X-part is a linear combination of X[a,b] entries.
    # Each X[a,b] is a linear combination of A entries via coeff_X[s].
    # Similarly for Y.
    
    # Let's define the X-part and Y-part coefficients for each t.
    # x_coeffs[t][a][b] = coefficient of X[a,b] in the X-part of term t
    # y_coeffs[t][c][d] = coefficient of Y[c,d] in the Y-part of term t
    
    x_coeffs = [
        [[1, 0], [0, 1]],  # t0: X[0,0] + X[1,1]
        [[0, 0], [1, 1]],  # t1: X[1,0] + X[1,1]
        [[1, 0], [0, 0]],  # t2: X[0,0]
        [[0, 0], [0, 1]],  # t3: X[1,1]
        [[1, 1], [0, 0]],  # t4: X[0,0] + X[0,1]
        [[-1, 0], [1, 0]], # t5: X[1,0] - X[0,0]
        [[0, 1], [0, -1]], # t6: X[0,1] - X[1,1]
    ]
    
    y_coeffs = [
        [[1, 0], [0, 1]],  # t0: Y[0,0] + Y[1,1]
        [[1, 0], [0, 0]],  # t1: Y[0,0]
        [[0, 1], [0, -1]], # t2: Y[0,1] - Y[1,1]
        [[-1, 0], [1, 0]], # t3: Y[1,0] - Y[0,0]
        [[0, 0], [0, 1]],  # t4: Y[1,1]
        [[1, 1], [0, 0]],  # t5: Y[0,0] + Y[0,1]
        [[0, 0], [1, 1]],  # t6: Y[1,0] + Y[1,1]
    ]
    
    # Now, for each s and t, the U-part is:
    # U[i*4+j, col] = sum over a,b,br,bc of x_coeffs[t][a][b] * coeff_X[s][a][b][br][bc] * delta(i == br*2+a, j == bc*2+b)
    
    # Similarly for V.
    
    # The W-part: the result of X_s * Y_s is a 2x2 matrix R_s.
    # R_s[ar, ac] = sum over t of w_coeffs[t][ar][ac] * t_s
    # where w_coeffs[t][ar][ac] is the coefficient of term t in R_s[ar,ac].
    
    # From Strassen:
    # R[0,0] = t0 + t3 - t4 + t6
    # R[0,1] = t2 + t4
    # R[1,0] = t1 + t3
    # R[1,1] = t0 - t1 + t2 + t5
    
    w_coeffs = [
        [[1, 0], [0, 0]],  # t0
        [[0, 0], [0, -1]], # t1
        [[0, 1], [0, 0]],  # t2
        [[1, 0], [0, 0]],  # t3
        [[-1, 0], [0, 0]], # t4
        [[0, 0], [0, 1]],  # t5
        [[1, 0], [0, 0]],  # t6
    ]
    # Wait, let me re-check:
    # R[0,0] = t0 + t3 - t4 + t6
    # So t0: R[0,0]+=1, t3: R[0,0]+=1, t4: R[0,0]+=-1, t6: R[0,0]+=1
    # R[0,1] = t2 + t4
    # So t2: R[0,1]+=1, t4: R[0,1]+=1
    # R[1,0] = t1 + t3
    # So t1: R[1,0]+=1, t3: R[1,0]+=1
    # R[1,1] = t0 - t1 + t2 + t5
    # So t0: R[1,1]+=1, t1: R[1,1]+=-1, t2: R[1,1]+=1, t5: R[1,1]+=1
    
    w_coeffs = [
        [[1, 0], [1, 0]],  # t0: R[0,0]+=1, R[1,1]+=1
        [[0, 0], [-1, 1]], # t1: R[1,0]+=1, R[1,1]+=-1
        [[1, 0], [0, 1]],  # t2: R[0,1]+=1, R[1,1]+=1
        [[1, 0], [1, 0]],  # t3: R[0,0]+=1, R[1,0]+=1
        [[-1, 1], [0, 0]], # t4: R[0,0]+=-1, R[0,1]+=1
        [[0, 0], [0, 1]],  # t5: R[1,1]+=1
        [[1, 0], [0, 0]],  # t6: R[0,0]+=1
    ]
    
    # Now, the block product M_s contributes to the 2x2 block C_s of the result.
    # The 4x4 result C is:
    # C11 = M0 + M3 - M4 + M6
    # C12 = M2 + M4
    # C21 = M1 + M3
    # C22 = M0 - M1 + M2 + M5
    
    # So the coefficient of M_s in C[br,bc] (where br,bc in {0,1} are block indices) is:
    # block_coeff[br][bc][s]
    
    block_coeff = [
        [  # br=0
            [1, 0, 0, 1, -1, 0, 1],  # bc=0: C11 = M0 + M3 - M4 + M6
            [0, 0, 1, 0, 1, 0, 0],   # bc=1: C12 = M2 + M4
        ],
        [  # br=1
            [0, 1, 0, 1, 0, 0, 0],   # bc=0: C21 = M1 + M3
            [1, -1, 1, 0, 0, 1, 0],  # bc=1: C22 = M0 - M1 + M2 + M5
        ]
    ]
    
    # Now, for each s, t, the contribution to W is:
    # W[k*4+i, col] = sum over ar,ac,br,bc of w_coeffs[t][ar][ac] * block_coeff[br][bc][s] * delta(i == br*2+ar, k == bc*2+ac)
    
    # Let's fill in U, V, W.
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # Fill U: for each A[i,j], determine coefficient
            for i in range(4):
                for j in range(4):
                    # i = br*2 + a, j = bc*2 + b
                    br = i // 2
                    bc = j // 2
                    a = i % 2
                    b = j % 2
                    # U[i*4+j, col] = sum over a',b' of x_coeffs[t][a'][b'] * coeff_X[s][a'][b'][br][bc]
                    # But coeff_X[s][a'][b'][br][bc] is the coefficient of A[br*2+a', bc*2+b'] in X_s[a',b']
                    # And we want the coefficient of A[i,j] = A[br*2+a, bc*2+b] in the X-part.
                    # The X-part is sum over a',b' of x_coeffs[t][a'][b'] * X_s[a',b']
                    # X_s[a',b'] = sum over br',bc' of coeff_X[s][a'][b'][br'][bc'] * A[br'*2+a', bc'*2+b']
                    # So the coefficient of A[br*2+a, bc*2+b] is:
                    # sum over a',b' of x_coeffs[t][a'][b'] * coeff_X[s][a'][b'][br][bc]
                    # But only if a' == a and b' == b? No, wait.
                    # A[br*2+a, bc*2+b] appears in X_s[a',b'] with coefficient coeff_X[s][a'][b'][br][bc]
                    # only if a' == a and b' == b? No, that's not right either.
                    # Actually, X_s[a',b'] is a linear combination of A entries.
                    # The entry A[br*2+a, bc*2+b] has a specific (br,bc) block and (a,b) position.
                    # It appears in X_s[a',b'] with coefficient coeff_X[s][a'][b'][br][bc] if a'==a and b'==b? 
                    # No, coeff_X[s][a'][b'][br][bc] is the coefficient of A[br*2+a', bc*2+b'] in X_s[a',b'].
                    # So A[br*2+a, bc*2+b] appears in X_s[a',b'] with coefficient coeff_X[s][a'][b'][br][bc] 
                    # only if a' == a and b' == b. Wait, no.
                    # A[br*2+a, bc*2+b] has block (br,bc) and position (a,b).
                    # It appears in X_s[a',b'] with coefficient coeff_X[s][a'][b'][br][bc] 
                    # if and only if a' == a and b' == b. Because X_s[a',b'] uses A[br'*2+a', bc'*2+b'] 
                    # with coefficient coeff_X[s][a'][b'][br'][bc'].
                    # So for a fixed A entry at (br,bc,a,b), it appears in X_s[a',b'] with 
                    # coefficient coeff_X[s][a'][b'][br][bc] only when a'==a and b'==b.
                    
                    # So the coefficient of A[i,j] in the X-part is:
                    # x_coeffs[t][a][b] * coeff_X[s][a][b][br][bc]
                    coeff = x_coeffs[t][a][b] * coeff_X[s][a][b][br][bc]
                    U[i * m + j][col] += coeff
            
            # Fill V: for each B[j,k], determine coefficient
            for j in range(4):
                for k in range(4):
                    br = j // 2
                    bc = k // 2
                    c = j % 2
                    d = k % 2
                    coeff = y_coeffs[t][c][d] * coeff_Y[s][c][d][br][bc]
                    V[j * p + k][col] += coeff
            
            # Fill W: for each C[i,k], determine coefficient
            for i in range(4):
                for k in range(4):
                    br = i // 2
                    bc = k // 2
                    ar = i % 2
                    ac = k % 2
                    coeff = w_coeffs[t][ar][ac] * block_coeff[br][bc][s]
                    W[k * n + i][col] += coeff
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
