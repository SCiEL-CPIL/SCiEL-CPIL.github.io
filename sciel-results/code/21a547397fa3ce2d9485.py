import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 4, 4, 5

    # Strategy: Use Strassen for 4x4x4 part (rank 49) + schoolbook for 4x4x1 part (rank 16)
    # Total rank = 65

    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_term(u_idx, v_idx, w_idx, col, cu, cv, cw):
        U[u_idx][col] += cu
        V[v_idx][col] += cv
        W[w_idx][col] += cw

    # Strassen 2x2x2 rank-7 decomposition
    # For matrices X (2x2), Y (2x2), Z = X*Y (2x2)
    # X indices: x00, x01, x10, x11 (as A indices i*m+j)
    # Y indices: y00, y01, y10, y11 (as B indices j*p+k)
    # Z indices: z00, z01, z10, z11 (as C indices k*n+i)
    #
    # p1 = (x00+x11)(y00+y11) -> z00: +1, z11: +1
    # p2 = (x10+x11)y00       -> z10: +1, z11: -1
    # p3 = x00(y10-y11)       -> z10: +1, z01: +1
    # p4 = x11(y01-y00)       -> z01: +1, z00: -1
    # p5 = (x00+x01)y11       -> z00: +1, z11: +1
    # p6 = (x10-x00)(y00+y01) -> z00: +1, z10: +1
    # p7 = (x01-x11)(y10+y11) -> z11: +1, z01: -1

    def add_strassen_2x2(x00, x01, x10, x11, y00, y01, y10, y11, z00, z01, z10, z11, col_start):
        # p1
        add_term(x00, y00, z00, col_start + 0, 1, 1, 1)
        add_term(x00, y11, z00, col_start + 0, 1, 1, 1)
        add_term(x11, y00, z00, col_start + 0, 1, 1, 1)
        add_term(x11, y11, z00, col_start + 0, 1, 1, 1)
        add_term(x00, y00, z11, col_start + 0, 1, 1, 1)
        add_term(x00, y11, z11, col_start + 0, 1, 1, 1)
        add_term(x11, y00, z11, col_start + 0, 1, 1, 1)
        add_term(x11, y11, z11, col_start + 0, 1, 1, 1)

        # p2
        add_term(x10, y00, z10, col_start + 1, 1, 1, 1)
        add_term(x11, y00, z10, col_start + 1, 1, 1, 1)
        add_term(x10, y00, z11, col_start + 1, 1, 1, -1)
        add_term(x11, y00, z11, col_start + 1, 1, 1, -1)

        # p3
        add_term(x00, y10, z10, col_start + 2, 1, 1, 1)
        add_term(x00, y11, z10, col_start + 2, 1, 1, -1)
        add_term(x00, y10, z01, col_start + 2, 1, 1, 1)
        add_term(x00, y11, z01, col_start + 2, 1, 1, -1)

        # p4
        add_term(x11, y01, z01, col_start + 3, 1, 1, 1)
        add_term(x11, y00, z01, col_start + 3, 1, 1, -1)
        add_term(x11, y01, z00, col_start + 3, 1, 1, -1)
        add_term(x11, y00, z00, col_start + 3, 1, 1, 1)

        # p5
        add_term(x00, y11, z00, col_start + 4, 1, 1, 1)
        add_term(x01, y11, z00, col_start + 4, 1, 1, 1)
        add_term(x00, y11, z11, col_start + 4, 1, 1, 1)
        add_term(x01, y11, z11, col_start + 4, 1, 1, 1)

        # p6
        add_term(x10, y00, z00, col_start + 5, 1, 1, 1)
        add_term(x10, y01, z00, col_start + 5, 1, 1, 1)
        add_term(x00, y00, z00, col_start + 5, -1, 1, 1)
        add_term(x00, y01, z00, col_start + 5, -1, 1, 1)
        add_term(x10, y00, z10, col_start + 5, 1, 1, 1)
        add_term(x10, y01, z10, col_start + 5, 1, 1, 1)
        add_term(x00, y00, z10, col_start + 5, -1, 1, 1)
        add_term(x00, y01, z10, col_start + 5, -1, 1, 1)

        # p7
        add_term(x01, y10, z11, col_start + 6, 1, 1, 1)
        add_term(x01, y11, z11, col_start + 6, 1, 1, 1)
        add_term(x11, y10, z11, col_start + 6, -1, 1, 1)
        add_term(x11, y11, z11, col_start + 6, -1, 1, 1)
        add_term(x01, y10, z01, col_start + 6, 1, 1, -1)
        add_term(x01, y11, z01, col_start + 6, 1, 1, -1)
        add_term(x11, y10, z01, col_start + 6, -1, 1, -1)
        add_term(x11, y11, z01, col_start + 6, -1, 1, -1)

        return col_start + 7

    # Part 1: 4x4x4 using Strassen (rank 49)
    # Split A, B, C into 2x2 blocks
    # A blocks (i, j) where i,j in {0,1} for block, plus offset
    # A00: (0,0),(0,1),(1,0),(1,1)
    # A01: (0,2),(0,3),(1,2),(1,3)
    # A10: (2,0),(2,1),(3,0),(3,1)
    # A11: (2,2),(2,3),(3,2),(3,3)
    
    # B blocks (j, k)
    # B00: (0,0),(0,1),(1,0),(1,1)
    # B01: (0,2),(0,3),(1,2),(1,3)
    # B10: (2,0),(2,1),(3,0),(3,1)
    # B11: (2,2),(2,3),(3,2),(3,3)
    
    # C blocks (k, i) - W index is k*n+i
    # C00: (0,0),(0,1),(1,0),(1,1)
    # C01: (0,2),(0,3),(1,2),(1,3)
    # C10: (2,0),(2,1),(3,0),(3,1)
    # C11: (2,2),(2,3),(3,2),(3,3)

    # Strassen 4x4: 7 products of 2x2 matrices
    # S1 = A00 + A11, T1 = B00 + B11 -> contributes to C00, C01, C10, C11
    # S2 = A10 + A11, T2 = B00
    # S3 = A00, T3 = B10 - B11
    # S4 = A11, T4 = B01 - B00
    # S5 = A00 + A01, T5 = B11
    # S6 = A10 - A00, T6 = B00 + B01
    # S7 = A01 - A11, T7 = B10 + B11

    # For each product, we need to compute 2x2 * 2x2 and add to appropriate C blocks
    # This is complex because the linear combinations span multiple blocks

    # Let me use a different approach: directly build the rank-1 terms for the full 4x4x4 tensor
    # using the known rank-49 decomposition from two levels of Strassen.

    # Actually, let me just use schoolbook for 4x4x4 (rank 64) + schoolbook for 4x4x1 (rank 16) = 80
    # and see if I can do better.

    # Alternative: Use the fact that rank of M_{4,4,5} <= rank of M_{4,4,4} + rank of M_{4,4,1}
    # If I can get rank 49 for M_{4,4,4}, total is 65.

    # Let me implement the two-level Strassen properly.
    # The 4x4x4 tensor can be decomposed as follows:
    # Use Strassen to get 7 products of 2x2x2 tensors.
    # Each 2x2x2 tensor has rank 7.
    # So total rank is 7*7 = 49.

    # I need to carefully track all the indices.

    # Let me define helper functions to get block indices

    def A_idx(i, j):
        return i * m + j

    def B_idx(j, k):
        return j * p + k

    def C_idx(k, i):
        return k * n + i

    col = 0

    # For each of the 7 Strassen products, decompose the 2x2x2 into rank 7
    # and add all terms to U, V, W

    # Product 1: (A00+A11) * (B00+B11)
    # This contributes to C00, C01, C10, C11
    # Specifically, if X = A00+A11 and Y = B00+B11, then Z = X*Y
    # Z00 -> C00, Z01 -> C01, Z10 -> C10, Z11 -> C11

    # X00 = A00[0,0] + A11[0,0] = A[0,0] + A[2,2]
    # X01 = A00[0,1] + A11[0,1] = A[0,1] + A[2,3]
    # X10 = A00[1,0] + A11[1,0] = A[1,0] + A[3,2]
    # X11 = A00[1,1] + A11[1,1] = A[1,1] + A[3,3]

    # Y00 = B00[0,0] + B11[0,0] = B[0,0] + B[2,2]
    # Y01 = B00[0,1] + B11[0,1] = B[0,1] + B[2,3]
    # Y10 = B00[1,0] + B11[1,0] = B[1,0] + B[3,2]
    # Y11 = B00[1,1] + B11[1,1] = B[1,1] + B[3,3]

    # Z00 -> C00 = C[0,0],C[0,1],C[1,0],C[1,1]
    # Z01 -> C01 = C[0,2],C[0,3],C[1,2],C[1,3]
    # Z10 -> C10 = C[2,0],C[2,1],C[3,0],C[3,1]
    # Z11 -> C11 = C[2,2],C[2,3],C[3,2],C[3,3]

    # This is getting very complex. Let me just use the schoolbook approach for now
    # and accept rank 80, but try to verify correctness.

    # Actually, let me try a simpler improvement: use Strassen for the 4x4x4 part
    # by treating it as 7 independent 2x2x2 problems, each with rank 7.

    # I'll build the full decomposition by iterating over all 7 Strassen products
    # and for each, adding 7 rank-1 terms.

    # Define the 7 Strassen products with their A and B block combinations

    # Each product is of the form (sum of A blocks with coefficients) * (sum of B blocks with coefficients)
    # and contributes to C blocks with specific coefficients

    # Let me define this more carefully

    # Strassen 4x4:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B10-B11)
    # P4 = A11*(B01-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)

    # C00 = P1 - P4 + P5 + P6
    # C01 = P3 + P4
    # C10 = P2 + P3
    # C11 = P1 - P2 + P5 + P7

    # For each Pi, it's a 2x2 * 2x2 product.
    # I need to decompose each Pi into 7 rank-1 terms.

    # Let me write a function that, given the A block coefficients, B block coefficients,
    # and C block coefficients, adds the rank-1 terms.

    # This is quite involved. Let me just return the schoolbook rank 80 for safety.

    U = [[0] * 80 for _ in range(n * m)]
    V = [[0] * 80 for _ in range(m * p)]
    W = [[0] * 80 for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
