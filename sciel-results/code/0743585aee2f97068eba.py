import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 100

    n, m, p = DIMENSIONS

    # Split p=7 into p1=3 and p2=4
    p1, p2 = 3, 4

    # Part 1: 3x3 * 3x3 with rank 23
    # Part 2: 3x3 * 3x4 with rank 36 (schoolbook)
    # Total rank = 59

    rank = 23 + 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # --- Part 1: 3x3x3 with 23 multiplications ---
    # Using the known 23-mult algorithm
    # A is 3x3, B is 3x3, C is 3x3
    # U indices: i*m+j = i*3+j for i,j in 0..2
    # V indices: j*p+k = j*3+k for j,k in 0..2 (first 3 columns of B)
    # W indices: k*n+i = k*3+i for k,i in 0..2

    def ai(i, j):
        return i * 3 + j

    def bi(j, k):
        return j * 3 + k  # for first 3 cols

    def ci(k, i):
        return k * 3 + i

    terms_23 = []

    # Strassen-like 7 mults for top-left 2x2 block
    # A[0,0]=a00, A[0,1]=a01, A[1,0]=a10, A[1,1]=a11
    # B[0,0]=b00, B[0,1]=b01, B[1,0]=b10, B[1,1]=b11

    # M1 = (a00+a11)(b00+b11)
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(0,0)] = 1; u[ai(1,1)] = 1
    v[bi(0,0)] = 1; v[bi(1,1)] = 1
    w[ci(0,0)] = 1; w[ci(1,1)] = 1
    terms_23.append((u, v, w))

    # M2 = (a10+a11)b00
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(1,0)] = 1; u[ai(1,1)] = 1
    v[bi(0,0)] = 1
    w[ci(1,0)] = 1
    terms_23.append((u, v, w))

    # M3 = a00(b01-b11)
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(0,0)] = 1
    v[bi(0,1)] = 1; v[bi(1,1)] = -1
    w[ci(0,1)] = 1
    terms_23.append((u, v, w))

    # M4 = a11(b10-b00)
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(1,1)] = 1
    v[bi(1,0)] = 1; v[bi(0,0)] = -1
    w[ci(1,1)] = 1
    terms_23.append((u, v, w))

    # M5 = (a00+a01)b11
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(0,0)] = 1; u[ai(0,1)] = 1
    v[bi(1,1)] = 1
    w[ci(0,0)] = 1
    terms_23.append((u, v, w))

    # M6 = (a10-a00)(b00+b01)
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(1,0)] = 1; u[ai(0,0)] = -1
    v[bi(0,0)] = 1; v[bi(0,1)] = 1
    w[ci(1,0)] = 1; w[ci(1,1)] = 1
    terms_23.append((u, v, w))

    # M7 = (a01-a11)(b10+b11)
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(0,1)] = 1; u[ai(1,1)] = -1
    v[bi(1,0)] = 1; v[bi(1,1)] = 1
    w[ci(0,1)] = 1
    terms_23.append((u, v, w))

    # Now we need 16 more multiplications to handle the 3rd row and column
    # This is the tricky part. Let me use a known complete 23-mult algorithm.
    # Reference: Winograd's 23-mult algorithm for 3x3 matrices

    # The remaining 16 terms handle interactions with the 3rd row/col.
    # I'll use a systematic approach based on known decompositions.

    # M8 = a02*b00
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(0,2)] = 1
    v[bi(0,0)] = 1
    w[ci(0,0)] = 1
    terms_23.append((u, v, w))

    # M9 = a02*b01
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(0,2)] = 1
    v[bi(0,1)] = 1
    w[ci(0,1)] = 1
    terms_23.append((u, v, w))

    # M10 = a02*b10
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(0,2)] = 1
    v[bi(1,0)] = 1
    w[ci(0,0)] = 1
    terms_23.append((u, v, w))

    # M11 = a02*b11
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(0,2)] = 1
    v[bi(1,1)] = 1
    w[ci(0,1)] = 1
    terms_23.append((u, v, w))

    # M12 = a12*b00
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(1,2)] = 1
    v[bi(0,0)] = 1
    w[ci(1,0)] = 1
    terms_23.append((u, v, w))

    # M13 = a12*b01
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(1,2)] = 1
    v[bi(0,1)] = 1
    w[ci(1,1)] = 1
    terms_23.append((u, v, w))

    # M14 = a12*b10
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(1,2)] = 1
    v[bi(1,0)] = 1
    w[ci(1,0)] = 1
    terms_23.append((u, v, w))

    # M15 = a12*b11
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(1,2)] = 1
    v[bi(1,1)] = 1
    w[ci(1,1)] = 1
    terms_23.append((u, v, w))

    # M16 = a20*b00
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(2,0)] = 1
    v[bi(0,0)] = 1
    w[ci(0,2)] = 1
    terms_23.append((u, v, w))

    # M17 = a20*b01
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(2,0)] = 1
    v[bi(0,1)] = 1
    w[ci(1,2)] = 1
    terms_23.append((u, v, w))

    # M18 = a20*b10
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(2,0)] = 1
    v[bi(1,0)] = 1
    w[ci(0,2)] = 1
    terms_23.append((u, v, w))

    # M19 = a20*b11
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(2,0)] = 1
    v[bi(1,1)] = 1
    w[ci(1,2)] = 1
    terms_23.append((u, v, w))

    # M20 = a21*b00
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(2,1)] = 1
    v[bi(0,0)] = 1
    w[ci(0,2)] = 1
    terms_23.append((u, v, w))

    # M21 = a21*b01
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(2,1)] = 1
    v[bi(0,1)] = 1
    w[ci(1,2)] = 1
    terms_23.append((u, v, w))

    # M22 = a21*b10
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(2,1)] = 1
    v[bi(1,0)] = 1
    w[ci(0,2)] = 1
    terms_23.append((u, v, w))

    # M23 = a21*b11
    u = [0]*9; v = [0]*9; w = [0]*9
    u[ai(2,1)] = 1
    v[bi(1,1)] = 1
    w[ci(1,2)] = 1
    terms_23.append((u, v, w))

    # That's 23 terms, but I need to verify this is correct.
    # The above is NOT a valid 23-mult algorithm - it's just a partial schoolbook.
    # Let me use a proper known decomposition.

    # Actually, let me just use schoolbook for the 3x3x3 part (rank 27)
    # and schoolbook for 3x3x4 (rank 36). Total = 63.
    # That's the same as full schoolbook. No improvement.

    # To actually beat 63, I need a valid rank-23 algorithm.
    # Let me implement one properly.

    # I'll clear and use a verified approach:
    # For now, return schoolbook (rank 63) to ensure validity.
    # The improvement path requires implementing the actual 23-mult algorithm.

    rank = n * m * p
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
