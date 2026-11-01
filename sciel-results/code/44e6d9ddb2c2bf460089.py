import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    # Rank-11 decomposition for 2x2 * 2x3 matrix multiplication
    # Output convention: W[k*n+i, r] where C[i,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]

    # Term 0: (A00+A11)(B00+B11) contributes to C00 and C11
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00 -> C01 -= A10*B00, C11 += A11*B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11) -> C00 += A00*B01, C10 += A00*B11 (wait, let me recheck)
    # Actually: A00*B01 contributes to C01, A00*B11 contributes to C01
    # Let me re-derive carefully.
    # C[i][k] = sum_j A[i][j] * B[j][k]
    # C[0][0] = A00*B00 + A01*B10
    # C[0][1] = A00*B01 + A01*B11
    # C[0][2] = A00*B02 + A01*B12
    # C[1][0] = A10*B00 + A11*B10
    # C[1][1] = A10*B01 + A11*B11
    # C[1][2] = A10*B02 + A11*B12

    # The existing decomposition from the seed is known to be correct (score 1.0909 = 12/11).
    # Let me keep the exact same decomposition as it's verified correct.

    # Reset and use the verified decomposition
    for i in range(n*m):
        for r in range(rank):
            U[i][r] = 0
    for i in range(m*p):
        for r in range(rank):
            V[i][r] = 0
    for i in range(p*n):
        for r in range(rank):
            W[i][r] = 0

    # 1. (A00+A11)(B00+B11) -> C00 += A00*B00 + A00*B11 + A11*B00 + A11*B11
    #    We want C00 to get A00*B00 and C11 to get A11*B11
    #    This term gives: C00: A00*B00+A00*B11+A11*B00+A11*B11, C11: same
    #    We'll correct with other terms.
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # 2. (A10+A11)B00 -> C10: A10*B00+A11*B00, C11: A10*B00+A11*B00
    #    We want C10 to get A10*B00 (has it), C11 to get A11*B00 (extra)
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # 3. A00(B01-B11) -> C01: A00*B01 - A00*B11
    #    C01 needs A00*B01 + A01*B11. This gives A00*B01 - A00*B11
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # 4. A11(B10-B00) -> C10: A11*B10 - A11*B00
    #    C10 needs A10*B00 + A11*B10. This gives A11*B10 - A11*B00
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # 5. (A00+A01)B11 -> C01: A00*B11+A01*B11, C11: A00*B11+A01*B11
    #    We want C01 to get A01*B11 (extra A00*B11), C11 to get A11*B11 (this gives A00*B11+A01*B11)
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # 6. (A10-A00)(B00+B01) -> C10: A10*B00+A10*B01-A00*B00-A00*B01
    #    C11: A10*B00+A10*B01-A00*B00-A00*B01
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # 7. (A01-A11)(B10+B11) -> C00: A01*B10+A01*B11-A11*B10-A11*B11
    #    C01: A01*B10+A01*B11-A11*B10-A11*B11
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # 8. A00*B02 -> C02: A00*B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # 9. A01*B12 -> C02: A01*B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # 10. A10*B02 -> C12: A10*B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # 11. A11*B12 -> C12: A11*B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    if time.time() > deadline:
        pass  # Return immediately

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
