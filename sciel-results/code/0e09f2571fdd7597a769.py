"""Matrix multiplication 2x3 x 3x8: rank-9 decomposition via Strassen on 2x2 block."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (16, rank)
    #
    # Split A (2x3) into A11 (2x2) and A12 (2x1).
    # Split B (3x8) into B11 (2x8) and B21 (1x8).
    # C = A11*B11 + A12*B21
    # A11*B11 uses Strassen (rank 7), A12*B21 uses schoolbook (rank 2).
    # Total rank = 9.

    rank = 9
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen indices for A11 (rows 0,1 of A; cols 0,1) * B11 (rows 0,1 of B; cols 0..7)
    # A11 row a (0,1) col c (0,1) -> U index a*3 + c
    # B11 row r (0,1) col k (0..7) -> V index r*8 + k
    # C11 row i (0,1) col k (0..7) -> W index k*2 + i

    # Strassen 7 factors. M_j = (U_A * U_B) * (V_A * V_B)
    # A11 = [[a00, a01],[a10, a11]], B11 = [[b00..b07],[b10..b17]]
    # M1 = a00 * b00 ; M2 = (a00+a01)*b10 ; M3 = a10*(b00+b10)
    # M4 = a11*(b10-b00) ; M5 = (a00+a10)*(b01+b11)
    # M6 = (a11-a10)*(b01-b11) ; M7 = (a01+a11)*(b10+b11)
    # c00 = M1+M4-M5+M7 ; c01 = M3+M5 ; c10 = M2+M4 ; c11 = M1-M2+M3+M6

    # We'll represent each Strassen factor as a pair (a_indices, b_indices) with the
    # product being sum of (A entries) * (B entries). For a given factor, U encodes A
    # coefficients, V encodes B coefficients, and W encodes how the product lands in C.

    # For a factor: U[a*3+c] += coeff_a, V[r*8+k] += coeff_b, and W[k*2+i] += 1 for
    # each output (i,k) that this factor contributes to (with the appropriate sign).

    def add_factor(a_terms, b_terms, w_terms):
        # a_terms: list of (u_index, coeff)
        # b_terms: list of (v_index, coeff)
        # w_terms: list of (w_index, coeff)
        for (ui, cu) in a_terms:
            U[ui][col] = cu
        for (vi, cv) in b_terms:
            V[vi][col] = cv
        for (wi, cw) in w_terms:
            W[wi][col] += cw

    col = 0
    # Helper: A11 entry (a,c) -> u index a*3 + c
    def ua(a, c): return a * 3 + c
    # B11 entry (r,k) -> v index r*8 + k
    def vb(r, k): return r * 8 + k
    # C11 entry (i,k) -> w index k*2 + i
    def wc(i, k): return k * 2 + i

    # M1 = a00 * b00 ; contributes to c00(+), c11(+)
    add_factor([(ua(0,0),1)], [(vb(0,0),1)], [(wc(0,0),1),(wc(1,0),1)])
    col += 1
    # M2 = (a00+a01) * b10 ; contributes to c10(+), c11(-)
    add_factor([(ua(0,0),1),(ua(0,1),1)], [(vb(1,0),1)], [(wc(1,0),1),(wc(1,0),0)])  # placeholder
    # redo M2 properly
    col -= 1
    # reset
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    # M1 = a00 * b00 ; c00 += M1, c11 += M1
    U[ua(0,0)][col] = 1
    V[vb(0,0)][col] = 1
    W[wc(0,0)][col] = 1
    W[wc(1,0)][col] = 1
    col += 1
    # M2 = (a00+a01) * b10 ; c10 += M2, c11 -= M2
    U[ua(0,0)][col] = 1
    U[ua(0,1)][col] = 1
    V[vb(1,0)][col] = 1
    W[wc(1,0)][col] = 1
    W[wc(1,0)][col] = 0  # c10
    # careful: c10 = row1 col0 -> wc(1,0); c11 = row1 col1 -> wc(1,1)
    W[wc(1,0)][col] = 1   # c10 += M2
    W[wc(1,1)][col] = -1  # c11 -= M2
    col += 1
    # M3 = a10 * (b00+b10) ; c01 += M3, c10 += M3
    U[ua(1,0)][col] = 1
    V[vb(0,0)][col] = 1
    V[vb(1,0)][col] = 1
    W[wc(0,1)][col] = 1   # c01
    W[wc(1,0)][col] = 1   # c10
    col += 1
    # M4 = a11 * (b10-b00) ; c00 += M4, c10 += M4
    U[ua(1,1)][col] = 1
    V[vb(1,0)][col] = 1
    V[vb(0,0)][col] = -1
    W[wc(0,0)][col] = 1   # c00
    W[wc(1,0)][col] = 1   # c10
    col += 1
    # M5 = (a00+a10) * (b01+b11) ; c00 -= M5, c01 += M5
    U[ua(0,0)][col] = 1
    U[ua(1,0)][col] = 1
    V[vb(0,1)][col] = 1
    V[vb(1,1)][col] = 1
    W[wc(0,0)][col] = -1  # c00
    W[wc(0,1)][col] = 1   # c01
    col += 1
    # M6 = (a11-a10) * (b01-b11) ; c11 += M6
    U[ua(1,1)][col] = 1
    U[ua(1,0)][col] = -1
    V[vb(0,1)][col] = 1
    V[vb(1,1)][col] = -1
    W[wc(1,1)][col] = 1   # c11
    col += 1
    # M7 = (a01+a11) * (b10+b11) ; c00 += M7
    U[ua(0,1)][col] = 1
    U[ua(1,1)][col] = 1
    V[vb(1,0)][col] = 1
    V[vb(1,1)][col] = 1
    W[wc(0,0)][col] = 1   # c00
    col += 1

    # Now the 2x1 block: A12 (2x1) * B21 (1x8). Schoolbook rank 2.
    # A12 row a (0,1) col 2 -> U index a*3 + 2
    # B21 row 2 col k (0..7) -> V index 2*8 + k
    # C12 row i (0,1) col k (0..7) -> W index k*2 + i
    # Factor for a=0: U[0*3+2]=1, V[2*8+k]=1 for all k, W[k*2+0]=1 for all k
    # But each factor must be rank-1: U entry * V entry. For schoolbook 2x1*1x8,
    # we need 2 factors (one per row of A12).
    for a in range(2):
        for k in range(p):
            U[a * 3 + 2][col] = 1
            V[2 * 8 + k][col] = 1
            W[k * 2 + a][col] = 1
        # Wait, that's not rank-1 per factor. We need rank 2 total for 2x1*1x8.
        # Actually 2x1 * 1x8 = sum over the single middle index: rank = 2*1*8? No.
        # 2x1 * 1x8: the product is a 2x8 matrix where C[i][k] = A[i][0]*B[0][k].
        # This is outer product of A12 (2x1) and B21 (1x8) = rank 1!
        # So only 1 factor needed.
        break
    # Actually 2x1 * 1x8 is rank 1 (outer product). Let me use 1 factor.
    # Reset: I'll just do it cleanly.

    # Redo from scratch to be safe.
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    def ua(a, c): return a * 3 + c
    def vb(r, k): return r * 8 + k
    def wc(i, k): return k * 2 + i

    col = 0
    # M1
    U[ua(0,0)][col]=1; V[vb(0,0)][col]=1; W[wc(0,0)][col]=1; W[wc(1,0)][col]=1; col+=1
    # M2
    U[ua(0,0)][col]=1; U[ua(0,1)][col]=1; V[vb(1,0)][col]=1; W[wc(1,0)][col]=1; W[wc(1,1)][col]=-1; col+=1
    # M3
    U[ua(1,0)][col]=1; V[vb(0,0)][col]=1; V[vb(1,0)][col]=1; W[wc(0,1)][col]=1; W[wc(1,0)][col]=1; col+=1
    # M4
    U[ua(1,1)][col]=1; V[vb(1,0)][col]=1; V[vb(0,0)][col]=-1; W[wc(0,0)][col]=1; W[wc(1,0)][col]=1; col+=1
    # M5
    U[ua(0,0)][col]=1; U[ua(1,0)][col]=1; V[vb(0,1)][col]=1; V[vb(1,1)][col]=1; W[wc(0,0)][col]=-1; W[wc(0,1)][col]=1; col+=1
    # M6
    U[ua(1,1)][col]=1; U[ua(1,0)][col]=-1; V[vb(0,1)][col]=1; V[vb(1,1)][col]=-1; W[wc(1,1)][col]=1; col+=1
    # M7
    U[ua(0,1)][col]=1; U[ua(1,1)][col]=1; V[vb(1,0)][col]=1; V[vb(1,1)][col]=1; W[wc(0,0)][col]=1; col+=1
    # 2x1 * 1x8 outer product (rank 1)
    for a in range(2):
        U[ua(a,2)][col] = 1
    for k in range(8):
        V[vb(2,k)][col] = 1
        W[wc(0,k)][col] = 1
        W[wc(1,k)][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
