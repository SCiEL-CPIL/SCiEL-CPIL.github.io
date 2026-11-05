"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Hybrid decomposition: Strassen for 2x2x2 sub-block (i,j,k in 0..1),
    # schoolbook for all remaining (i,j,k) triples.
    n, m, p = DIMENSIONS  # 3, 5, 5
    rank = 75
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    # ---- Strassen for 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1}) ----
    # 7 Strassen products M1..M7. Each M_t = (linear combo of A) * (linear combo of B)
    # For 2x2 matrices A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b02,b02]]
    # Standard Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b00)   # careful with B indexing: B is [j][k], so B[1][0]=b10
    # Wait: B has indices (j,k). In our 2x2 block, j in {0,1}, k in {0,1}.
    # B[0][0], B[0][1], B[1][0], B[1][1]. Let me denote B_jk.
    # Standard Strassen for C = A*B where A[i][j], B[j][k]:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B11 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Mapping: for term t in 0..6, we set U[i*m+j][col], V[j*p+k][col], W[k*n+i][col]
    # with coefficients.

    # M1: (a00+a11)(b00+b11)
    # U coeffs: i=0,j=0 -> +1; i=1,j=1 -> +1
    # V coeffs: j=0,k=0 -> +1; j=1,k=1 -> +1
    U[0*5+0][col] += 1; V[0*5+0][col] += 1; W[0*3+0][col] += 1  # i=0,k=0 -> M1 contributes to C00
    U[0*5+0][col] += 1; V[1*5+1][col] += 1; W[1*3+0][col] += 1  # i=0,k=1
    U[1*5+1][col] += 1; V[0*5+0][col] += 1; W[0*3+1][col] += 1  # i=1,k=0
    U[1*5+1][col] += 1; V[1*5+1][col] += 1; W[1*3+1][col] += 1  # i=1,k=1
    col += 1

    # M2: (a10+a11)b00
    # U: i=1,j=0 -> +1; i=1,j=1 -> +1
    # V: j=0,k=0 -> +1
    U[1*5+0][col] += 1; V[0*5+0][col] += 1; W[0*3+1][col] += 1  # i=1,k=0
    col += 1

    # M3: a00(b10 - b00)
    # U: i=0,j=0 -> +1
    # V: j=1,k=0 -> +1; j=0,k=0 -> -1
    U[0*5+0][col] += 1; V[1*5+0][col] += 1; W[0*3+0][col] += 1  # i=0,k=0: +1 from b10
    U[0*5+0][col] += 0; V[0*5+0][col] += -1; W[0*3+0][col] += -1  # i=0,k=0: -1 from b00
    U[0*5+0][col] += 1; V[1*5+1][col] += 0; W[1*3+0][col] += 0  # not used, but let me be careful
    # Actually M3 = a00 * (b10 - b00), so V has j=1,k=0: +1 and j=0,k=0: -1
    # C01 += M3, so k=1. Let me redo:
    # W index is k*n+i. For M3 contributing to C01 (i=0,k=1):
    # W[1*3+0] should get coefficient 1 for the whole product.
    # U[0*5+0] gets +1 (a00)
    # V[1*5+0] gets +1 (b10), V[0*5+0] gets -1 (b00)
    # Let me reset col and redo this properly.

    # Let me restart with a cleaner approach.
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    # M1: (a00+a11)(b00+b11)
    # A part: a00 (+1), a11 (+1) -> U[0*5+0]=+1, U[1*5+1]=+1
    # B part: b00 (+1), b11 (+1) -> V[0*5+0]=+1, V[1*5+1]=+1
    # C: M1 adds to C00, C01, C10, C11
    # C00 (i=0,k=0): W[0*3+0] += 1
    # C01 (i=0,k=1): W[1*3+0] += 1
    # C10 (i=1,k=0): W[0*3+1] += 1
    # C11 (i=1,k=1): W[1*3+1] += 1
    U[0*5+0][col] = 1; U[1*5+1][col] = 1
    V[0*5+0][col] = 1; V[1*5+1][col] = 1
    W[0*3+0][col] = 1
    W[1*3+0][col] = 1
    W[0*3+1][col] = 1
    W[1*3+1][col] = 1
    col += 1

    # M2: (a10+a11)b00
    # A: a10 (+1), a11 (+1) -> U[1*5+0]=+1, U[1*5+1]=+1
    # B: b00 (+1) -> V[0*5+0]=+1
    # C: C10 += M2, C11 -= M2
    # C10 (i=1,k=0): W[0*3+1] += 1
    # C11 (i=1,k=1): W[1*3+1] += -1
    U[1*5+0][col] = 1; U[1*5+1][col] = 1
    V[0*5+0][col] = 1
    W[0*3+1][col] = 1
    W[1*3+1][col] = -1
    col += 1

    # M3: a00(b10 - b00)
    # A: a00 (+1) -> U[0*5+0]=+1
    # B: b10 (+1), b00 (-1) -> V[1*5+0]=+1, V[0*5+0]=-1
    # C: C01 += M3, C11 += M3
    # C01 (i=0,k=1): W[1*3+0] += 1
    # C11 (i=1,k=1): W[1*3+1] += 1
    U[0*5+0][col] = 1
    V[1*5+0][col] = 1; V[0*5+0][col] = -1
    W[1*3+0][col] = 1
    W[1*3+1][col] = 1
    col += 1

    # M4: a11(b11 - b00)
    # A: a11 (+1) -> U[1*5+1]=+1
    # B: b11 (+1), b00 (-1) -> V[1*5+1]=+1, V[0*5+0]=-1
    # C: C00 += M4, C10 += M4
    # C00 (i=0,k=0): W[0*3+0] += 1
    # C10 (i=1,k=0): W[0*3+1] += 1
    U[1*5+1][col] = 1
    V[1*5+1][col] = 1; V[0*5+0][col] = -1
    W[0*3+0][col] = 1
    W[0*3+1][col] = 1
    col += 1

    # M5: (a00+a01)b11
    # A: a00 (+1), a01 (+1) -> U[0*5+0]=+1, U[0*5+1]=+1
    # B: b11 (+1) -> V[1*5+1]=+1
    # C: C00 -= M5, C01 += M5
    # C00 (i=0,k=0): W[0*3+0] += -1
    # C01 (i=0,k=1): W[1*3+0] += 1
    U[0*5+0][col] = 1; U[0*5+1][col] = 1
    V[1*5+1][col] = 1
    W[0*3+0][col] = -1
    W[1*3+0][col] = 1
    col += 1

    # M6: (a10-a00)(b00+b01)
    # A: a10 (+1), a00 (-1) -> U[1*5+0]=+1, U[0*5+0]=-1
    # B: b00 (+1), b01 (+1) -> V[0*5+0]=+1, V[0*5+1]=+1
    # C: C11 += M6
    # C11 (i=1,k=1): W[1*3+1] += 1
    U[1*5+0][col] = 1; U[0*5+0][col] = -1
    V[0*5+0][col] = 1; V[0*5+1][col] = 1
    W[1*3+1][col] = 1
    col += 1

    # M7: (a01-a11)(b10+b11)
    # A: a01 (+1), a11 (-1) -> U[0*5+1]=+1, U[1*5+1]=-1
    # B: b10 (+1), b11 (+1) -> V[1*5+0]=+1, V[1*5+1]=+1
    # C: C00 += M7
    # C00 (i=0,k=0): W[0*3+0] += 1
    U[0*5+1][col] = 1; U[1*5+1][col] = -1
    V[1*5+0][col] = 1; V[1*5+1][col] = 1
    W[0*3+0][col] = 1
    col += 1

    # ---- Schoolbook for all other (i,j,k) ----
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
