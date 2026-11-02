"""Matrix multiplication 2x2x16 with Strassen-based rank-56 decomposition."""

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (32, rank)
    #
    # We decompose the 2x16 matrix B into 8 blocks of 2x2.
    # For each block, we use a Strassen decomposition of rank 7.
    # Total rank = 8 * 7 = 56.
    #
    # B block b (b=0..7) occupies columns 2b and 2b+1.
    # B[0, 2b] -> V row 0*16 + 2b = 2b
    # B[0, 2b+1] -> V row 0*16 + 2b+1 = 2b+1
    # B[1, 2b] -> V row 1*16 + 2b = 16+2b
    # B[1, 2b+1] -> V row 1*16 + 2b+1 = 16+2b+1
    #
    # A matrix: A[0,0]->U row 0, A[0,1]->U row 1, A[1,0]->U row 2, A[1,1]->U row 3
    #
    # Output C[2x16]: C[i, k] where i in {0,1}, k in {0..15}
    # W index: k*n + i = 2*k + i
    # C[0, 2b] -> W[2*2b + 0] = 4b
    # C[0, 2b+1] -> W[2*(2b+1) + 0] = 4b+2
    # C[1, 2b] -> W[2*2b + 1] = 4b+1
    # C[1, 2b+1] -> W[2*(2b+1) + 1] = 4b+3
    #
    # Strassen for 2x2 * 2x2:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+g)
    # M2 = (c+d)(e)
    # M3 = (a)(f-g)
    # M4 = (d)(h-g)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = -M1 + M3 + M5 + M7
    # C[1,0] = M2 + M1 - M4 - M6
    # C[1,1] = M2 - M3 + M4 + M6
    #
    # Mapping to our indices for block b:
    # a = A[0,0] -> U[0], b = A[0,1] -> U[1], c = A[1,0] -> U[2], d = A[1,1] -> U[3]
    # e = B[0,2b] -> V[2b], f = B[0,2b+1] -> V[2b+1]
    # g = B[1,2b] -> V[16+2b], h = B[1,2b+1] -> V[16+2b+1]
    #
    # C[0,2b] -> W[4b], C[0,2b+1] -> W[4b+2], C[1,2b] -> W[4b+1], C[1,2b+1] -> W[4b+3]

    rank = 56
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(32)]
    W = [[0.0] * rank for _ in range(32)]

    for b in range(8):
        base = 7 * b
        # M1 = (a+d)(e+g)
        U[0][base] = 1.0   # a
        U[3][base] = 1.0   # d
        V[2*b][base] = 1.0       # e
        V[16+2*b][base] = 1.0    # g
        # C contributions: C[0,2b]+=M1, C[0,2b+1]-=M1, C[1,2b]+=M1, C[1,2b+1] no M1
        W[4*b][base] = 1.0
        W[4*b+2][base] = -1.0
        W[4*b+1][base] = 1.0

        # M2 = (c+d)(e)
        U[2][base+1] = 1.0   # c
        U[3][base+1] = 1.0   # d
        V[2*b][base+1] = 1.0       # e
        # C contributions: C[1,2b]+=M2, C[1,2b+1]+=M2
        W[4*b+1][base+1] = 1.0
        W[4*b+3][base+1] = 1.0

        # M3 = a(f-g)
        U[0][base+2] = 1.0   # a
        V[2*b+1][base+2] = 1.0      # f
        V[16+2*b][base+2] = -1.0    # -g
        # C contributions: C[0,2b+1]+=M3, C[1,2b+1]-=M3
        W[4*b+2][base+2] = 1.0
        W[4*b+3][base+2] = -1.0

        # M4 = d(h-g)
        U[3][base+3] = 1.0   # d
        V[16+2*b+1][base+3] = 1.0   # h
        V[16+2*b][base+3] = -1.0    # -g
        # C contributions: C[0,2b]+=M4, C[1,2b]-=M4, C[1,2b+1]+=M4
        W[4*b][base+3] = 1.0
        W[4*b+1][base+3] = -1.0
        W[4*b+3][base+3] = 1.0

        # M5 = (a+b)(h)
        U[0][base+4] = 1.0   # a
        U[1][base+4] = 1.0   # b
        V[16+2*b+1][base+4] = 1.0   # h
        # C contributions: C[0,2b]-=M5, C[0,2b+1]+=M5
        W[4*b][base+4] = -1.0
        W[4*b+2][base+4] = 1.0

        # M6 = (c-a)(e+f)
        U[2][base+5] = 1.0   # c
        U[0][base+5] = -1.0  # -a
        V[2*b][base+5] = 1.0       # e
        V[2*b+1][base+5] = 1.0     # f
        # C contributions: C[1,2b]-=M6, C[1,2b+1]+=M6
        W[4*b+1][base+5] = -1.0
        W[4*b+3][base+5] = 1.0

        # M7 = (b-d)(g+h)
        U[1][base+6] = 1.0   # b
        U[3][base+6] = -1.0  # -d
        V[16+2*b][base+6] = 1.0      # g
        V[16+2*b+1][base+6] = 1.0    # h
        # C contributions: C[0,2b]+=M7, C[0,2b+1]+=M7
        W[4*b][base+6] = 1.0
        W[4*b+2][base+6] = 1.0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
