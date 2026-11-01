"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 120.0

def construct():
    # Rank-35 decomposition: Strassen (rank-7) applied independently to each
    # of the 5 disjoint 2-column blocks of the 2x10 matrix B.
    # The tensor is a direct sum over blocks, so rank is additive: 5*7 = 35.
    n, m, p = DIMENSIONS  # 2, 2, 10
    num_blocks = p // 2   # 5
    rank = num_blocks * 7 # 35

    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35

    # Strassen 7 multiplications for 2x2 matmul:
    # A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]]
    # M1=(A00+A11)(B00+B11)  -> C00+=M1, C11+=M1
    # M2=(A10+A11)B00        -> C10+=M2, C11-=M2
    # M3=A00(B01-B11)        -> C01+=M3, C11+=M3
    # M4=A11(B10-B00)        -> C00+=M4, C10+=M4
    # M5=(A00+A01)B11        -> C01+=M5, C00-=M5
    # M6=(A10-A00)(B00+B01)  -> C11+=M6
    # M7=(A01-A11)(B10+B11)  -> C00+=M7
    #
    # U row idx: 0=A00, 1=A01, 2=A10, 3=A11
    # V within-block koff: 0=B00, 1=B01, 2=B10, 3=B11  (actual row = j*p + 2b + koff%2, j=koff//2)
    # W within-block koff: 0=C00, 1=C01, 2=C10, 3=C11  (actual row = (2b+koff%2)*n + koff//2)

    terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),          # M1
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),                # M2
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),                # M3
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),                # M4
        ([(0,1),(1,1)], [(3,1)], [(1,1),(0,-1)]),                # M5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),                # M6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),                # M7
    ]

    for b in range(num_blocks):
        base = b * 7
        for t in range(7):
            col = base + t
            uc, vc, wc = terms[t]
            for ri, c in uc:
                U[ri][col] = c
            for koff, c in vc:
                j = koff >> 1
                k = 2 * b + (koff & 1)
                V[j * p + k][col] = c
            for koff, c in wc:
                i = koff >> 1
                k = 2 * b + (koff & 1)
                W[k * n + i][col] = c

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
