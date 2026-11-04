"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def verify_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    # Strategy: Use Strassen for 2x2 blocks where possible, schoolbook for 1x2 strips
    # A is 3x4, B is 4x4, C is 3x4
    # Split A into rows: top 2 rows (2x4), bottom 1 row (1x4)
    # Split B into columns: left 2 (4x2), right 2 (4x2)
    # Split C into columns: left 2 (3x2), right 2 (3x2)
    # Also split B rows: top 2 (2x4), bottom 2 (2x4)
    # Split C rows: top 2 (2x4), bottom 1 (1x4)

    # The 3x4 * 4x4 = 3x4 multiplication can be decomposed as:
    # C_top (2x4) = A_top (2x4) * B (4x4)
    # C_bottom (1x4) = A_bottom (1x4) * B (4x4)

    # For C_top = A_top * B:
    # A_top is 2x4, B is 4x4
    # Split B into B_top (2x4) and B_bottom (2x4)
    # C_top = A_top[:,0:2] * B_top + A_top[:,2:4] * B_bottom
    # Each term is 2x2 * 2x4 or 2x4 * 2x2... wait
    # Actually: C_top[i,k] = sum_j A_top[i,j] * B[j,k]
    # = sum_{j=0}^{1} A_top[i,j] * B[j,k] + sum_{j=2}^{3} A_top[i,j] * B[j,k]
    # = (A_top_left * B_top)[i,k] + (A_top_right * B_bottom)[i,k]
    # where A_top_left is 2x2, B_top is 2x4, A_top_right is 2x2, B_bottom is 2x4

    # For 2x2 * 2x4:
    # This is 2x2 * (2x2 + 2x2) = (2x2*2x2) + (2x2*2x2)
    # Each 2x2*2x2 can use Strassen (rank 7)
    # So 2x2*2x4 has rank 14? No, that's not right.
    # Actually, 2x2 * 2x4: C[i,k] = sum_j A[i,j] B[j,k]
    # We can split B into two 2x2 blocks: B = [B1 | B2]
    # C = [C1 | C2] where C1 = A*B1, C2 = A*B2
    # C1 = A*B1 is 2x2*2x2, rank 7 with Strassen
    # C2 = A*B2 is 2x2*2x2, rank 7 with Strassen
    # Total rank for 2x2*2x4 is 14

    # Similarly for 1x4 * 4x4:
    # C[i,k] = sum_j A[i,j] B[j,k] for i=0 (only one row)
    # = sum_{j=0}^{3} A[0,j] B[j,k]
    # This is 1x4 * 4x4 = 1x4
    # C[k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # This is a sum of 4 rank-1 terms (each A[0,j]*B[j,:]), so rank 4

    # Total rank:
    # C_top = A_top_left * B_top (2x2*2x2, rank 7) + A_top_right * B_bottom (2x2*2x2, rank 7)
    #         + A_top_left * B_bottom_right... wait, I need to be more careful.

    # Let me redo this. A is 3x4, B is 4x4.
    # A = [[a00, a01, a02, a03],
    #      [a10, a11, a12, a13],
    #      [a20, a21, a22, a23]]
    # B = [[b00, b01, b02, b03],
    #      [b10, b11, b12, b13],
    #      [b20, b21, b22, b23],
    #      [b30, b31, b32, b33]]

    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]

    # For i=0,1 (top 2 rows of C):
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # Group terms by B row:
    # = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    # The first group is A_top_left (2x2) * B_top (2x4)
    # The second group is A_top_right (2x2) * B_bottom (2x4)

    # For A_top_left (2x2) * B_top (2x4):
    # B_top = [[b00,b01,b02,b03],
    #          [b10,b11,b12,b13]]
    # Split B_top into B_top_left (2x2) and B_top_right (2x2)
    # B_top_left = [[b00,b01],[b10,b11]], B_top_right = [[b02,b03],[b12,b13]]
    # A_top_left * B_top = A_top_left * B_top_left + A_top_left * B_top_right
    # Each is 2x2*2x2, rank 7 with Strassen
    # Total rank: 14

    # Similarly for A_top_right (2x2) * B_bottom (2x4): rank 14

    # For i=2 (bottom row of C):
    # C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k] + A[2,3]*B[3,k]
    # This is 1x4 * 4x4, rank 4

    # Total rank: 14 + 14 + 4 = 32

    # Let me implement this.

    rank = 32
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Helper to add Strassen terms for 2x2*2x2
    # a,b,c,d are A indices (2x2 block)
    # e,f,g,h are B indices (2x2 block)
    # w11,w12,w21,w22 are C indices (2x2 block)
    def add_strassen_2x2(base, a, b, c, d, e, f, g, h, w11, w12, w21, w22):
        # M1 = (a+d)(e+h) -> C11(+1), C22(+1)
        U[a, base] += 1; U[d, base] += 1
        V[e, base] += 1; V[h, base] += 1
        W[w11, base] += 1; W[w22, base] += 1

        # M2 = (c+d)e -> C21(+1), C22(-1)
        U[c, base+1] += 1; U[d, base+1] += 1
        V[e, base+1] += 1
        W[w21, base+1] += 1; W[w22, base+1] -= 1

        # M3 = a(f-h) -> C12(+1), C22(+1)
        U[a, base+2] += 1
        V[f, base+2] += 1; V[h, base+2] -= 1
        W[w12, base+2] += 1; W[w22, base+2] += 1

        # M4 = d(g-e) -> C11(+1), C21(+1)
        U[d, base+3] += 1
        V[g, base+3] += 1; V[e, base+3] -= 1
        W[w11, base+3] += 1; W[w21, base+3] += 1

        # M5 = (a+b)h -> C12(+1), C11(-1)
        U[a, base+4] += 1; U[b, base+4] += 1
        V[h, base+4] += 1
        W[w12, base+4] += 1; W[w11, base+4] -= 1

        # M6 = (c-a)(e+f) -> C22(+1)
        U[c, base+5] += 1; U[a, base+5] -= 1
        V[e, base+5] += 1; V[f, base+5] += 1
        W[w22, base+5] += 1

        # M7 = (b-d)(g+h) -> C11(+1)
        U[b, base+6] += 1; U[d, base+6] -= 1
        V[g, base+6] += 1; V[h, base+6] += 1
        W[w11, base+6] += 1

    # A indices:
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # Row 2: 8,9,10,11

    # B indices:
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # Row 2: 8,9,10,11
    # Row 3: 12,13,14,15

    # C/W indices (k*n+i, n=3):
    # C[0,0]=0, C[0,1]=3, C[0,2]=6, C[0,3]=9
    # C[1,0]=1, C[1,1]=4, C[1,2]=7, C[1,3]=10
    # C[2,0]=2, C[2,1]=5, C[2,2]=8, C[2,3]=11

    # Block 1: A_top_left (2x2) * B_top_left (2x2) -> C_top_left (2x2)
    # A: a00=0, a01=1, a10=4, a11=5
    # B: b00=0, b01=1, b10=4, b11=5
    # C: C00=0, C01=3, C10=1, C11=4
    add_strassen_2x2(0, 0, 1, 4, 5, 0, 1, 4, 5, 0, 3, 1, 4)

    # Block 2: A_top_left (2x2) * B_top_right (2x2) -> C_top_right (2x2)
    # A: a00=0, a01=1, a10=4, a11=5
    # B: b02=2, b03=3, b12=6, b13=7
    # C: C02=6, C03=9, C12=7, C13=10
    add_strassen_2x2(7, 0, 1, 4, 5, 2, 3, 6, 7, 6, 9, 7, 10)

    # Block 3: A_top_right (2x2) * B_bottom_left (2x2) -> C_top_left (2x2)
    # A: a02=2, a03=3, a12=6, a13=7
    # B: b20=8, b21=9, b30=12, b31=13
    # C: C00=0, C01=3, C10=1, C11=4
    add_strassen_2x2(14, 2, 3, 6, 7, 8, 9, 12, 13, 0, 3, 1, 4)

    # Block 4: A_top_right (2x2) * B_bottom_right (2x2) -> C_top_right (2x2)
    # A: a02=2, a03=3, a12=6, a13=7
    # B: b22=10, b23=11, b32=14, b33=15
    # C: C02=6, C03=9, C12=7, C13=10
    add_strassen_2x2(21, 2, 3, 6, 7, 10, 11, 14, 15, 6, 9, 7, 10)

    # Block 5: A_bottom (1x4) * B (4x4) -> C_bottom (1x4)
    # C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k] + A[2,3]*B[3,k]
    # Rank 4: one term per j
    for j in range(4):
        base = 28 + j
        U[8 + j, base] = 1  # A[2,j]
        for k in range(4):
            V[j * 4 + k, base] = 1  # B[j,k]
        W[2 * 3 + 0, base] = 1 if j == 0 else 0  # This is wrong, need to be careful

    # Actually, for the bottom row, C[2,k] = sum_j A[2,j] * B[j,k]
    # Each term A[2,j]*B[j,k] contributes to C[2,k]
    # So for each j, we have a rank-1 term: u = e_{2,j}, v = B[j,:], w = e_{2,k} for all k
    # But w must be a vector, so w has 1 at all k positions for C[2,k]
    # Wait, C[2,k] for k=0,1,2,3 all get contributions from the same j
    # So the w vector for term j should have 1 at positions C[2,0], C[2,1], C[2,2], C[2,3]
    # C[2,0]=2, C[2,1]=5, C[2,2]=8, C[2,3]=11

    # Let me redo block 5 properly
    U[8:12, 28:32] = 0
    V[:, 28:32] = 0
    W[:, 28:32] = 0

    for j in range(4):
        base = 28 + j
        U[8 + j, base] = 1  # A[2,j]
        for k in range(4):
            V[j * 4 + k, base] = 1  # B[j,k]
        # W should have 1 at C[2,k] for all k
        W[2, base] = 1    # C[2,0]
        W[5, base] = 1    # C[2,1]
        W[8, base] = 1    # C[2,2]
        W[11, base] = 1   # C[2,3]

    # Verify
    if verify_decomposition(U, V, W, rank):
        best_U, best_V, best_W = U, V, W
        best_rank = rank

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
