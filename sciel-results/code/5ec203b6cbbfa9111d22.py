"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 solution using 4x (3x2x2 rank 11)
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Sub-problems: (j_start, k_start)
    # 0: j 0-1, k 0-1
    # 1: j 0-1, k 2-3
    # 2: j 2-3, k 0-1
    # 3: j 2-3, k 2-3
    for j_start in [0, 2]:
        for k_start in [0, 2]:
            # 11 terms for 3x2x2
            # Local A indices: i in 0..2, j_local in 0..1
            # Local B indices: j_local in 0..1, k_local in 0..1
            # Local C indices: i in 0..2, k_local in 0..1
            
            # Term 1: (a11+a22)(b11+b22)
            # U: A[0, j0] + A[1, j0]
            # V: B[j0, k0] + B[j1, k1]
            # W: C[0, k0] += 1, C[1, k1] += 1
            U[0 * m + (j_start + 0)][col] = 1
            U[1 * m + (j_start + 0)][col] = 1
            V[(j_start + 0) * p + (k_start + 0)][col] = 1
            V[(j_start + 1) * p + (k_start + 1)][col] = 1
            W[(k_start + 0) * n + 0][col] = 1
            W[(k_start + 1) * n + 1][col] = 1
            col += 1
            
            # Term 2: (a12+a22)b21
            # U: A[0, j1] + A[1, j1]
            # V: B[j1, k0]
            # W: C[0, k1] += 1 (from c12 = ... + M2 ...? No, c12 = M2 + M4 - M6 - M7. So C[0,1] gets M2)
            # Wait, c12 is C[0, 1]. c21 is C[1, 0].
            # c12 = M2 + M4 - M6 - M7. So W[k=1, i=0] += 1.
            U[0 * m + (j_start + 1)][col] = 1
            U[1 * m + (j_start + 1)][col] = 1
            V[(j_start + 1) * p + (k_start + 0)][col] = 1
            W[(k_start + 1) * n + 0][col] = 1
            col += 1

            # Term 3: a11(b22-b11)
            # U: A[0, j0]
            # V: B[j1, k1] - B[j0, k0]
            # W: C[1, 0] -= 1 (c21 = M3 ...), C[0, 0] -= 1 (c11 = ... - M3 ...)
            # c11 = M1 - M3 + M5 + M7. So C[0,0] gets -1.
            # c21 = M3 - M4 + M6 + M7. So C[1,0] gets +1.
            U[0 * m + (j_start + 0)][col] = 1
            V[(j_start + 1) * p + (k_start + 1)][col] = 1
            V[(j_start + 0) * p + (k_start + 0)][col] = -1
            W[(k_start + 0) * n + 0][col] = -1
            W[(k_start + 0) * n + 1][col] = 1
            col += 1

            # Term 4: a22(b11-b21)
            # U: A[1, j0]
            # V: B[j0, k0] - B[j1, k0]
            # W: C[0, 1] += 1 (c12 = ... + M4 ...), C[1, 0] -= 1 (c21 = ... - M4 ...), C[1, 1] += 1 (c22 = ... + M4 ...)
            # c12 = M2 + M4 - M6 - M7. C[0,1] +1.
            # c21 = M3 - M4 + M6 + M7. C[1,0] -1.
            # c22 = M1 - M2 + M4 - M5. C[1,1] +1.
            U[1 * m + (j_start + 0)][col] = 1
            V[(j_start + 0) * p + (k_start + 0)][col] = 1
            V[(j_start + 1) * p + (k_start + 0)][col] = -1
            W[(k_start + 1) * n + 0][col] = 1
            W[(k_start + 0) * n + 1][col] = -1
            W[(k_start + 1) * n + 1][col] = 1
            col += 1

            # Term 5: (a21+a11)b11
            # U: A[1, j0] + A[0, j0]
            # V: B[j0, k0]
            # W: C[0, 0] += 1 (c11 = ... + M5 ...), C[1, 1] -= 1 (c22 = ... - M5 ...)
            U[1 * m + (j_start + 0)][col] = 1
            U[0 * m + (j_start + 0)][col] = 1
            V[(j_start + 0) * p + (k_start + 0)][col] = 1
            W[(k_start + 0) * n + 0][col] = 1
            W[(k_start + 1) * n + 1][col] = -1
            col += 1

            # Term 6: (a22-a11)(b12+b22)
            # U: A[1, j1] - A[0, j1]
            # V: B[j0, k1] + B[j1, k1]
            # W: C[0, 1] -= 1 (c12 = ... - M6 ...), C[1, 0] += 1 (c21 = ... + M6 ...)
            U[1 * m + (j_start + 1)][col] = 1
            U[0 * m + (j_start + 1)][col] = -1
            V[(j_start + 0) * p + (k_start + 1)][col] = 1
            V[(j_start + 1) * p + (k_start + 1)][col] = 1
            W[(k_start + 1) * n + 0][col] = -1
            W[(k_start + 0) * n + 1][col] = 1
            col += 1

            # Term 7: (a12-a21)(b21+b22)
            # U: A[0, j1] - A[1, j1]
            # V: B[j1, k0] + B[j1, k1]
            # W: C[0, 0] += 1 (c11 = ... + M7 ...), C[0, 1] -= 1 (c12 = ... - M7 ...), C[1, 0] += 1 (c21 = ... + M7 ...)
            U[0 * m + (j_start + 1)][col] = 1
            U[1 * m + (j_start + 1)][col] = -1
            V[(j_start + 1) * p + (k_start + 0)][col] = 1
            V[(j_start + 1) * p + (k_start + 1)][col] = 1
            W[(k_start + 0) * n + 0][col] = 1
            W[(k_start + 1) * n + 0][col] = -1
            W[(k_start + 0) * n + 1][col] = 1
            col += 1

            # Term 8: a31 b11
            # U: A[2, j0]
            # V: B[j0, k0]
            # W: C[2, 0] += 1
            U[2 * m + (j_start + 0)][col] = 1
            V[(j_start + 0) * p + (k_start + 0)][col] = 1
            W[(k_start + 0) * n + 2][col] = 1
            col += 1

            # Term 9: a32 b21
            # U: A[2, j1]
            # V: B[j1, k0]
            # W: C[2, 0] += 1
            U[2 * m + (j_start + 1)][col] = 1
            V[(j_start + 1) * p + (k_start + 0)][col] = 1
            W[(k_start + 0) * n + 2][col] = 1
            col += 1

            # Term 10: a31 b12
            # U: A[2, j0]
            # V: B[j0, k1]
            # W: C[2, 1] += 1
            U[2 * m + (j_start + 0)][col] = 1
            V[(j_start + 0) * p + (k_start + 1)][col] = 1
            W[(k_start + 1) * n + 2][col] = 1
            col += 1

            # Term 11: a32 b22
            # U: A[2, j1]
            # V: B[j1, k1]
            # W: C[2, 1] += 1
            U[2 * m + (j_start + 1)][col] = 1
            V[(j_start + 1) * p + (k_start + 1)][col] = 1
            W[(k_start + 1) * n + 2][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
