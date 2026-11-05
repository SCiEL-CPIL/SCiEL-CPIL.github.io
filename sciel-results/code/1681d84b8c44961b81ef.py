"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 5

    # Strassen 2x2x2 factorization (rank 7)
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    def build_decomposition_strassen():
        """Build rank-71 decomposition using 4 Strassen 2x2x2 blocks + schoolbook."""
        U = [[0] * 71 for _ in range(n * m)]
        V = [[0] * 71 for _ in range(m * p)]
        W = [[0] * 71 for _ in range(p * n)]

        col = 0
        blocks = [
            ([0, 1], [0, 1], [0, 1]),
            ([0, 1], [0, 1], [2, 3]),
            ([0, 1], [2, 3], [0, 1]),
            ([0, 1], [2, 3], [2, 3]),
        ]

        covered = set()
        for i_idx, j_idx, k_idx in blocks:
            for i in i_idx:
                for j in j_idx:
                    for k in k_idx:
                        covered.add((i, j, k))

            i0, i1 = i_idx
            j0, j1 = j_idx
            k0, k1 = k_idx

            u_rows = [i0 * m + j0, i0 * m + j1, i1 * m + j0, i1 * m + j1]
            v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            w_rows = [k0 * n + i0, k1 * n + i0, k0 * n + i1, k1 * n + i1]

            for c in range(7):
                for s in range(4):
                    U[u_rows[s]][col + c] = U_strassen[s][c]
                    V[v_rows[s]][col + c] = V_strassen[s][c]
                    W[w_rows[s]][col + c] = W_strassen[s][c]
            col += 7

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

        assert col == 71, f"Expected rank 71, got {col}"
        return U, V, W

    def build_decomposition_alt():
        """Try alternative partition: 5 Strassen blocks with careful overlap management."""
        # Blocks:
        # A: i{0,1}, j{0,1}, k{0,1} - 8 terms
        # B: i{0,1}, j{2,3}, k{2,3} - 8 terms  
        # C: i{0,1}, j{0,1}, k{2,3} - 8 terms
        # D: i{0,1}, j{2,3}, k{0,1} - 8 terms
        # E: i{1,2}, j{3,4}, k{3,4} - 8 terms
        # Overlap D&E: i=1, j=3, k=3 -> 1 term
        # Total covered: 4*8 + 8 - 1 = 39
        # Remaining: 75 - 39 = 36
        # Total rank: 5*7 + 36 = 71 (same as before)
        # Try different: skip E, use 4 blocks = 71
        return build_decomposition_strassen()

    U, V, W = build_decomposition_strassen()

    # Verify using numpy for speed
    def verify_fast(U_mat, V_mat, W_mat):
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i*m+j, :] * V_np[j*p+k, :], W_np[k*n+i, :])
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    if not verify_fast(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    if time.time() > deadline:
        # Should not happen for this simple construction, but safety guard
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
