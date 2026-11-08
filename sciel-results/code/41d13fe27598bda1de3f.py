"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 5

    def verify(U, V, W):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    def schoolbook():
        """Return the rank-75 schoolbook decomposition."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def add_strassen_block(i0, j0, k0, U, V, W, col):
        """Add a 2x2x2 Strassen block at position (i0, j0, k0)."""
        # M1 = (A00+A11)(B00+B11)
        c = col + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M2 = (A10+A11)B00
        c = col + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1

        # M3 = A00(B01-B11)
        c = col + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M4 = A11(B10-B00)
        c = col + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1

        # M5 = (A00+A01)B11
        c = col + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1

        # M6 = (A10-A00)(B00+B01)
        c = col + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M7 = (A01-A11)(B10+B11)
        c = col + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    def try_strassen_blocks():
        """Try various placements of 2x2x2 Strassen blocks."""
        # All possible block positions: i0 in {0}, j0 in {0,2,3}, k0 in {0,2,3}
        # (i0=1 would exceed n=3 since we need i0+1=2 < 3)
        i_positions = [0]
        j_positions = [0, 2, 3]
        k_positions = [0, 2, 3]

        all_blocks = [(i0, j0, k0) for i0 in i_positions for j0 in j_positions for k0 in k_positions]

        # Try all subsets of size up to 4 (to keep rank manageable)
        # But we need to be careful about overlaps
        best_result = None
        best_rank = n * m * p  # 75

        # Try single blocks
        for i0, j0, k0 in all_blocks:
            if time.time() - start_time > DEADLINE:
                break
            rank = 7 + (n*m*p - 8)  # 7 + 67 = 74
            if rank >= best_rank:
                continue

            covered = set()
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if i0+di < n and j0+dj < m and k0+dk < p:
                            covered.add((i0+di, j0+dj, k0+dk))

            if len(covered) != 8:
                continue

            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]

            add_strassen_block(i0, j0, k0, U, V, W, 0)
            col = 7

            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i, j, k) not in covered:
                            U[i*m+j][col] += 1
                            V[j*p+k][col] += 1
                            W[k*n+i][col] += 1
                            col += 1

            if col == rank and verify(U, V, W):
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}

        # Try pairs of non-overlapping blocks
        for b1_idx in range(len(all_blocks)):
            if time.time() - start_time > DEADLINE:
                break
            for b2_idx in range(b1_idx + 1, len(all_blocks)):
                i0a, j0a, k0a = all_blocks[b1_idx]
                i0b, j0b, k0b = all_blocks[b2_idx]

                # Check overlap
                covered_a = set()
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            if i0a+di < n and j0a+dj < m and k0a+dk < p:
                                covered_a.add((i0a+di, j0a+dj, k0a+dk))

                covered_b = set()
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            if i0b+di < n and j0b+dj < m and k0b+dk < p:
                                covered_b.add((i0b+di, j0b+dj, k0b+dk))

                if covered_a & covered_b:
                    continue

                rank = 14 + (n*m*p - 16)  # 14 + 59 = 73
                if rank >= best_rank:
                    continue

                covered = covered_a | covered_b
                U = [[0.0] * rank for _ in range(n * m)]
                V = [[0.0] * rank for _ in range(m * p)]
                W = [[0.0] * rank for _ in range(p * n)]

                add_strassen_block(i0a, j0a, k0a, U, V, W, 0)
                add_strassen_block(i0b, j0b, k0b, U, V, W, 7)
                col = 14

                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if (i, j, k) not in covered:
                                U[i*m+j][col] += 1
                                V[j*p+k][col] += 1
                                W[k*n+i][col] += 1
                                col += 1

                if col == rank and verify(U, V, W):
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}

        # Try triples of non-overlapping blocks
        for b1_idx in range(len(all_blocks)):
            if time.time() - start_time > DEADLINE:
                break
            for b2_idx in range(b1_idx + 1, len(all_blocks)):
                if time.time() - start_time > DEADLINE:
                    break
                for b3_idx in range(b2_idx + 1, len(all_blocks)):
                    i0a, j0a, k0a = all_blocks[b1_idx]
                    i0b, j0b, k0b = all_blocks[b2_idx]
                    i0c, j0c, k0c = all_blocks[b3_idx]

                    covered_a = set()
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                if i0a+di < n and j0a+dj < m and k0a+dk < p:
                                    covered_a.add((i0a+di, j0a+dj, k0a+dk))

                    covered_b = set()
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                if i0b+di < n and j0b+dj < m and k0b+dk < p:
                                    covered_b.add((i0b+di, j0b+dj, k0b+dk))

                    covered_c = set()
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                if i0c+di < n and j0c+dj < m and k0c+dk < p:
                                    covered_c.add((i0c+di, j0c+dj, k0c+dk))

                    if (covered_a & covered_b) or (covered_a & covered_c) or (covered_b & covered_c):
                        continue

                    rank = 21 + (n*m*p - 24)  # 21 + 51 = 72
                    if rank >= best_rank:
                        continue

                    covered = covered_a | covered_b | covered_c
                    U = [[0.0] * rank for _ in range(n * m)]
                    V = [[0.0] * rank for _ in range(m * p)]
                    W = [[0.0] * rank for _ in range(p * n)]

                    add_strassen_block(i0a, j0a, k0a, U, V, W, 0)
                    add_strassen_block(i0b, j0b, k0b, U, V, W, 7)
                    add_strassen_block(i0c, j0c, k0c, U, V, W, 14)
                    col = 21

                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if (i, j, k) not in covered:
                                    U[i*m+j][col] += 1
                                    V[j*p+k][col] += 1
                                    W[k*n+i][col] += 1
                                    col += 1

                    if col == rank and verify(U, V, W):
                        best_rank = rank
                        best_result = {"U": U, "V": V, "W": W}

        return best_result, best_rank

    # Try to find a better decomposition using Strassen blocks
    result, best_rank = try_strassen_blocks()

    if result is not None and verify(result["U"], result["V"], result["W"]):
        return result
    else:
        # Fallback to schoolbook
        return schoolbook()

# EVOLVE-BLOCK-END
