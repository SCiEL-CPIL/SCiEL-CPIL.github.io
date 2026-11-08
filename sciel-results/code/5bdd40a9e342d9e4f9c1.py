import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin

    n, m, p = DIMENSIONS

    def add_strassen_block(U, V, W, i0, j0, k0, col_offset):
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1

        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1

        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1

        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1

        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    # Try different numbers of Strassen blocks
    best_rank = n * m * p
    best_result = None

    # Possible positions for 2x2x2 blocks
    all_positions = [(i0, j0, k0) for i0 in range(n-1) for j0 in range(m-1) for k0 in range(p-1)]

    def blocks_overlap(b1, b2):
        i0_1, j0_1, k0_1 = b1
        i0_2, j0_2, k0_2 = b2
        if i0_1 + 1 < i0_2 or i0_2 + 1 < i0_1:
            return False
        if j0_1 + 1 < j0_2 or j0_2 + 1 < j0_1:
            return False
        if k0_1 + 1 < k0_2 or k0_2 + 1 < k0_1:
            return False
        return True

    # Try 1, 2, 3, 4 non-overlapping blocks
    for num_blocks in range(1, 5):
        if time.time() > deadline:
            break

        # Generate all sets of num_blocks non-overlapping blocks
        candidate_sets = []
        for combo in itertools.combinations(all_positions, num_blocks):
            valid = True
            for idx in range(num_blocks):
                for jdx in range(idx+1, num_blocks):
                    if blocks_overlap(combo[idx], combo[jdx]):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                candidate_sets.append(combo)

        for combo in candidate_sets:
            if time.time() > deadline:
                break

            strassen_rank = num_blocks * 7
            covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
            for i0, j0, k0 in combo:
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered_3d[i0+di][j0+dj][k0+dk] = True

            schoolbook_count = sum(1 for i in range(n) for j in range(m) for k in range(p) if not covered_3d[i][j][k])
            rank = strassen_rank + schoolbook_count

            if rank < best_rank:
                best_rank = rank

                U = [[0.0] * rank for _ in range(n * m)]
                V = [[0.0] * rank for _ in range(m * p)]
                W = [[0.0] * rank for _ in range(p * n)]

                col = 0
                for i0, j0, k0 in combo:
                    add_strassen_block(U, V, W, i0, j0, k0, col)
                    col += 7

                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if not covered_3d[i][j][k]:
                                U[i*m+j][col] += 1
                                V[j*p+k][col] += 1
                                W[k*n+i][col] += 1
                                col += 1

                best_result = {"U": U, "V": V, "W": W}

    if best_result is None:
        # Fallback: pure schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i*m*j + j*p + k
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
        best_result = {"U": U, "V": V, "W": W}

    return best_result

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")

    # Verify correctness
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                expected = 1.0
                err = abs(val - expected)
                max_err = max(max_err, err)
    print(f"Max error: {max_err}")
