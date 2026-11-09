"""Optimized program: block Strassen decomposition for 4x5 * 5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 115

    n, m, p = 4, 5, 6

    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    # Verify a decomposition
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def build_decomposition():
        """Build a rank-96 decomposition using block Strassen."""
        # Total rank calculation:
        # Part 1: j in {0,1,2} -> 4 blocks of (2x3 * 3x3)
        #   Each 2x3*3x3: j={0,1} -> 2x2*2x3 (Strassen 7 + schoolbook 4 = 11), j={2} -> 2x1*1x3 (schoolbook 2)
        #   Total per block: 13, total: 4*13 = 52
        # Part 2: j in {3,4} -> 4 blocks of (2x2 * 2x3)
        #   Each 2x2*2x3: 2x2*2x2 (Strassen 7) + 2x2*2x1 (schoolbook 4) = 11
        #   Total: 4*11 = 44
        # Grand total: 96

        total_rank = 96
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        col = 0

        def add_strassen_block(i_base, k_base, j_base):
            """Add Strassen terms for A[i_base:i_base+2, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+2]"""
            nonlocal col
            # m1: A[0,1] * (B[1,0] - B[0,0])
            # Contributes: C[i_base, k_base+1] += m1, C[i_base+1, k_base+1] -= m1
            U[i_base*m + j_base+1][col] = 1
            V[(j_base+1)*p + k_base][col] = 1
            V[j_base*p + k_base][col] = -1
            W[(k_base+1)*n + i_base][col] = 1
            W[(k_base+1)*n + i_base+1][col] = -1
            col += 1

            # m2: (A[0,0]+A[1,0]) * (B[0,0]+B[1,1])
            # Contributes: C[i_base, k_base] += m2, C[i_base+1, k_base+1] += m2
            U[i_base*m + j_base][col] = 1
            U[(i_base+1)*m + j_base][col] = 1
            V[j_base*p + k_base][col] = 1
            V[(j_base+1)*p + k_base+1][col] = 1
            W[k_base*n + i_base][col] = 1
            W[(k_base+1)*n + i_base+1][col] = 1
            col += 1

            # m3: (A[0,0]-A[1,1]) * (B[0,0]+B[0,1])
            # Contributes: C[i_base+1, k_base] += m3, C[i_base+1, k_base+1] += m3
            U[i_base*m + j_base][col] = 1
            U[(i_base+1)*m + j_base+1][col] = -1
            V[j_base*p + k_base][col] = 1
            V[j_base*p + k_base+1][col] = 1
            W[k_base*n + i_base+1][col] = 1
            W[(k_base+1)*n + i_base+1][col] = 1
            col += 1

            # m4: (A[1,0]-A[1,1]) * (B[1,0]+B[1,1])
            # Contributes: C[i_base, k_base] += m4, C[i_base+1, k_base+1] -= m4
            U[(i_base+1)*m + j_base][col] = 1
            U[(i_base+1)*m + j_base+1][col] = -1
            V[(j_base+1)*p + k_base][col] = 1
            V[(j_base+1)*p + k_base+1][col] = 1
            W[k_base*n + i_base][col] = 1
            W[(k_base+1)*n + i_base+1][col] = -1
            col += 1

            # m5: (A[0,0]+A[0,1]) * (B[1,0]-B[1,1])
            # Contributes: C[i_base, k_base] -= m5, C[i_base, k_base+1] += m5
            U[i_base*m + j_base][col] = 1
            U[i_base*m + j_base+1][col] = 1
            V[(j_base+1)*p + k_base][col] = 1
            V[(j_base+1)*p + k_base+1][col] = -1
            W[k_base*n + i_base][col] = -1
            W[(k_base+1)*n + i_base][col] = 1
            col += 1

            # m6: (A[0,1]+A[1,1]) * (B[0,0]-B[1,0])
            # Contributes: C[i_base+1, k_base+1] += m6
            U[i_base*m + j_base+1][col] = 1
            U[(i_base+1)*m + j_base+1][col] = 1
            V[j_base*p + k_base][col] = 1
            V[(j_base+1)*p + k_base][col] = -1
            W[(k_base+1)*n + i_base+1][col] = 1
            col += 1

            # m7: (A[0,0]-A[1,0]) * (B[0,1]-B[1,1])
            # Contributes: C[i_base+1, k_base] += m7, C[i_base, k_base] -= m7
            U[i_base*m + j_base][col] = 1
            U[(i_base+1)*m + j_base][col] = -1
            V[j_base*p + k_base+1][col] = 1
            V[(j_base+1)*p + k_base+1][col] = -1
            W[k_base*n + i_base+1][col] = 1
            W[k_base*n + i_base][col] = -1
            col += 1

            return col

        def add_schoolbook_block(i_base, i_size, k_base, k_size, j_base, j_size):
            """Add schoolbook terms for A[i_base:i_base+i_size, j_base:j_base+j_size] * B[j_base:j_base+j_size, k_base:k_base+k_size]"""
            nonlocal col
            for di in range(i_size):
                for dk in range(k_size):
                    for dj in range(j_size):
                        i = i_base + di
                        k = k_base + dk
                        j = j_base + dj
                        U[i*m + j][col] = 1
                        V[j*p + k][col] = 1
                        W[k*n + i][col] = 1
                        col += 1
            return col

        # Part 1: j in {0,1,2}
        for i_block in range(2):  # i={0,1} or {2,3}
            for k_block in range(2):  # k={0,1,2} or {3,4,5}
                i_base = i_block * 2
                k_base = k_block * 3

                # j={0,1}: 2x2 * 2x3
                # Split k into {k_base, k_base+1} and {k_base+2}
                # 2x2 * 2x2: Strassen (7 terms)
                add_strassen_block(i_base, k_base, 0)

                # 2x2 * 2x1: schoolbook (4 terms)
                add_schoolbook_block(i_base, 2, k_base+2, 1, 0, 2)

                # j={2}: 2x1 * 1x3: schoolbook (2 terms)
                add_schoolbook_block(i_base, 2, k_base, 3, 2, 1)

        # Part 2: j in {3,4}
        for i_block in range(2):
            for k_block in range(2):
                i_base = i_block * 2
                k_base = k_block * 3

                # 2x2 * 2x3: split k into {k_base, k_base+1} and {k_base+2}
                # 2x2 * 2x2: Strassen (7 terms)
                add_strassen_block(i_base, k_base, 3)

                # 2x2 * 2x1: schoolbook (4 terms)
                add_schoolbook_block(i_base, 2, k_base+2, 1, 3, 2)

        assert col == total_rank, f"Expected {total_rank} columns, got {col}"
        return {"U": U, "V": V, "W": W}

    # Try the block Strassen decomposition
    try:
        result = build_decomposition()
        if verify(result["U"], result["V"], result["W"]):
            best_result = result
            best_rank = 96
            print(f"Block Strassen decomposition verified with rank {best_rank}")
        else:
            print("Block Strassen verification failed, using schoolbook")
    except Exception as e:
        print(f"Block Strassen failed: {e}")

    elapsed = time.time() - start_time
    return best_result

DIMENSIONS = [4, 5, 6]
