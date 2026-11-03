"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s limit

    n, m, p = DIMENSIONS

    # Rank-44 block-partitioned decomposition:
    # Split m=3 into m=2 (rows 0,1) and m=1 (row 2).
    # For m=2 part:
    #   Split p=8 into four p=2 blocks.
    #   Each p=2 block: Strassen 2x2x2 (rank 7) -> total rank 28
    # For m=1 part:
    #   Schoolbook 2x1x8 -> rank 16
    # Total rank: 44

    total_rank = 44

    U = [[0] * total_rank for _ in range(n * m)]  # shape (6, 44)
    V = [[0] * total_rank for _ in range(m * p)]  # shape (24, 44)
    W = [[0] * total_rank for _ in range(p * n)]  # shape (16, 44)

    col = 0

    def _add_strassen_block(i_off, j_off, k_off):
        """Add a Strassen 2x2x2 block at given offsets."""
        nonlocal col
        # m1: (a+d)(e+h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        # m2: (c+d)e
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1

        # m3: a(f-h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        # m4: d(g-e)
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += -1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        col += 1

        # m5: (a+b)h
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        col += 1

        # m6: (c-a)(e+f)
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+0)][col] += -1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        # m7: (b-d)(g+h)
        U[(i_off+0)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        col += 1

    def _add_schoolbook_2x1x8(i_off, j_off):
        """Add schoolbook terms for 2x1x8 block."""
        nonlocal col
        for i in range(2):
            for k in range(8):
                U[(i_off+i)*m + (j_off)][col] = 1
                V[(j_off)*p + k][col] = 1
                W[k*n + (i_off+i)][col] = 1
                col += 1

    # Part 1: m=1 part (j=2), schoolbook 2x1x8
    _add_schoolbook_2x1x8(0, 2)

    # Part 2: m=2 part (j=0,1), split p=8 into four p=2 blocks
    _add_strassen_block(0, 0, 0)   # Strassen for m rows 0-1, p cols 0-1
    _add_strassen_block(0, 0, 2)   # Strassen for m rows 0-1, p cols 2-3
    _add_strassen_block(0, 0, 4)   # Strassen for m rows 0-1, p cols 4-5
    _add_strassen_block(0, 0, 6)   # Strassen for m rows 0-1, p cols 6-7

    if time.time() - start_time > DEADLINE:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
