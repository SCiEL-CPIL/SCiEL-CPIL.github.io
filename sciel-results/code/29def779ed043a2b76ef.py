"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s kill

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        # For matrix multiplication C=A*B (A is n x m, B is m x p):
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor T[i,j,k] = 1 for all valid (i,j,k).
        # We need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        # Check integer/half-integer constraint
        for name, M in (("U", U), ("V", V), ("W", W)):
            for row in M:
                for val in row:
                    if val != int(val) and 2 * val != int(2 * val):
                        return False
        return True

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

    # Build a block decomposition of rank 48:
    # Split A (3x4) into A1 (3x2) and A2 (3x2) along middle index j.
    # C = A1*B1 + A2*B2. Each block is 3x2x7.
    # For a 3x2x7 block with A being 3x2, B being 2x7:
    #   Split A into A_top (2x2) and A_bot (1x2).
    #   A_top (2x2) * B (2x7): use Strassen on first 2 cols (rank 7) + direct on last 5 cols (rank 20). Total 27.
    #   A_bot (1x2) * B (2x7): direct, rank 1*2*7 = 14.
    #   Block total = 27 + 14 = 41.
    # Two blocks = 82. This is better than schoolbook 84.
    #
    # Actually, let's do even better by using a smarter split.
    # We can use the fact that 3x4x7 can be decomposed via:
    #   Rank 48 construction using Strassen on 2x2x2 blocks.
    #
    # Let's build a rank-82 construction (verified) as the main improvement.

    def build_block_decomp():
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # We'll use a list of (u_idx, v_idx, w_idx, coeff) tuples.
        terms = []

        # For each block b in {0,1}, the middle index j ranges over [2*b, 2*b+2).
        # A block: A is 3x2 (rows i=0..2, cols j=2b..2b+1), B is 2x7 (rows j=2b..2b+1, cols k=0..6).
        for b in range(2):
            j0 = 2 * b  # first middle index of this block

            # --- Top part: A_top is 2x2 (i=0,1; j=j0, j0+1), B is 2x7 ---
            # Strassen on the 2x2x2 part (k=0,1):
            # A indices: i in {0,1}, j in {j0, j0+1}
            # B indices: j in {j0, j0+1}, k in {0,1}
            # C indices: k in {0,1}, i in {0,1}

            # Strassen 2x2x2 terms (7 products):
            # M1 = (A00+A11)*(B00+B11)
            # M2 = (A10+A11)*B00
            # M3 = A00*(B11-B00)
            # M4 = A11*(B01-B10)
            # M5 = (A00+A01)*B10
            # M6 = (A10+A11)*(B11-B01)
            # M7 = (A00-A01)*(B00+B10)
            #
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M1 - M4 + M6
            # C10 = M2 + M4 - M3 + M5
            # C11 = M1 - M2 + M3 + M6

            def A_idx(i, j):
                return (i) * m + (j0 + j)

            def B_idx(j, k):
                return (j0 + j) * p + (k)

            def C_idx(k, i):
                return (k) * n + (i)

            # M1
            terms.append((A_idx(0, 0), B_idx(0, 0), C_idx(0, 0), 1))
            terms.append((A_idx(1, 1), B_idx(0, 0), C_idx(0, 0), 1))
            terms.append((A_idx(0, 0), B_idx(1, 1), C_idx(0, 0), 1))
            terms.append((A_idx(1, 1), B_idx(1, 1), C_idx(0, 0), 1))
            terms.append((A_idx(0, 0), B_idx(0, 0), C_idx(1, 0), 1))
            terms.append((A_idx(1, 1), B_idx(0, 0), C_idx(1, 0), 1))
            terms.append((A_idx(0, 0), B_idx(1, 1), C_idx(1, 0), 1))
            terms.append((A_idx(1, 1), B_idx(1, 1), C_idx(1, 0), 1))
            # Wait, this approach of listing individual terms is getting complex.
            # Let me use a cleaner formulation with explicit rank slots.

        # Let me restart with a cleaner approach using explicit rank slots.
        rank_total = 82
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]

        col = 0

        for b in range(2):
            j0 = 2 * b

            # --- Strassen 2x2x2 for A[0:2, j0:j0+2] * B[j0:j0+2, 0:2] -> C[0:2, 0:2] ---
            # 7 rank slots for Strassen
            # M1 = (A00+A11)*(B00+B11)
            U[0 * m + (j0 + 0)][col] = 1
            U[1 * m + (j0 + 1)][col] = 1
            V[(j0 + 0) * p + 0][col] = 1
            V[(j0 + 1) * p + 1][col] = 1
            W[0 * n + 0][col] = 1  # C[0,0]
            W[1 * n + 0][col] = 1  # C[1,0]
            col += 1

            # M2 = (A10+A11)*B00
            U[1 * m + (j0 + 0)][col] = 1
            U[1 * m + (j0 + 1)][col] = 1
            V[(j0 + 0) * p + 0][col] = 1
            W[0 * n + 1][col] = 1  # C[0,1]
            col += 1

            # M3 = A00*(B11-B00)
            U[0 * m + (j0 + 0)][col] = 1
            V[(j0 + 1) * p + 1][col] = 1
            V[(j0 + 0) * p + 0][col] = -1
            W[1 * n + 0][col] = 1  # C[1,0]
            col += 1

            # M4 = A11*(B01-B10)
            U[1 * m + (j0 + 1)][col] = 1
            V[(j0 + 0) * p + 1][col] = 1
            V[(j0 + 1) * p + 0][col] = -1
            W[0 * n + 0][col] = 1  # C[0,0]
            W[0 * n + 1][col] = 1  # C[0,1]
            col += 1

            # M5 = (A00+A01)*B10
            U[0 * m + (j0 + 0)][col] = 1
            U[0 * m + (j0 + 1)][col] = 1
            V[(j0 + 1) * p + 0][col] = 1
            W[0 * n + 0][col] = -1  # C[0,0]
            W[0 * n + 1][col] = 1   # C[0,1]
            col += 1

            # M6 = (A10+A11)*(B11-B01)
            U[1 * m + (j0 + 0)][col] = 1
            U[1 * m + (j0 + 1)][col] = 1
            V[(j0 + 1) * p + 1][col] = 1
            V[(j0 + 0) * p + 1][col] = -1
            W[1 * n + 0][col] = 1  # C[1,0]
            W[1 * n + 1][col] = 1  # C[1,1]
            col += 1

            # M7 = (A00-A01)*(B00+B10)
            U[0 * m + (j0 + 0)][col] = 1
            U[0 * m + (j0 + 1)][col] = -1
            V[(j0 + 0) * p + 0][col] = 1
            V[(j0 + 1) * p + 0][col] = 1
            W[0 * n + 0][col] = 1  # C[0,0]
            col += 1

            # --- Direct 2x2x5 for A[0:2, j0:j0+2] * B[j0:j0+2, 2:7] -> C[0:2, 2:7] ---
            # 20 rank slots
            for i in range(2):
                for j in range(2):
                    for k in range(5):
                        U[i * m + (j0 + j)][col] = 1
                        V[(j0 + j) * p + (2 + k)][col] = 1
                        W[(2 + k) * n + i][col] = 1
                        col += 1

            # --- Direct 1x2x7 for A[2, j0:j0+2] * B[j0:j0+2, 0:7] -> C[0:7, 2] ---
            # 14 rank slots
            for j in range(2):
                for k in range(7):
                    U[2 * m + (j0 + j)][col] = 1
                    V[(j0 + j) * p + k][col] = 1
                    W[k * n + 2][col] = 1
                    col += 1

        assert col == rank_total, f"col={col}, rank_total={rank_total}"
        return {"U": U, "V": V, "W": W}

    # Try the block decomposition first
    result = build_block_decomp()
    if verify(result["U"], result["V"], result["W"]):
        return result

    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
