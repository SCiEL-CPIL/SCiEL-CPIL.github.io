import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def add_strassen_block(U, V, W, col, i0, j0, k0):
        """
        Adds a Strassen decomposition for a 2x2x2 block starting at indices i0, j0, k0.
        Returns the next available column index.
        """
        # M1: (a+d)(e+h)
        U[i0 * m + j0, col] += 1
        U[(i0 + 1) * m + (j0 + 1), col] += 1
        V[j0 * p + k0, col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] += 1
        W[k0 * n + i0, col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] += 1
        col += 1

        # M2: (c+d)e
        U[(i0 + 1) * m + j0, col] += 1
        U[(i0 + 1) * m + (j0 + 1), col] += 1
        V[j0 * p + k0, col] += 1
        W[k0 * n + (i0 + 1), col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] -= 1
        col += 1

        # M3: a(f-h)
        U[i0 * m + j0, col] += 1
        V[j0 * p + (k0 + 1), col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] -= 1
        W[(k0 + 1) * n + i0, col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] += 1
        col += 1

        # M4: d(g-e)
        U[(i0 + 1) * m + (j0 + 1), col] += 1
        V[(j0 + 1) * p + k0, col] += 1
        V[j0 * p + k0, col] -= 1
        W[k0 * n + i0, col] += 1
        W[k0 * n + (i0 + 1), col] += 1
        col += 1

        # M5: (a+b)h
        U[i0 * m + j0, col] += 1
        U[i0 * m + (j0 + 1), col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] += 1
        W[k0 * n + i0, col] -= 1
        W[(k0 + 1) * n + i0, col] += 1
        col += 1

        # M6: (c-a)(e+f)
        U[(i0 + 1) * m + j0, col] += 1
        U[i0 * m + j0, col] -= 1
        V[j0 * p + k0, col] += 1
        V[j0 * p + (k0 + 1), col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] += 1
        col += 1

        # M7: (b-d)(g+h)
        U[i0 * m + (j0 + 1), col] += 1
        U[(i0 + 1) * m + (j0 + 1), col] -= 1
        V[(j0 + 1) * p + k0, col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] += 1
        W[k0 * n + i0, col] += 1
        col += 1

        return col

    def get_rank50_tiling():
        """
        Structural change: Tile 2x4x4 with 4 Strassen blocks (Rank 28)
        and 2x4x3 with 2 Hybrid blocks (Rank 22).
        Total Rank 50.
        """
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        # Part 1: 2x4x4 (k=0..3)
        # A is 2x4, B is 4x4.
        # Split A into A1 (rows 0, cols 0-1) and A2 (rows 0, cols 2-3)?
        # No, A is 2x4. i is 0..1, j is 0..3.
        # We can view this as two 2x2x2 blocks in j, and two in k?
        # Actually, the standard tiling for 2x4x4 uses 4 Strassen blocks.
        # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
        col = add_strassen_block(U, V, W, col, 0, 0, 0)
        # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
        col = add_strassen_block(U, V, W, col, 0, 2, 0)
        # Block 3: i in {0,1}, j in {0,1}, k in {2,3}
        col = add_strassen_block(U, V, W, col, 0, 0, 2)
        # Block 4: i in {0,1}, j in {2,3}, k in {2,3}
        col = add_strassen_block(U, V, W, col, 0, 2, 2)
        
        # Part 2: 2x4x3 (k=4..6)
        # We can split this into two 2x2x3 blocks?
        # j in {0,1} and j in {2,3}.
        # For each 2x2x3 block, we use Strassen for k=4,5 and Schoolbook for k=6.
        
        # Block 5: i in {0,1}, j in {0,1}, k in {4,5} (Strassen)
        col = add_strassen_block(U, V, W, col, 0, 0, 4)
        
        # Block 6: i in {0,1}, j in {2,3}, k in {4,5} (Strassen)
        col = add_strassen_block(U, V, W, col, 0, 2, 4)
        
        # Remaining: k=6 for all j.
        # This is 2x4x1. Rank 8.
        for j in range(m):
            U[0 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            col += 1
            U[1 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    # Get the rank-50 decomposition
    U50, V50, W50 = get_rank50_tiling()

    # Verify it's correct
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
# EVOLVE-BLOCK-END
