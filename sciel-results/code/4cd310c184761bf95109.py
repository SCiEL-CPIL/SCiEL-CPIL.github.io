import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 118

    # Target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-8):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

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

    def get_strassen_2x2x2(i0, j0, k0):
        """
        Returns U, V, W contributions for a 2x2x2 Strassen block.
        A rows: i0, i0+1
        A cols/B rows: j0, j0+1
        B cols: k0, k0+1
        C cols: k0, k0+1 (for A rows i0, i0+1)
        """
        U = np.zeros((n * m, 7))
        V = np.zeros((m * p, 7))
        W = np.zeros((p * n, 7))
        
        # Standard Strassen 7 multiplications
        # M1 = (A11 + A22)(B11 + B22)
        U[i0*m + j0, 0] += 1; U[(i0+1)*m + (j0+1), 0] += 1
        V[j0*p + k0, 0] += 1; V[(j0+1)*p + (k0+1), 0] += 1
        W[k0*n + i0, 0] += 1; W[(k0+1)*n + (i0+1), 0] += 1
        
        # M2 = A22(B21 - B11)
        U[(i0+1)*m + (j0+1), 1] += 1
        V[j0*p + k0, 1] += 1; V[(j0+1)*p + (k0+1), 1] -= 1
        W[(k0+1)*n + (i0+1), 1] += 1
        
        # M3 = A11(B12 - B22)
        U[i0*m + j0, 2] += 1
        V[j0*p + (k0+1), 2] += 1; V[(j0+1)*p + (k0+1), 2] -= 1
        W[(k0+1)*n + i0, 2] += 1
        
        # M4 = A22(B11 + B12)
        U[(i0+1)*m + (j0+1), 3] += 1
        V[j0*p + k0, 3] += 1; V[j0*p + (k0+1), 3] += 1
        W[k0*n + (i0+1), 3] += 1; W[(k0+1)*n + (i0+1), 3] += 1
        
        # M5 = A11(B21 + B22)
        U[i0*m + j0, 4] += 1
        V[(j0+1)*p + k0, 4] += 1; V[(j0+1)*p + (k0+1), 4] += 1
        W[k0*n + i0, 4] += 1; W[(k0+1)*n + i0, 4] += 1
        
        # M6 = (A12 - A11)(B11 + B21)
        U[i0*m + (j0+1), 5] += 1; U[i0*m + j0, 5] -= 1
        V[j0*p + k0, 5] += 1; V[(j0+1)*p + k0, 5] += 1
        W[(k0+1)*n + i0, 5] += 1
        
        # M7 = (A21 - A22)(B12 + B22)
        U[(i0+1)*m + j0, 6] += 1; U[(i0+1)*m + (j0+1), 6] -= 1
        V[j0*p + (k0+1), 6] += 1; V[(j0+1)*p + (k0+1), 6] += 1
        W[k0*n + (i0+1), 6] += 1
        
        return U, V, W

    def get_rank46():
        """
        Constructs rank 46 decomposition.
        Splits B columns into 3 blocks of 2 and 1 block of 1.
        Each 2x4x2 block uses two 2x2x2 Strassen blocks (rank 14).
        The 2x4x1 block uses 4 terms (rank 4).
        Total: 3 * 14 + 4 = 46.
        """
        rank = 46
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        # Block 1: B cols 0, 1
        # A cols 0,1 and 2,3
        U1, V1, W1 = get_strassen_2x2x2(0, 0, 0)
        U[:, col:col+7] += U1
        V[:, col:col+7] += V1
        W[:, col:col+7] += W1
        col += 7

        U1, V1, W1 = get_strassen_2x2x2(0, 2, 0)
        U[:, col:col+7] += U1
        V[:, col:col+7] += V1
        W[:, col:col+7] += W1
        col += 7

        # Block 2: B cols 2, 3
        U1, V1, W1 = get_strassen_2x2x2(0, 0, 2)
        U[:, col:col+7] += U1
        V[:, col:col+7] += V1
        W[:, col:col+7] += W1
        col += 7

        U1, V1, W1 = get_strassen_2x2x2(0, 2, 2)
        U[:, col:col+7] += U1
        V[:, col:col+7] += V1
        W[:, col:col+7] += W1
        col += 7

        # Block 3: B cols 4, 5
        U1, V1, W1 = get_strassen_2x2x2(0, 0, 4)
        U[:, col:col+7] += U1
        V[:, col:col+7] += V1
        W[:, col:col+7] += W1
        col += 7

        U1, V1, W1 = get_strassen_2x2x2(0, 2, 4)
        U[:, col:col+7] += U1
        V[:, col:col+7] += V1
        W[:, col:col+7] += W1
        col += 7

        # Block 4: B col 6
        # C[i, 6] = sum_j A[i, j] B[j, 6]
        for i in range(n):
            for j in range(m):
                U[i * m + j, col] = 1
                V[j * p + 6, col] = 1
                W[6 * n + i, col] = 1
                col += 1

        assert col == rank, f"Expected rank 46, got {col}"
        return U, V, W

    # Get baseline
    U46, V46, W46 = get_rank46()
    if verify_decomp(U46, V46, W46):
        return {"U": U46.tolist(), "V": V46.tolist(), "W": W46.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
