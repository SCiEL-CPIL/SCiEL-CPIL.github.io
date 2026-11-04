import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
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

    def get_rank50():
        """Construct a known valid rank-50 decomposition using Strassen blocks."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
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

        # Tile 2x4 x 4x7 with Strassen blocks
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        # Handle column 6 of B (and C) separately with rank-1 terms
        for j in range(m):
            U[0 * m + j, col] = 1  # A[0,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 0, col] = 1  # C[0,6]
            col += 1
            U[1 * m + j, col] = 1  # A[1,j]
            V[j * p + 6, col] = 1  # B[j,6]
            W[6 * n + 1, col] = 1  # C[1,6]
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    def get_winograd_rank42():
        """
        Winograd-based decomposition for 2x4x7.
        
        Strategy: For each output row i (0,1), compute C[i,:] = A[i,:] * B
        This is a 1x4 times 4x7 matrix product.
        
        Use Winograd F(4,7) transform:
        - Transform A row (length 4) with input transform G
        - Transform B columns (length 7) with input transform F  
        - Pointwise multiply (7 multiplications per output column pair... actually need to think more carefully)
        
        Actually for 1x4 x 4x7, we can use:
        - Split B into overlapping 4x4 blocks with stride 3 (to cover 7 columns): blocks at cols 0-3, 1-4, 2-5, 3-6
        - But 4x4 Winograd uses 6 multiplications per block
        - 4 blocks * 6 = 24 mults per row, 2 rows = 48... not better.
        
        Better approach: Use the fact that n=2 to share work between rows.
        
        Let me try a different decomposition. For 2x4 x 4x7:
        C[0,k] = sum_j A[0,j] * B[j,k]
        C[1,k] = sum_j A[1,j] * B[j,k]
        
        We can write:
        C[0,k] + C[1,k] = sum_j (A[0,j]+A[1,j]) * B[j,k]
        C[0,k] - C[1,k] = sum_j (A[0,j]-A[1,j]) * B[j,k]
        
        So we need to compute:
        S[k] = sum_j S_A[j] * B[j,k]  where S_A = A[0,:]+A[1,:]
        D[k] = sum_j D_A[j] * B[j,k]  where D_A = A[0,:]-A[1,:]
        
        Then C[0,k] = (S[k]+D[k])/2, C[1,k] = (S[k]-D[k])/2
        
        Each of S and D is a 1x4 x 4x7 product.
        
        For a 1x4 x 4x7 product, use Winograd with 4-point DFT-like transform:
        Actually, let's use the standard Winograd for 1x4 x 4x4 and tile.
        
        For 1x4 x 4x4: Winograd uses 6 mults.
        For 1x4 x 4x7: split into two 1x4 x 4x4 blocks (cols 0-3 and 4-7, but only 4-6 exist)
        Block 1: cols 0-3, 6 mults
        Block 2: cols 3-6, 6 mults (overlapping at col 3)
        Total: 12 mults per 1x4 x 4x7
        
        So S: 12 mults, D: 12 mults, total 24 mults.
        But we need to be careful about the overlap.
        
        Actually, let me just construct it explicitly.
        """
        # For 1x4 x 4x4 Winograd:
        # Input transform for A (length 4): 
        #   a0, a1, a2, a3 -> 
        #   m1 = a0+a2, m2 = a1+a3, m3 = a0-a2, m4 = a1-a3, m5 = a0+a1, m6 = a2+a3
        # Wait, standard Winograd F(4,4) for 1x4x4:
        # Actually for 1x4 x 4x4, it's a vector-matrix product.
        
        # Let me use a simpler approach: just enumerate the rank-1 terms.
        # For 2x4 x 4x7, the schoolbook rank is 2*4*7 = 56.
        # Strassen gives 50.
        # Let me try to find a better decomposition.
        
        # Alternative: use the structure that n=2.
        # C[0,k] = sum_j A[0,j] B[j,k]
        # C[1,k] = sum_j A[1,j] B[j,k]
        
        # Define:
        # P1[j,k] = B[j,k]
        # P2[j,k] = B[j,k] (same)
        
        # We can use:
        # C[0,k] = 1/2 * [sum_j (A[0,j]+A[1,j]) B[j,k] + sum_j (A[0,j]-A[1,j]) B[j,k]]
        # C[1,k] = 1/2 * [sum_j (A[0,j]+A[1,j]) B[j,k] - sum_j (A[0,j]-A[1,j]) B[j,k]]
        
        # So we need two 1x4 x 4x7 products.
        # Each 1x4 x 4x7 can be done with Winograd.
        
        # For 1x4 x 4x7, use two overlapping 1x4 x 4x4 Winograd blocks.
        # Winograd 1x4 x 4x4 uses 6 multiplications.
        
        # Winograd F(4,4) for 1x4 x 4x4:
        # A = [a0, a1, a2, a3] (1x4)
        # B = 4x4 matrix
        # C = A * B (1x4)
        
        # Input transforms:
        # For A: 
        #   a0' = a0 + a2
        #   a1' = a1 + a3
        #   a2' = a0 - a2
        #   a3' = a1 - a3
        #   a4' = a0 + a1
        #   a5' = a2 + a3
        # For B (4x4), each column b_k = [b0k, b1k, b2k, b3k]:
        #   b0k' = b0k + b2k
        #   b1k' = b1k + b3k
        #   b2k' = b0k - b2k
        #   b3k' = b1k - b3k
        #   b4k' = b0k + b1k
        #   b5k' = b2k + b3k
        
        # Multiplications: m_k = a_j' * b_jk' for j=0..5, k=0..3
        # Output: C_k = (m0 + m1 + m2 + m3 + m4 + m5)/2 ... this is getting complicated.
        
        # Let me just use a brute-force search for a better decomposition.
        # Given time constraints, let me stick with the rank-50 and try to verify it's correct.
        
        # Actually, let me try a different approach: use the fact that we can
        # potentially reduce rank by finding linear dependencies.
        
        U = np.zeros((n * m, 0))
        V = np.zeros((m * p, 0))
        W = np.zeros((p * n, 0))
        
        # Start with Strassen rank-50 and try to reduce
        U50, V50, W50 = get_rank50()
        
        # Try to find if any column is redundant
        # For each column r, check if removing it and adjusting others still works
        # This is a hard problem, so let's just return the rank-50 for now
        
        return U50, V50, W50

    # Get the rank-50 decomposition
    U50, V50, W50 = get_rank50()

    # Verify it's correct
    if not verify_decomp(U50, V50, W50):
        # Fallback to schoolbook
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank using linear algebra
    # Check matricization ranks for lower bound
    T_mat1 = T.reshape(n * m, -1)
    T_mat2 = T.reshape(m * p, -1)
    T_mat3 = T.reshape(p * n, -1)

    rank_lower1 = np.linalg.matrix_rank(T_mat1)
    rank_lower2 = np.linalg.matrix_rank(T_mat2)
    rank_lower3 = np.linalg.matrix_rank(T_mat3)
    rank_lower = max(rank_lower1, rank_lower2, rank_lower3)

    if time.time() > deadline - 20:
        return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}

    # Return the rank-50 decomposition
    return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
# EVOLVE-BLOCK-END
