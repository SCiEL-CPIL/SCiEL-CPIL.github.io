"""
Winograd-style decomposition for 4x5 by 5x6 matrix multiplication.
"""
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = 4, 5, 6
    
    # Partition:
    # A (4x5) -> A1 (4x4) [cols 0-3], A2 (4x1) [col 4]
    # B (5x6) -> B1 (4x6) [rows 0-3], B2 (1x6) [row 4]
    # C (4x6) = A1 @ B1 + A2 @ B2
    
    # For A1 (4x4) @ B1 (4x6), we split B1 into B1a (4x4) and B1b (4x2)
    # A1 @ B1a: Winograd 4x4x4 -> rank 16
    # A1 @ B1b: schoolbook 4x4x2 -> rank 32
    # A2 @ B2: schoolbook 4x1x6 -> rank 24
    # Total rank: 16 + 32 + 24 = 72
    
    # 4-point Winograd/FFT matrices
    W4 = np.fft.fft(np.eye(4))  # 4-point DFT matrix
    W4_H = np.fft.ifft(np.eye(4))  # 4-point IDFT matrix (with 1/4 scaling)
    
    # For Winograd 4x4x4 matrix multiplication:
    # C = A @ B where A, B, C are 4x4
    # A_win = W4 @ A.T  (shape 4x4)
    # B_win = B @ W4_H (shape 4x4)
    # C_win = A_win * B_win (elementwise, shape 4x4)
    # C = (W4_H.T @ C_win).T  (shape 4x4)
    
    rank = 16 + 32 + 24
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    
    # Part 1: A1 (4x4) @ B1a (4x4) using Winograd, rank 16
    # A1 is A[0:4, 0:4], B1a is B[0:4, 0:4]
    # C1a[0:4, 0:4] = A1 @ B1a
    for i in range(4):
        for j in range(4):
            # A_win[i, j'] = sum_j W4[i, j] * A[j, j'] = sum_j W4[i, j] * A[j, j']
            # Since A1[j, j'] = A[j, j'], U entry for (i, j) in A_win domain
            for j_win in range(4):
                for j_prime in range(4):
                    # Contribution to U[i*m + j_prime] from Winograd
                    # Actually, let me think about this more carefully.
                    # In the Winograd domain:
                    # A_win[i, k] = sum_j W4[i, j] * A[j, k]  (i is frequency, k is output col of A row)
                    # Wait, A is 4x4, A[j, k] where j is row, k is col
                    # A_win[i, k] = sum_j W4[i, j] * A[j, k]
                    # So U[i * 4 + k] += W4[i, j] for the j-th row of A
                    # Actually U is indexed by [i*m + j] where i is output row (0-3), j is input col (0-4)
                    # For A1, i is 0-3, j is 0-3
                    # U[i*5 + j] corresponds to A[i, j]
                    
                    # In Winograd: A_win[f, k] = sum_j W4[f, j] * A[j, k]
                    # So for fixed f and k, U[f*5 + j] gets contribution W4[f, j]
                    # And the Winograd multiplication index is (f, k) where f in 0-3, k in 0-3
                    pass
    
    # Let me use a cleaner approach with explicit Winograd matrices.
    # The 4x4x4 Winograd scheme:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # Winograd: C = F_H @ ((F @ A.T) * (B @ F_H)) .T
    # Where F is the 4-point DFT matrix, F_H is the 4-point IDFT matrix
    # Let's verify: (F @ A.T)[f, k] = sum_i F[f, i] * A.T[i, k] = sum_i F[f, i] * A[k, i]
    # (B @ F_H)[j, k] = sum_k' B[j, k'] * F_H[k', k]
    # Elementwise product: P[f, k] = (F @ A.T)[f, k] * (B @ F_H)[f, k]
    # C = (F_H.T @ P).T => C[i, k] = sum_f F_H.T[f, i] * P[f, k] = sum_f F_H[f, i] * P[f, k]
    # = sum_f F_H[f, i] * (F @ A.T)[f, k] * (B @ F_H)[f, k]
    
    # This is getting complex. Let me just use the direct construction.
    
    # For the 4x4x4 Winograd core, the rank is 16 (4 frequency points x 4 output positions)
    # Let me construct it directly.
    
    # Actually, let me just do schoolbook for everything to be safe and correct.
    # The current score is 1.0 with rank 120. Let me try to improve.
    
    # Known: mu(4,4,4) <= 49 (best known upper bound)
    # For 4x5x6, a reasonable upper bound is less than 120.
    
    # Let me use a block approach:
    # Split A into A1 (4x3) and A2 (4x2)
    # Split B into B1 (3x6) and B2 (2x6)
    # C = A1 @ B1 + A2 @ B2
    # Schoolbook: 4*3*6 + 4*2*6 = 72 + 48 = 120. Same.
    
    # Let me try a different partition:
    # A (4x5) -> A1 (4x2), A2 (4x3)
    # B (5x6) -> B1 (2x6), B2 (3x6)
    # C = A1 @ B1 + A2 @ B2
    # Schoolbook: 4*2*6 + 4*3*6 = 48 + 72 = 120. Same.
    
    # The issue is that schoolbook always gives n*m*p for the total.
    # To get a better rank, we need a non-schoolbook decomposition.
    
    # Let me implement the Winograd 4x4x4 core properly.
    # The 4-point FFT/Winograd for matrix multiplication:
    # For C = A @ B, A, B, C are 4x4:
    # Step 1: A' = F4 @ A.T (4x4)
    # Step 2: B' = B @ F4_H (4x4)
    # Step 3: C' = A' * B' (elementwise, 4x4)
    # Step 4: C = (F4_H.T @ C').T (4x4)
    
    # This uses 16 multiplications (4x4 elementwise product).
    # The rank is 16 for the 4x4x4 core.
    
    # Now for 4x4 @ 4x6:
    # Split B into B1 (4x4) and B2 (4x2)
    # C1 = A @ B1: Winograd, rank 16
    # C2 = A @ B2: schoolbook, rank 4*4*2 = 32
    # Total for 4x4 @ 4x6: 16 + 32 = 48
    
    # For 4x1 @ 1x6: schoolbook, rank 4*1*6 = 24
    
    # Total rank: 48 + 24 = 72
    
    # Let me implement this.
    
    rank = 72
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    
    # Part 1: A1 (4x4) @ B1a (4x4) using Winograd, rank 16
    # A1 = A[0:4, 0:4], B1a = B[0:4, 0:4], C1a = C[0:4, 0:4]
    F4 = np.fft.fft(np.eye(4))
    F4_H = np.fft.ifft(np.eye(4))
    
    # A'[f, i] = sum_j F4[f, j] * A1[j, i]
    # B'[j, k] = sum_l B1a[j, l] * F4_H[l, k]
    # C'[f, k] = A'[f, i] * B'[f, k] ... wait, the indices don't match.
    
    # Let me re-derive. For C = A @ B:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # Winograd: 
    # A_win[f, i] = sum_j F4[f, j] * A[i, j]  (transform rows of A)
    # B_win[j, k] = sum_f F4_H[f, k] * B[j, f]  (transform cols of B)
    # Hmm, this doesn't seem right either.
    
    # Let me use a verified construction. The standard FFT-based matrix multiplication:
    # C = A @ B
    # C = F^{-1} @ ( (F @ A) * (B @ F^{-1}) ) where * is elementwise
    # But this only works if A, B, C are square and the FFT is over the right dimension.
    
    # Actually, for 4x4x4, the Winograd/FFT approach is:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # This is a convolution in j. For each (i, k), it's a dot product, not a convolution.
    # So FFT doesn't directly apply to matrix multiplication in the way I was thinking.
    
    # The rank-16 claim for 4x4x4 is incorrect. The schoolbook rank is 64, and the best known
    # upper bound for mu(4,4,4) is around 49, not 16.
    
    # Let me just do schoolbook for everything. Rank = 120.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
