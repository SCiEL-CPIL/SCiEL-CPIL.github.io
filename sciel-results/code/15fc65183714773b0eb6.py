"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Block-circulant embedding strategy:
    # Embed the 3x5x7 multiplication into a larger cyclic structure.
    # We use a cyclic convolution of size N where N >= m + p - 1 to avoid wraparound.
    # For m=5, p=7, we need N >= 11. Use N=12 (next power of 2 is 16, but 12 works).
    # Actually, let's use N = m + p - 1 = 11 to minimize the embedding size.
    
    # The key insight: matrix multiplication C[i,k] = sum_j A[i,j] * B[j,k]
    # can be viewed as a convolution when we embed A and B appropriately.
    
    # For each i in [0,n), the row A[i,:] and the columns of B form a convolution.
    # Specifically, if we define:
    #   a_j = A[i,j] for j in [0,m)
    #   b_k = B[k, :] ... wait, this needs more care.
    
    # Let me use a different approach: Winograd-style fast matrix multiplication.
    # For small matrices, we can use known optimal or near-optimal decompositions.
    
    # For 3x5x7, let's use a block decomposition approach:
    # Split the 5x7 matrix into blocks and use smaller multiplications.
    
    # Actually, let's use a clean construction based on the fact that we can
    # decompose the tensor using a structured approach.
    
    # The rank of the matrix multiplication tensor for n x m x p is at most n*m*p.
    # We can do better by using the structure of the problem.
    
    # Let's use a direct construction that exploits the linearity:
    # The tensor T[i,j,k] = 1 for i < n, j < m, k < p
    # We want to write T = sum_c U_c ⊗ V_c ⊗ W_c
    
    # One approach: use the fact that the multiplication can be done with
    # fewer multiplications by combining terms.
    
    # For now, let's use a verified lower-rank construction.
    # The rank is at most (n + m - 1) * (m + p - 1) / something... 
    
    # Actually, let me use a simple but effective construction:
    # Use the schoolbook decomposition but with some grouping.
    
    # The minimal rank for 3x5x7 is not known, but we can achieve rank <= 105.
    # Let's try to do better using a structured approach.
    
    # Use the following: decompose using the Kronecker product structure.
    # The tensor T can be written as:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's use a rank-1 decomposition with the following structure:
    # For each (i, j, k) triple, we have a rank-1 term, but we can group them.
    
    # Use a block-circulant approach:
    # Embed into a cyclic convolution of size N = m + p - 1 = 11
    
    N = m + p - 1  # 11
    
    # The cyclic convolution of size N can be diagonalized by the DFT.
    # The DFT matrix F of size N has entries F[a,b] = omega^(a*b) where omega = exp(2*pi*i/N)
    
    # For the cyclic convolution, we have:
    # (a * b)[k] = sum_j a[j] * b[k-j mod N]
    
    # To compute matrix multiplication, we can embed each row of A and each column of B
    # into cyclic sequences and use the convolution structure.
    
    # Let's construct the decomposition using the DFT of size N.
    # The DFT gives N orthogonal rank-1 terms for the cyclic convolution.
    
    # For each i in [0,n), we can use the N frequency components.
    # This gives a rank of n * N = 3 * 11 = 33.
    
    # But we need to verify this is exact.
    
    # Let's use a different approach: use the number-theoretic transform over a field
    # where N is invertible, or use the real DFT with careful handling.
    
    # For simplicity and correctness, let's use the schoolbook decomposition
    # but with a small optimization: combine terms where possible.
    
    # The rank is n * m * p = 105 for the schoolbook.
    # Let's see if we can do better.
    
    # Use the following construction:
    # The tensor T[i,j,k] = 1 for i < n, j < m, k < p
    # We can write this as a sum of rank-1 tensors.
    
    # One approach: use the fact that T[i,j,k] = delta(i < n) * delta(j < m) * delta(k < p)
    # This is a product of three indicator functions, which is already rank-1 in a sense,
    # but the indices are coupled in the multiplication.
    
    # Let me use a clean FFT-based construction:
    # The multiplication C = A * B can be computed using the following:
    # For each i, k: C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can write this using the DFT if we embed the vectors appropriately.
    
    # Let's use a direct construction with rank = n * N where N = m + p - 1.
    
    N = m + p - 1  # 11
    omega = np.exp(2j * np.pi / N)
    
    # Precompute DFT matrix
    F = np.array([[omega**(a*b) for b in range(N)] for a in range(N)])
    
    rank = n * N
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for f in range(N):
            # U part: U[i*m + j, col] = 1 for all j
            for j in range(m):
                U[i * m + j][col] = 1.0
            # V part: V[j*p + k, col] = conj(omega^(f*j)) / sqrt(N) for j < m, k < p
            # But we need to handle the embedding: the convolution is over the embedded space.
            for j in range(m):
                for k in range(p):
                    # The convolution term: a[j] * b[k-j mod N]
                    # In the DFT domain: A[f] * B[f] * omega^(f*k)
                    # So V should capture the B part in the DFT domain.
                    V[j * p + k][col] = np.conj(omega ** (f * j)) / np.sqrt(N)
            # W part: W[k*n + i, col] = omega^(f*k) / sqrt(N)
            for k in range(p):
                W[k * n + i][col] = np.real(omega ** (f * k)) / np.sqrt(N)
            col += 1
    
    # This may not be exact. Let me verify and use the schoolbook if needed.
    
    # For correctness, let's return the schoolbook decomposition.
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
# EVOLVE-BLOCK-END
