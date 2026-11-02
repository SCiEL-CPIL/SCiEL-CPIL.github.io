"""
Block-circulant embedding for 3x5x7 matrix multiplication.
Strategy: embed the 5x7 convolution into a cyclic convolution of size 8,
decompose the circulant structure using DFT, and project back.
"""

import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    N = 8  # Cyclic convolution size (>= m and >= p)
    
    # We'll use a DFT-based decomposition of the circulant convolution.
    # The key idea: for each frequency f in [0, N), we can diagonalize
    # the cyclic convolution. This gives N rank-1 terms per "slice" in the
    # i-dimension, but we need to be careful about the exact decomposition.
    
    # Let's use a more direct approach: the cyclic convolution of size N
    # can be decomposed into N rank-1 terms using the DFT.
    # For each i in [0, n), we have a convolution of A[i, :] (length m)
    # with B[:, k] (length p) to get C[i, k].
    
    # The cyclic convolution (A * B)[k] = sum_j A[j] * B[k-j mod N]
    # can be written as sum_f (1/N) * (sum_j A[j] * omega^(-f*j)) * (sum_l B[l] * omega^(f*l)) * omega^(f*k)
    
    # But we need exact decomposition over reals/rationals, so we use the
    # real DFT (cosine/sine) or work with the algebraic structure.
    
    # Actually, let's use a cleaner construction:
    # The rank of the cyclic convolution tensor of size N is exactly N
    # (it's a circulant tensor). So for each i, we get N terms.
    # Total rank = n * N = 3 * 8 = 24.
    
    # But we need to verify this is exact for the truncated convolution.
    # The issue: the actual multiplication is NOT a full cyclic convolution;
    # it's a linear convolution truncated to valid indices.
    
    # For exactness, let's use a different approach:
    # We can use the fact that the tensor T[i,j,k] = 1 for i<3, j<5, k<7
    # can be decomposed using the "border rank" or structured decomposition.
    
    # Let me use a verified construction based on the following:
    # The multiplication can be embedded in a larger space where it becomes
    # a product of circulant matrices, and we use the eigen-decomposition.
    
    # For simplicity and correctness, let's use the schoolbook but with
    # a small optimization: use Strassen for 2x2x2 sub-blocks.
    
    # Actually, let me try the FFT-based construction properly:
    rank = n * N  # 24
    omega = np.exp(2j * np.pi / N)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for f in range(N):
            # U[i*m + j, col] = 1 for all j in [0, m)
            for j in range(m):
                U[i * m + j][col] = 1.0
            
            # V[j*p + k, col] = conj(omega^(f*j)) * (1/sqrt(N)) for valid (j,k)
            # W[k*n + i, col] = omega^(f*k) * (1/sqrt(N)) for valid (k)
            # But this gives a convolution, not exact multiplication.
            
            # The problem: this construction computes (A * B)_cyclic[k],
            # not (A * B)[k]. For exact multiplication, we need the linear
            # convolution, which is not exactly captured by a single cyclic
            # convolution of size N unless N >= m + p - 1.
            
            # For m=5, p=7, we need N >= 11 for exact linear convolution.
            # Let's use N = 16 (next power of 2 >= 11).
            
            # Actually, let me reconsider. The schoolbook is safe.
            # Let me just return the schoolbook decomposition.
            
            col += 1
    
    # The FFT approach above doesn't give exact multiplication for the
    # truncated case. Let me use the schoolbook decomposition which is
    # guaranteed to be exact.
    
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
