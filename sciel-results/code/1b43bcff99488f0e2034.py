"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Block-circulant embedding strategy:
    # Embed the 3x5x7 multiplication into a cyclic convolution structure of size N=8.
    # The key insight: for each output row i, the computation C[i,k] = sum_j A[i,j] * B[j,k]
    # can be related to a cyclic convolution when properly embedded.
    # 
    # We use the DFT of size N=8 to diagonalize the circulant structure.
    # For each frequency f in [0,8) and each i in [0,3), we create a rank-1 term.
    # 
    # The construction ensures that the bilinear form computes the correct matrix product.
    
    N = 8  # DFT size, >= max(m, p) = 7
    omega = np.exp(2j * np.pi / N)
    
    # Precompute DFT values
    # F[a, b] = omega^(a*b)
    
    rank = n * N  # 3 * 8 = 24
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for f in range(N):
            # For this (i, f) pair, construct the rank-1 factor
            # U[i*m + j, col] = 1 for all j
            for j in range(m):
                U[i * m + j][col] = 1.0
            
            # V[j*p + k, col] = conj(omega^(f*j)) for all j, k
            for j in range(m):
                val = np.conj(omega ** (f * j))
                for k in range(p):
                    V[j * p + k][col] = val.real if abs(val.imag) < 1e-10 else val.real / np.sqrt(2)
            
            # W[k*n + i, col] = omega^(f*k) / N for all k
            for k in range(p):
                val = omega ** (f * k)
                W[k * n + i][col] = val.real / N if abs(val.imag) < 1e-10 else val.real / (N * np.sqrt(2))
            
            col += 1
    
    # Verify: this construction may not be exact for matrix multiplication.
    # The DFT-based approach computes a cyclic convolution, not standard matrix multiplication.
    # 
    # For correctness, fall back to schoolbook if the above is not verified.
    # Given the complexity of verifying the FFT-based construction, use schoolbook.
    
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
