"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Block-circulant embedding strategy:
    # Embed the 3x5x7 multiplication into a cyclic convolution of size N.
    # We need N >= m + p - 1 = 11 to avoid wraparound, so use N = 16.
    # The rank will be n * N = 3 * 16 = 48.
    # 
    # Key identity: For the DFT of size N,
    # sum_{f=0}^{N-1} omega^(-f*j) * omega^(f*k) = N * delta_{j,k}  (for j,k in [0,N))
    # But we need: sum_j A[i,j] * B[j,k]
    # 
    # Correct construction using the fact that:
    # A[i,:] * B[:,k]^T can be computed via cyclic convolution if we pad.
    # 
    # For each i and frequency f, define:
    #   U[i*m+j, c(i,f)] = 1
    #   V[j*p+k, c(i,f)] = omega^(-f*j)  
    #   W[k*n+i, c(i,f)] = omega^(f*k)
    #
    # Then sum_f V[j*p+k,c] * W[k*n+i,c] = sum_f omega^(-f*j) * omega^(f*k) = N*delta_{j,k}
    # This gives N*delta_{j,k}, not 1 for j=k. We need to scale by 1/N.
    #
    # But 1/N = 1/16 is not integer. We need integer/half-integer entries.
    #
    # Alternative: Use the identity directly with the Kronecker structure.
    # The tensor T[i,j,k] = 1 can be decomposed as:
    # T[i,j,k] = (1/N) * sum_f omega^(-f*j) * omega^(f*k) * N * delta_{i,i}
    # This doesn't work directly.
    #
    # Let me use a different approach: the rank is at most n*m*p = 105.
    # We can achieve rank = n * (m+p-1) = 3 * 11 = 33 using a Toeplitz structure.
    #
    # Actually, the correct block-circulant approach:
    # The multiplication C = AB where A is n x m and B is m x p.
    # For each i, C[i,:] = sum_j A[i,j] * B[j,:]
    # This is a linear combination of rows of B.
    #
    # We can use the following: represent each row of B as a vector, and
    # the coefficients are A[i,j]. The tensor decomposition is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # For the schoolbook: c = (i,j,k), U=V=W=1.
    #
    # To reduce rank, we use the fact that for fixed i, the mapping
    # (j,k) -> T[i,j,k] = 1 for all j,k has rank 1 in the (j,k) plane.
    # But we need to share structure across i.
    #
    # Use the DFT-based construction with N = 16, and scale to get integers.
    # omega = exp(2*pi*i/16) = exp(pi*i/8)
    # The entries involve cos(pi*k/8) and sin(pi*k/8), which are not rational.
    #
    # Given the requirement for integer/half-integer entries, I'll use the
    # schoolbook decomposition but verify it's correct.
    
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
