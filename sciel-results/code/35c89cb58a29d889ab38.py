"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Block-circulant embedding strategy:
    # Embed the 5x7 multiplication into an 8x8 cyclic convolution.
    # For each row i of A, decompose sum_j A[i,j]*B[j,k] using 8-point DFT.
    # 
    # Key idea: The cyclic convolution of length 8 can be diagonalized by DFT.
    # We use the fact that for the "valid" entries (j<5, k<7), we can recover
    # the linear convolution from the cyclic convolution by embedding.
    #
    # Construction: For each i in [0,3) and each frequency f in [0,8):
    #   The rank-1 term contributes to T[i,j,k] via:
    #   U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
    #
    # Using DFT: omega = exp(2*pi*i/8)
    # For cyclic convolution: (a * b)_k = sum_j a_j * b_{k-j mod N}
    # We want: sum_j A[i,j] * B[j,k] for k < 7
    #
    # Embed B's columns into length-8 vectors: b_j[k] for k in [0,8)
    # The cyclic convolution gives sum_j a_j * b_{k-j mod 8}
    # For k < 7 and j < 5, k-j mod 8 = k-j if k>=j, else k-j+8
    # We need k-j in [0,7) i.e. j <= k < j+5, but we want all j,k with j<5,k<7
    #
    # This doesn't directly work. Let me use a different valid construction.
    #
    # Alternative: Use the tensor decomposition based on the observation that
    # T[i,j,k] = 1 for i<3,j<5,k<7 can be written as:
    # T[i,j,k] = sum_f U[i,j,f] * V[j,k,f] * W[k,i,f]
    # where we use the DFT basis for the j-k interaction.
    #
    # Valid construction using polynomial interpolation / FFT:
    # Represent the multiplication as evaluating a polynomial at roots of unity.
    #
    # For each i, define polynomial P_i(x) = sum_j A[i,j] x^j (degree < 5)
    # And Q(x) = sum_k B[j,k] ... no, B is a matrix.
    #
    # Let me use the standard approach: the rank of 3x5x7 is at most 3*5*7=105.
    # I'll use a verified lower-rank construction.
    #
    # Known: For m x p multiplication, rank is at most (m+p-1)*something via FFT.
    # Actually, the bilinear rank of matrix multiplication [n,m,p] <= n*m*p.
    #
    # Let me use a clean construction: embed into 8x8 and use DFT.
    # The 8x8 cyclic convolution tensor has rank 8 (diagonalized by DFT).
    # We project back to get the 5x7 part.
    
    N = 8  # Embedding size (power of 2, >= max(m,p)=7)
    omega = 2j * np.pi / N  # 2*pi*i/8
    
    # DFT matrix entries: F[a,b] = exp(2*pi*i*a*b/N)
    # We'll use rational approximations or exact algebraic numbers.
    # For exactness, use the minimal polynomial approach or just use floats
    # and verify they're close to rationals.
    
    # Precompute DFT values
    # omega^0 = 1
    # omega^1 = exp(i*pi/4) = (1+i)/sqrt(2)
    # omega^2 = i
    # omega^3 = (-1+i)/sqrt(2)
    # omega^4 = -1
    # omega^5 = (-1-i)/sqrt(2)
    # omega^6 = -i
    # omega^7 = (1-i)/sqrt(2)
    
    # These involve sqrt(2), so entries will be in Q(sqrt(2)).
    # The problem says integer/half-integer entries. sqrt(2)/2 is not half-integer.
    # 
    # I need a different approach that uses only rational numbers.
    #
    # Use a prime modulus or a different structure.
    #
    # Alternative: Use the fact that we can decompose using the Vandermonde
    # matrix with rational points. Use N points x_0,...,x_{N-1} that are rational.
    #
    # For the 5x7 case, I can use 12 points (or more) and interpolate.
    # But this increases rank.
    #
    # Simplest correct approach: schoolbook with rank 105, but let me try
    # a block decomposition.
    #
    # Split 5 = 2+3, 7 = 3+4. Use 2x3 and 3x4 multiplications.
    # rank(2x3) <= 6, rank(3x4) <= 12.
    # But combining blocks doesn't directly help for the full 3x5x7.
    #
    # Let me just use the schoolbook decomposition which is guaranteed correct.
    
    rank = n * m * p  # 105
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
