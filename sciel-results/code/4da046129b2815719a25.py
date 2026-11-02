"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Block-circulant embedding strategy:
    # Embed the 3x5x7 multiplication into a larger structure where we can use
    # cyclic convolution structure to reduce rank.
    
    # Key insight: For each row i of A (size n x m), we multiply by B (m x p).
    # We can group the j-index and use a circulant structure.
    
    # Use N = 8 as the cyclic convolution size (>= m=5 and >= p=7)
    N = 8
    
    # The idea: For each pair (i, k), the inner product sum_j A[i,j] * B[j,k]
    # can be computed using a structured decomposition.
    
    # We'll use the following approach:
    # 1. For each i in [0, n), we have a vector a_i of length m
    # 2. For each k in [0, p), we have a column b_k of length m
    # 3. The product is sum_j a_i[j] * b_k[j]
    
    # Using the embedding into Z_N, we can write:
    # sum_j a_i[j] * b_k[j] = (1/N) * sum_f DFT(a_i)[f] * DFT(b_k)[f]
    
    # But this requires complex numbers. Instead, we use a real-valued approach
    # by separating into even and odd frequencies, or by using a different structure.
    
    # Alternative: Use the fact that we can decompose the m x p multiplication
    # using a block structure.
    
    # Let's use a simpler but effective approach:
    # Split the m=5 dimension into two parts and use a 2x2 block structure.
    
    # Actually, let's use a clean construction based on the following:
    # The rank of matrix multiplication [n,m,p] is at most n*m*p (schoolbook).
    # We can do better by using the structure of the problem.
    
    # For 3x5x7, we'll use the following construction:
    # rank = 3 * 8 = 24 using the 8-point DFT structure, but we need to
    # ensure the entries are rational/half-integer.
    
    # Use the number-theoretic transform approach with a prime modulus,
    # or use a direct construction with rational entries.
    
    # Let's use a verified construction:
    # The multiplication tensor can be decomposed using the following:
    # For each i in [0,3), and each "frequency" f in [0,8), we have a rank-1 term.
    
    # The DFT of size 8 has entries that are roots of unity. To get rational
    # entries, we can use the fact that the 8-point DFT can be expressed using
    # cos and sin of multiples of pi/4, which involve sqrt(2).
    
    # Since we need integer/half-integer entries, let's use a different approach:
    # Use the Kronecker product structure with small known decompositions.
    
    # Decompose 3x5x7 as:
    # 3x5x7 = 3 * (5x7)
    # The 5x7 multiplication can be decomposed using a 2x2 block structure.
    
    # Let's use the following construction:
    # Split A (3x5) into columns: A = [A_0, A_1, A_2, A_3, A_4]
    # Split B (5x7) into rows: B = [B_0; B_1; B_2; B_3; B_4]
    # Product = A_0*B_0 + A_1*B_1 + A_2*B_2 + A_3*B_3 + A_4*B_4
    
    # Each term A_i * B_i is an outer product of a 3-vector and a 7-vector,
    # which has rank 1 in the bilinear sense.
    
    # So the schoolbook gives rank = 3*5*7 = 105.
    
    # To reduce rank, we need to combine terms.
    
    # Use the following trick: group the 5 terms into 3 groups using
    # linear combinations.
    
    # Let's use a 2x2 block structure:
    # Split 5 = 2 + 3
    # A = [A_left (3x2), A_right (3x3)]
    # B = [B_top (2x7); B_bottom (3x7)]
    
    # Product = A_left * B_top + A_right * B_bottom
    
    # Each of these is a matrix multiplication of smaller dimensions:
    # 3x2 * 2x7 and 3x3 * 3x7
    
    # The rank of 3x2x7 is at most 3*2*7 = 42
    # The rank of 3x3x7 is at most 3*3*7 = 63
    # Total: 42 + 63 = 105 (same as before)
    
    # This doesn't help. Let's use a different approach.
    
    # Use the Strassen-like idea: combine multiple products.
    
    # For 2x2x2, Strassen gives rank 7 instead of 8.
    # We can embed our problem into a larger structure and use this.
    
    # Given the complexity, let's use a clean construction that's guaranteed
    # to work with a rank better than 105.
    
    # Use the following: the rank of [3,5,7] is at most 3*5*7 = 105.
    # We can achieve rank = 3*7 + 5*7 - 7 = 21 + 35 - 7 = 49 using a
    # different decomposition? No, that's not correct.
    
    # Let's just use the schoolbook decomposition with a small optimization:
    # We can combine some terms using the linearity of the multiplication.
    
    # For now, return the schoolbook decomposition which is guaranteed correct.
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
