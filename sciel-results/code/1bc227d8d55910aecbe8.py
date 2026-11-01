"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    WFT over Z_2 x Z_7 for 2x4x7 matrix multiplication.
    
    Key idea: Use the group G = Z_2 x Z_7 (order 14).
    Embed the 4-dimensional middle dimension into Z_7 by zero-padding.
    The spectral decomposition over Z_7 gives 7 Fourier modes, and Z_2 gives 2 modes.
    
    For exact integer/half-integer entries, we use the real form of the WFT.
    The 7th roots of unity can be handled using the fact that we only need
    the structure to be exact, and we can use the minimal polynomial approach.
    
    However, a cleaner approach: Since m=4 < p=7, we can use the fact that
    the multiplication tensor has rank at most p * (number of non-zero modes in Z_2 direction)
    times something related to the support in Z_7.
    
    Actually, let's use a more direct construction:
    
    For C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    We can use the WFT over Z_7 for the j-index (with zero padding to length 7).
    Let A_padded[i, j] = A[i,j] for j in 0..3, 0 for j in 4..6
    Let B_padded[j, k] = B[j,k] for j in 0..3, 0 for j in 4..6
    
    Then C[i,k] = sum_{j=0}^{6} A_padded[i,j] * B_padded[j,k]
    
    Using WFT over Z_7:
    A_hat[i, v] = sum_{j=0}^{6} A_padded[i,j] * omega_7^{v*j}
    B_hat[v, k] = sum_{j=0}^{6} B_padded[j,k] * omega_7^{v*j}  (conjugate for inverse)
    
    Then C[i,k] = (1/7) * sum_{v=0}^{6} A_hat[i,v] * B_hat[v,k]
    
    This gives rank 7 for the Z_7 part. Combined with the n=2 dimension,
    the total rank would be 2*7 = 14 if we can separate the dimensions cleanly.
    
    But we also need to handle the Z_2 structure. Let's think about this more carefully.
    
    The full group is Z_2 x Z_7. The characters are indexed by (u, v) where u in {0,1}, v in {0,...,6}.
    
    For the matrix multiplication tensor, we need to embed:
    - A[i,j] where i in Z_2, j in {0,...,3} (padded to Z_7)
    - B[j,k] where j in {0,...,3} (padded to Z_7), k in {0,...,6}
    
    The tensor T_{i,j,k} = delta_{j,j} for the appropriate indices.
    
    In the WFT framework, the rank equals the number of non-zero Fourier coefficients
    of the tensor on the group.
    
    Let me construct this explicitly using the real representation.
    
    For Z_7, the real Fourier basis consists of:
    - v=0: constant (1)
    - v=1,6: cos(2*pi*1*j/7), sin(2*pi*1*j/7)
    - v=2,5: cos(2*pi*2*j/7), sin(2*pi*2*j/7)  
    - v=3,4: cos(2*pi*3*j/7), sin(2*pi*3*j/7)
    
    But these involve irrational numbers (cos and sin of multiples of 2*pi/7).
    For exact decomposition with rational entries, we need a different approach.
    
    Alternative: Use the fact that we can work over the cyclotomic field Q(zeta_7).
    But the problem requires integer/half-integer entries.
    
    Let me reconsider. The schoolbook rank is 56. Can we do better?
    
    Known results:
    - For (2 x m) * (m x p), the rank is at most 2*p + m - 1 (from certain constructions)
    - For m=4, p=7: 2*7 + 4 - 1 = 17? That seems too optimistic.
    
    Actually, let's use a simpler but valid improvement:
    Split the 4 middle dimension into two 2-dimensional blocks.
    C = A1*B1 + A2*B2 where A1,A2 are 2x2 and B1,B2 are 2x7.
    
    For each 2x2 * 2x7 multiplication, we can use a rank-14 decomposition.
    But we can do better using the structure.
    
    For 2x2 * 2x7: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    This is a sum of 2*2*7 = 28 rank-1 terms in schoolbook.
    
    But we can use: Let S[i,k] = (A[i,0]+A[i,1])*(B[0,k]+B[1,k])
    Let D[i,k] = (A[i,0]-A[i,1])*(B[0,k]-B[1,k])
    Then A[i,0]*B[0,k] + A[i,1]*B[1,k] = (S[i,k] + D[i,k])/2
    And A[i,0]*B[1,k] + A[i,1]*B[0,k] = (S[i,k] - D[i,k])/2
    
    This doesn't directly help for our case since we need specific pairings.
    
    Let me just implement a clean schoolbook decomposition and verify correctness.
    The rank is 56.
    
    """
    n, m, p = DIMENSIONS
    
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
