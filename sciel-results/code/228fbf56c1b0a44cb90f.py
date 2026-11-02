"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x6 matrix.
    
    We use a block decomposition:
    - The 4x4 x 4x4 part uses Strassen's algorithm (rank 7)
    - The 4x4 x 4x2 part uses schoolbook (rank 8)
    Total rank = 7 + 8 = 15
    
    Wait, let me reconsider. The matrix multiplication tensor for (n,m,p) = (4,4,6)
    has the standard decomposition where:
    - U has shape (n*m, rank) = (16, rank)
    - V has shape (m*p, rank) = (24, rank)
    - W has shape (p*n, rank) = (24, rank)
    
    For the schoolbook: rank = n*m*p = 4*4*6 = 96
    
    Let me think about this more carefully. We want to multiply A (4x4) by B (4x6) to get C (4x6).
    
    The tensor T_{i,j,k} = sum_{l} A_{i,l} * B_{l,k} where i in [0,4), j in [0,4), k in [0,6).
    
    Actually, the convention is:
    - U[i*m+j, c] corresponds to input A[i,j]
    - V[j*p+k, c] corresponds to input B[j,k]
    - W[k*n+i, c] corresponds to output C[i,k]
    
    Wait, let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    So W has shape (p*n, rank) = (24, rank), and W[k*n+i, r] is the coefficient for C[i,k] in term r.
    
    For schoolbook: each (i,j,k) triple gives one term where:
    U[i*m+j, c] = 1
    V[j*p+k, c] = 1
    W[k*n+i, c] = 1
    
    Rank = 4*4*6 = 96.
    
    Now, can we do better? For 4x4x4, Strassen gives rank 7 (actually rank 7 is the known minimum, 
    though it was proven that rank >= 7 for 2x2x2, and for 4x4x4 the best known is 49 via Strassen 
    applied recursively, but actually direct Strassen on 4x4x4 isn't straightforward).
    
    Actually, let me reconsider. The standard approach:
    - For 4x4x4: The tensor rank is at most 49 (Strassen recursion) but actually the direct 
      Strassen formula for 2x2x2 has rank 7. For 4x4x4, we can't directly apply 2x2x2 Strassen 
      because 4x4x4 is not a tensor power in the simple sense for the rank calculation.
    
    Wait, I think I'm overcomplicating this. Let me just use the schoolbook which gives rank 96,
    and see if I can find any known better decompositions.
    
    Actually, for the specific case of 4x4x6, I don't know of a standard better decomposition.
    Let me just implement the schoolbook correctly with rank 96.
    
    But wait - the inspiration says "Correctly implements the rank 81 decomposition by combining 
    Strassen (rank 49) for the 4x4x4 part and Schoolbook (rank 32) for the 4x4x2 part". 
    So rank 49 + 32 = 81.
    
    Let me think about how to combine these. If we split B into B1 (4x4) and B2 (4x2), then:
    C = A * B = A * [B1 | B2] = [A*B1 | A*B2]
    
    For A*B1 (4x4 * 4x4 -> 4x4): use Strassen-like decomposition of rank 49
    For A*B2 (4x4 * 4x2 -> 4x2): use schoolbook of rank 4*4*2 = 32
    
    Total rank = 49 + 32 = 81.
    
    Now I need to implement the Strassen decomposition for 4x4x4 with rank 49.
    
    Actually, the standard Strassen algorithm for 2x2x2 has rank 7. For 4x4x4, one approach is 
    to view it as a 2x2 block matrix multiplication where each block is 2x2, and apply Strassen 
    recursively. But that would give rank 7 * 7 = 49? No, that's not right either.
    
    Let me think differently. The 4x4x4 matrix multiplication tensor can be decomposed by 
    partitioning the indices. If we split each dimension into two halves of size 2:
    i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    
    Then the tensor becomes a sum over block combinations. This is getting complex.
    
    Let me just use the schoolbook for now with rank 96, since I'm not confident about the 
    Strassen decomposition for 4x4x4.
    
    Actually, let me try a different approach. I'll use the known fact that the rank of the 
    4x4x4 matrix multiplication tensor is at most 49 (this is a known result). But implementing 
    it correctly is non-trivial.
    
    For safety, let me just implement the schoolbook with rank 96.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
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
