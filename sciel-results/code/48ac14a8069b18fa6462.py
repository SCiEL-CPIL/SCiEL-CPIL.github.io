"""Matrix multiplication 6x6x6 tensor decomposition - rank minimization."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # We use a block-structured decomposition.
    # The 6x6 matrices are partitioned into 2x2 blocks of size 3x3.
    # There are 4 block-positions for A, 4 for B, and 4 for C.
    # We compute C = A * B where A, B are 2x2 block matrices of 3x3 blocks.
    # C_ij = sum_k A_ik * B_kj
    # We can use Strassen on the 2x2 block level, but each "block multiply"
    # A_ik * B_kj is a 3x3 * 3x3 multiplication.
    # 
    # Alternatively, we can partition into 3x3 blocks of size 2x2.
    # Then there are 9 block positions, and we need to handle 3x3 block matrix mult.
    # That's more complex.
    #
    # Let's stick with 2x2 blocks of size 3x3.
    # Strassen for 2x2 blocks requires 7 block multiplications.
    # Each block multiplication is 3x3 * 3x3 = 27 scalar multiplications.
    # Total: 7 * 27 = 189.
    #
    # But wait, we can do better by using a rank decomposition for the 3x3 case.
    # The rank of 3x3 matrix multiplication is at most 27 (trivial) but can be lower.
    # Actually, for 3x3, the best known is 23 (Strassen-like) but let's check.
    # 
    # Actually, let's think differently. We can use a direct rank decomposition
    # that exploits the structure.
    #
    # The matrix multiplication tensor T_{i,j,k} = delta_{j,k} (in appropriate indices)
    # has rank at most n*m*p = 216.
    #
    # Using Strassen on 2x2 blocks with 3x3 sub-blocks:
    # We need 7 multiplications of 3x3 matrices.
    # If we use the trivial 3x3 decomposition (rank 27), total rank = 7*27 = 189.
    #
    # Can we do better? What if we use a shared decomposition?
    # The 7 Strassen combinations involve sums/differences of blocks.
    # We could potentially share some rank terms.
    #
    # Actually, a cleaner approach: use the fact that 6 = 2 * 3.
    # We can use a Kronecker product structure.
    # The tensor for 6x6x6 mult can be related to 2x2x2 and 3x3x3.
    # But the rank of a Kronecker product is the product of ranks.
    # rank(T_6) <= rank(T_2) * rank(T_3) = 7 * 27 = 189.
    #
    # So 189 is achievable. Can we go lower?
    # 
    # Let's try a different block size: 3x3 blocks of 2x2.
    # Then we have a 3x3 block matrix multiplication.
    # We can use a rank decomposition for 3x3 block mult.
    # The 3x3 block mult has 9*9 = 81 block entries, each a 2x2 mult.
    # If we use Strassen for the 2x2 blocks, each block mult has rank 7.
    # But the 3x3 structure means we need to handle the 3x3 coefficient structure.
    #
    # Actually, the standard result is that the rank of n×n matrix multiplication
    # for n=6 is at most 189 using the Strassen-based approach.
    #
    # Let's implement the 2x2 Strassen with 3x3 blocks, using trivial 3x3 mult.
    
    block_size = 3
    blocks_per_dim = 2
    
    # Strassen patterns for 2x2 matrices
    # Each pattern is (U_coeffs, V_coeffs, W_coeffs)
    # where coeffs maps (i,j) -> coefficient
    strassen_patterns = [
        # M1 = (A00 + A11)(B00 + B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10 + A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00(B01 - B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11(B10 - B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00 + A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10 - A00)(B00 + B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01 - A11)(B10 + B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    rank = (blocks_per_dim ** 3) * block_size * block_size
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    # For each scalar position in the 3x3 block
                    for si in range(block_size):
                        for sj in range(block_size):
                            # U: contributes to row I*bs+si, col J*bs+sj
                            for (pi, pj), val in u_coeffs.items():
                                if pi == I and pj == J:
                                    r = I * block_size + si
                                    c = J * block_size + sj
                                    U[r * m + c][col_idx] += val
                            # V: contributes to row J*bs+si, col K*bs+sj
                            for (pi, pj), val in v_coeffs.items():
                                if pi == J and pj == K:
                                    r = J * block_size + si
                                    c = K * block_size + sj
                                    V[r * p + c][col_idx] += val
                            # W: contributes to row K*bs+sj, col I*bs+si
                            for (pi, pj), val in w_coeffs.items():
                                if pi == K and pj == I:
                                    k = K * block_size + sj
                                    i = I * block_size + si
                                    W[k * n + i][col_idx] += val
                            col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
