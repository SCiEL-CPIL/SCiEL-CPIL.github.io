import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 (Strassen) + Rank 48 for 4x4x3 (Schoolbook) = 97
    rank = 97
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. Construct 4x4x4 Strassen Decomposition (Rank 49)
    # Indices for 2x2 Strassen terms
    # u, v, w are vectors of length 4 for 2x2 matrices
    # 0:00, 1:01, 2:10, 3:11
    s2_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    s2_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    s2_w = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    col = 0
    for i_block in range(7):
        for j_block in range(7):
            # This term corresponds to the product of the i_block-th and j_block-th Strassen terms
            # at the block level, combined with the full 2x2 multiplication at the local level?
            # No, the 4x4x4 tensor is the tensor product of two 2x2x2 tensors.
            # The decomposition is the set of 49 terms formed by the Kronecker products of the 7 terms.
            
            u_block = s2_u[i_block]
            v_block = s2_v[i_block]
            w_block = s2_w[i_block]
            
            u_local = s2_u[j_block]
            v_local = s2_v[j_block]
            w_local = s2_w[j_block]
            
            # We need to map the 16-length vectors to U, V, W
            # U index: i_global * 4 + j_global
            # Block index: I, J (0..1)
            # Local index: i_local, j_local (0..1)
            # i_global = 2*I + i_local
            # j_global = 2*J + j_local
            
            # The Kronecker product u_block \otimes u_local gives a vector of length 16.
            # Index in Kronecker: (I*2 + J)*4 + (i_local*2 + j_local) = 8I + 4J + 2i_local + j_local
            # Global Index: (2I + i_local)*4 + (2J + j_local) = 8I + 4i_local + 4J + 2J + j_local? No.
            # Global Index = 4*(2I + i_local) + 2J + j_local = 8I + 4i_local + 2J + j_local.
            
            # The Kronecker order is (Block Index, Local Index).
            # Block Index is 2x2 row-major: 2*I + J.
            # Local Index is 2x2 row-major: 2*i_local + j_local.
            # Kronecker Index = (2*I + J)*4 + (2*i_local + j_local) = 8I + 4J + 2i_local + j_local.
            
            # We need to fill U[global_idx] with the coefficient from the Kronecker product.
            # Coeff = u_block[2*I + J] * u_local[2*i_local + j_local]
            
            for I in range(2):
                for J in range(2):
                    for i_l in range(2):
                        for j_l in range(2):
                            coeff = u_block[2*I + J] * u_local[2*i_l + j_l]
                            if coeff != 0:
                                i_glob = 2*I + i_l
                                j_glob = 2*J + j_l
                                U[i_glob * 4 + j_glob][col] += coeff
                                
                            coeff_v = v_block[2*I + J] * v_local[2*i_l + j_l]
                            if coeff_v != 0:
                                # V index: j_global * 7 + k_global
                                # k_global is the column index of the 4x4 part, so 0..3
                                # j_glob is row of B block?
                                # Wait, V is for B. B is 4x4.
                                # j_glob is row of B, k_glob is col of B.
                                # In the tensor product, the second factor is the local multiplication.
                                # So j_glob corresponds to the local row, and the block row is I?
                                # No.
                                # The tensor product structure:
                                # T = T_block \otimes T_local.
                                # T_block acts on the block indices.
                                # T_local acts on the local indices.
                                # The indices of U, V, W in the 4x4x4 case are (I, i_l), (J, j_l) etc?
                                # Let's re-verify the indices.
                                # A is 4x4. A[I, J] is a 2x2 block.
                                # Entry A[2I+i_l, 2J+j_l].
                                # U index is (i, j) of A.
                                # So U index is (2I+i_l, 2J+j_l).
                                # Coefficient is u_block[I, J] * u_local[i_l, j_l].
                                # This matches the loop above.
                                
                                # Now for V. V index is (j, k) of B.
                                # B is 4x4. B[2I+i_l, 2J+j_l]?
                                # No. The multiplication is A * B.
                                # A indices (i, j). B indices (j, k).
                                # In the block decomposition:
                                # A blocks A_{IJ}. B blocks B_{JK}.
                                # The block multiplication is A_{IJ} * B_{JK} summed over J.
                                # The Strassen algorithm on blocks computes linear combinations of A blocks and B blocks.
                                # So the "block" part of the tensor decomposition handles the linear combinations of the block indices.
                                # The "local" part handles the multiplication of the 2x2 blocks.
                                
                                # For V (which is B):
                                # The block index for B is (J, K)?
                                # No, the Strassen terms mix the indices.
                                # In the 2x2 Strassen, u, v, w are vectors over the 4 entries of the 2x2 matrices.
                                # So u_block is a vector over the 4 block positions of A.
                                # v_block is a vector over the 4 block positions of B.
                                # w_block is a vector over the 4 block positions of C.
                                
                                # Similarly, u_local is over the 4 local entries of a block.
                                # v_local is over the 4 local entries of a block.
                                # w_local is over the 4 local entries of a block.
                                
                                # So for V, the index is (j, k) of B.
                                # j = 2*J + j_l? No.
                                # The block index J is the row of the block in A?
                                # A is 2x2 blocks. A_{IJ}. I is row, J is col.
                                # B is 2x2 blocks. B_{JK}. J is row, K is col.
                                # So the block index for B is (J, K).
                                # The local index for B is (j_l, k_l).
                                # So B global index is (2J + j_l, 2K + k_l).
                                # V index is (2J + j_l) * 4 + (2K + k_l).
                                
                                # The coefficient for V is v_block[J, K] * v_local[j_l, k_l].
                                # Note: The loop variables I, J, i_l, j_l were for U.
                                # We need to sum over all possible block and local indices.
                                # The term is a sum over all indices.
                                # So we should not just use the same I, J, i_l, j_l from the U loop.
                                # We need to iterate over all indices for V and W separately.
                                
                                # Let's restart the filling of V and W for this term.
                                pass # Placeholder

            # Correct approach:
            # For each of the 49 terms, we define U, V, W vectors of length 16.
            # U[i, j] = u_block[I, J] * u_local[i_l, j_l] where i=2I+i_l, j=2J+j_l.
            # V[j, k] = v_block[J, K] * v_local[j_l, k_l] where j=2J+j_l, k=2K+k_l.
            # W[k, i] = w_block[K, I] * w_local[k_l, i_l] where k=2K+k_l, i=2I+i_l.
            
            # We can fill U, V, W by iterating over all 16 indices for each.
            
            for i_glob in range(16):
                I = i_glob // 4
                rem = i_glob % 4
                i_l = rem // 2
                J = rem % 2
                # Wait, i_glob = 4*i + j.
                # i = 2*I + i_l. j = 2*J + j_l.
                # i_glob = 4*(2I+i_l) + 2J + j_l = 8I + 4i_l + 2J + j_l.
                # I = i_glob // 8
                # rem = i_glob % 8
                # i_l = rem // 4
                # rem2 = rem % 4
                # J = rem2 // 2
                # j_l = rem2 % 2
                
                I = i_glob // 8
                rem = i_glob % 8
                i_l = rem // 4
                rem2 = rem % 4
                J = rem2 // 2
                j_l = rem2 % 2
                
                coeff = u_block[2*I + J] * u_local[2*i_l + j_l]
                if coeff != 0:
                    U[i_glob][col] += coeff

            for j_glob in range(16):
                # j_glob is index for B row/col? No, V is m x p = 4 x 7.
                # The 4x4 part of B has indices 0..15 in the 4x4 submatrix.
                # But V is 28 x rank.
                # The first 16 rows of V correspond to B columns 0..3?
                # No, V index is j * p + k.
                # For the 4x4 part, k is 0..3.
                # So V[j * 7 + k] for j in 0..3, k in 0..3.
                # This is not a contiguous block of 16 in V.
                # We must map (j, k) to V index.
                
                # j_glob in the 4x4 sense is the row index of B.
                # Let's iterate j in 0..3 and k in 0..3.
                pass # Handled below

            # Iterate j, k for V
            for j in range(4):
                for k in range(4):
                    J = j // 2
                    j_l = j % 2
                    K = k // 2
                    k_l = k % 2
                    coeff = v_block[2*J + K] * v_local[2*j_l + k_l]
                    if coeff != 0:
                        V[j * 7 + k][col] += coeff

            # Iterate k, i for W
            # W index is k * n + i = k * 4 + i.
            for k in range(4):
                for i in range(4):
                    K = k // 2
                    k_l = k % 2
                    I = i // 2
                    i_l = i % 2
                    coeff = w_block[2*K + I] * w_local[2*k_l + i_l]
                    if coeff != 0:
                        W[k * 4 + i][col] += coeff

            col += 1

    # 2. Construct 4x4x3 Schoolbook Decomposition (Rank 48)
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i * 4 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}

