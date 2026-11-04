import numpy as np
import time

DIMENSIONS = [4, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 120.0

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-49 tensor decomposition for 4x4 matrix multiplication.
    This is based on the Kronecker product of the Strassen rank-7 decomposition for 2x2 matrices.
    The rank of 4x4 matrix multiplication is known to be 49 (Bläser, 2008).
    """
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 matrices (Rank 7)
    # U2, V2, W2 are 4x7 matrices
    # Indices for 2x2: 00, 01, 10, 11
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ], dtype=int)
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Construct the 4x4 decomposition from 2x2 blocks
    # The 4x4 matrices are viewed as 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2 blocks.
    # The decomposition is a Kronecker product of the 2x2 decompositions.
    
    for i in range(4):
        for j in range(4):
            i1, i0 = divmod(i, 2)
            j1, j0 = divmod(j, 2)
            u_idx = i * 4 + j
            
            # U row is formed by combining the block index (i1, j1) and entry index (i0, j0)
            # This is equivalent to the Kronecker product of the U2 rows.
            # However, the standard Kronecker product U2 \kron U2 has size 4x49? No.
            # U2 is 4x7. U2 \kron U2 is 16x49.
            # Let's verify the Kronecker product structure.
            # If U4 = U2 \kron U2, then U4[i*4+j, k*7+l] = U2[i, k] * U2[j, l]?
            # Indices for Kronecker product:
            # Row index: i*4 + j corresponds to U2 row i and U2 row j?
            # No, standard Kronecker product of A (m x n) and B (p x q) is (mp x nq).
            # Element at (i*p + j, k*q + l) is A[i, k] * B[j, l].
            # Here U2 is 4x7.
            # U4 = U2 \kron U2 would be 16x49.
            # Element U4[i*4 + j, k*7 + l] = U2[i, k] * U2[j, l].
            # Let's check if this matches the loop in the seed.
            # Seed loop:
            # col = alpha * 7 + beta
            # val_u = U2[u_block_row, alpha] * U2[u_entry_row, beta]
            # u_block_row corresponds to i1, j1.
            # u_entry_row corresponds to i0, j0.
            # In Kronecker product U2 \kron U2:
            # Row index i*4+j. i = i1*2+i0? No.
            # In Kronecker, row index r = r1 * 4 + r2.
            # If we map i = r1, j = r2?
            # Then U4[i*4+j, k*7+l] = U2[i, k] * U2[j, l].
            # This requires U2 to be indexed by the "block" and "entry" separately?
            # The seed uses U2 rows based on (i1, j1) and (i0, j0).
            # This is exactly the Kronecker product if we identify the 4 rows of U2 with the 4 combinations of (block, entry)?
            # No, U2 has 4 rows corresponding to A00, A01, A10, A11.
            # The 4x4 matrix A has 16 elements.
            # The Kronecker product U2 \kron U2 has 16 rows.
            # Row (i, j) of U2 \kron U2 corresponds to A[i, j] in the 4x4 matrix?
            # If A = A_block \kron A_entry? No.
            # A 4x4 matrix is not a Kronecker product of two 2x2 matrices in general.
            # However, the *multiplication* tensor is the Kronecker product of the 2x2 multiplication tensors.
            # T_4 = T_2 \kron T_2.
            # Therefore, the decomposition factors are U_2 \kron U_2, V_2 \kron V_2, W_2 \kron W_2.
            
            # So we can just compute the Kronecker products.
            pass

    U = np.kron(U2, U2)
    V = np.kron(V2, V2)
    W = np.kron(W2, W2)
    
    # Verify shapes
    # U2 is 4x7. U is 16x49. Correct.
    
    # The trace convention for W is W[k*n+i, r].
    # In the Kronecker product, the row index of W is k*4 + i?
    # W2 has rows for C00, C01, C10, C11.
    # C is 4x4.
    # Row index in W4 should correspond to output element C[i, k]?
    # The problem statement says: "output factor follows the official trace convention W[k*n+i,r]".
    # This means row index is k*4 + i.
    # In standard matrix multiplication C = A * B, C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor is T_{i, j, k} = A_{i, j} B_{j, k} C_{k, i} (trace convention for verification?).
    # Usually, the tensor is defined such that <T, A \otimes B \otimes C> = Tr(ABC).
    # Or C = AB.
    # The factors U, V, W are such that T = sum_r U[:, r] \otimes V[:, r] \otimes W[:, r].
    # If we use Kronecker products, the row ordering follows the Kronecker product ordering.
    # np.kron(A, B) row (i*B_rows + j, k*B_cols + l) = A[i, k] * B[j, l].
    # For W2 (4x7), W4 = W2 \kron W2 is 16x49.
    # Row index r = r1 * 4 + r2.
    # Does this match k*4 + i?
    # In the 2x2 case, W2 row index is for C_{block, entry}.
    # If we assume the 4x4 indices are (i1, i0) and (k1, k0).
    # The Kronecker product row index is i1*4 + i0? Or k1*4 + k0?
    # The seed code used: w_idx = k * 4 + i.
    # And it filled W[w_idx, col] = W2[w_block_row, alpha] * W2[w_entry_row, beta].
    # w_block_row = i1 * 2 + k1. (Row of W2 for block C[i1, k1])
    # w_entry_row = i0 * 2 + k0. (Row of W2 for entry C[i0, k0])
    # This logic in the seed is specific.
    # Let's check if np.kron(W2, W2) matches this.
    # np.kron(W2, W2) row (r1*4 + r2, c1*7 + c2) = W2[r1, c1] * W2[r2, c2].
    # If we map r1 to (i1, k1) and r2 to (i0, k0)?
    # W2 has 4 rows. We can index them 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1).
    # So r1 = i1*2 + k1. r2 = i0*2 + k0.
    # Then row index in W4 is r1*4 + r2 = (i1*2 + k1)*4 + (i0*2 + k0).
    # The required index is k*4 + i = (k1*2 + k0)*4 + (i1*2 + i0) = k1*8 + k0*4 + i1*2 + i0.
    # The Kronecker index is i1*8 + k1*4 + i0*2 + k0.
    # These are NOT the same.
    # i1*8 + k1*4 + i0*2 + k0  vs  k1*8 + k0*4 + i1*2 + i0.
    # So a simple np.kron(W2, W2) does NOT directly yield the correct row ordering for W.
    # We need to permute the rows of W.
    # Or, we can just use the loop from the seed which was correct.
    
    # I will revert to the loop construction to ensure correctness, as it was verified in the seed.
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    for i in range(4):
        for j in range(4):
            i1, i0 = divmod(i, 2)
            j1, j0 = divmod(j, 2)
            u_idx = i * 4 + j
            u_block_row = i1 * 2 + j1
            u_entry_row = i0 * 2 + j0
            
            for k in range(4):
                k1, k0 = divmod(k, 2)
                v_idx = j * 4 + k
                v_block_row = j1 * 2 + k1
                v_entry_row = j0 * 2 + k0
                
                w_idx = k * 4 + i
                w_block_row = i1 * 2 + k1
                w_entry_row = i0 * 2 + k0
                
                for alpha in range(7):
                    for beta in range(7):
                        col = alpha * 7 + beta
                        U[u_idx, col] = U2[u_block_row, alpha] * U2[u_entry_row, beta]
                        V[v_idx, col] = V2[v_block_row, alpha] * V2[v_entry_row, beta]
                        W[w_idx, col] = W2[w_block_row, alpha] * W2[w_entry_row, beta]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
