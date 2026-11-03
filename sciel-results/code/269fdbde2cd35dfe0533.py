import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen formulas for 2x2 block multiplication of 4x4 matrices
    # Blocks are 2x2.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Indices in flattened matrices:
    # A row i, col j -> U[i*m+j, r]
    # B row j, col k -> V[j*p+k, r] (Note: p=8, but we split into two 4-col parts)
    # C row i, col k -> W[k*n+i, r]
    
    def set_block(matrix, block_row, block_col, block_size, value):
        """
        Sets a block of size block_size x block_size in matrix.
        block_row, block_col are 0 or 1 (index of the 2x2 block).
        """
        r_start = block_row * block_size
        c_start = block_col * block_size
        matrix[r_start:r_start+block_size, c_start:c_start+block_size] = value

    # Define the 7 Strassen terms (U, V, W) for a single 4x4x4 multiplication
    # We represent U, V, W as 4x4 matrices of coefficients for the blocks.
    # Actually, it's easier to just define the non-zero indices.
    
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We will construct U, V, W for the first 4 columns of B (k=0..3)
    # and then for the next 4 columns of B (k=4..7)
    
    for half in range(2):
        offset_k = half * 4
        offset_w = half * 16 # Since W index is k*n+i, and n=4, shifting k by 4 shifts index by 16
        
        for r_idx in range(7):
            rank_idx = half * 7 + r_idx
            
            # Determine U, V, W patterns for this term
            # U pattern (4x4), V pattern (4x4), W pattern (4x4)
            # We'll use 0, 1, -1
            
            U_mat = np.zeros((4, 4), dtype=int)
            V_mat = np.zeros((4, 4), dtype=int)
            W_mat = np.zeros((4, 4), dtype=int)
            
            if r_idx == 0:
                # P1 = (A00 + A11)(B00 + B11)
                set_block(U_mat, 0, 0, 2, 1)
                set_block(U_mat, 1, 1, 2, 1)
                set_block(V_mat, 0, 0, 2, 1)
                set_block(V_mat, 1, 1, 2, 1)
                # C00 += P1, C11 += P1
                set_block(W_mat, 0, 0, 2, 1)
                set_block(W_mat, 1, 1, 2, 1)
            elif r_idx == 1:
                # P2 = (A10 + A11)B00
                set_block(U_mat, 1, 0, 2, 1)
                set_block(U_mat, 1, 1, 2, 1)
                set_block(V_mat, 0, 0, 2, 1)
                # C10 += P2, C11 -= P2
                set_block(W_mat, 1, 0, 2, 1)
                set_block(W_mat, 1, 1, 2, -1)
            elif r_idx == 2:
                # P3 = A00(B01 - B11)
                set_block(U_mat, 0, 0, 2, 1)
                set_block(V_mat, 0, 1, 2, 1)
                set_block(V_mat, 1, 1, 2, -1)
                # C01 += P3, C11 += P3
                set_block(W_mat, 0, 1, 2, 1)
                set_block(W_mat, 1, 1, 2, 1)
            elif r_idx == 3:
                # P4 = A11(B10 - B00)
                set_block(U_mat, 1, 1, 2, 1)
                set_block(V_mat, 1, 0, 2, 1)
                set_block(V_mat, 0, 0, 2, -1)
                # C00 += P4, C10 += P4
                set_block(W_mat, 0, 0, 2, 1)
                set_block(W_mat, 1, 0, 2, 1)
            elif r_idx == 4:
                # P5 = (A00 + A01)B11
                set_block(U_mat, 0, 0, 2, 1)
                set_block(U_mat, 0, 1, 2, 1)
                set_block(V_mat, 1, 1, 2, 1)
                # C00 -= P5, C01 += P5
                set_block(W_mat, 0, 0, 2, -1)
                set_block(W_mat, 0, 1, 2, 1)
            elif r_idx == 5:
                # P6 = (A10 - A00)(B00 + B01)
                set_block(U_mat, 1, 0, 2, 1)
                set_block(U_mat, 0, 0, 2, -1)
                set_block(V_mat, 0, 0, 2, 1)
                set_block(V_mat, 0, 1, 2, 1)
                # C11 += P6
                set_block(W_mat, 1, 1, 2, 1)
            elif r_idx == 6:
                # P7 = (A01 - A11)(B10 + B11)
                set_block(U_mat, 0, 1, 2, 1)
                set_block(U_mat, 1, 1, 2, -1)
                set_block(V_mat, 1, 0, 2, 1)
                set_block(V_mat, 1, 1, 2, 1)
                # C00 += P7
                set_block(W_mat, 0, 0, 2, 1)
            
            # Map U_mat (4x4) to U (16x14)
            # U_mat[i, j] corresponds to U[i*m+j, rank_idx]
            for i in range(4):
                for j in range(4):
                    val = U_mat[i, j]
                    if val != 0:
                        U[i * m + j, rank_idx] = val
                        
            # Map V_mat (4x4) to V (32x14)
            # V_mat[j, k] corresponds to V[j*p+k, rank_idx]
            # Note: k is relative to the 4 columns in this half.
            # Actual col index is k + offset_k.
            for j in range(4):
                for k in range(4):
                    val = V_mat[j, k]
                    if val != 0:
                        V[j * p + (k + offset_k), rank_idx] = val
                        
            # Map W_mat (4x4) to W (32x14)
            # W_mat[i, k] corresponds to C[i, k]
            # W index is k*n + i.
            # Actual col index is k + offset_k.
            for i in range(4):
                for k in range(4):
                    val = W_mat[i, k]
                    if val != 0:
                        W[(k + offset_k) * n + i, rank_idx] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
