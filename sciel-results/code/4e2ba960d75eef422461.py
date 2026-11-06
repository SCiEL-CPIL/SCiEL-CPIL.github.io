import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_factors(n):
    """
    Returns U, V, W for n x n matrix multiplication using Strassen's algorithm.
    Shapes: U (n*n, rank), V (n*n, rank), W (n*n, rank).
    Indices are row-major: i*n + j.
    """
    if n == 1:
        return np.array([[1]]), np.array([[1]]), np.array([[1]])
    
    m = n // 2
    U_sub, V_sub, W_sub = _strassen_factors(m)
    rank_sub = U_sub.shape[1]
    
    # Strassen combinations for 2x2 block
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to construct U, V, W for the 2n x 2n case.
    # The total rank will be 7 * rank_sub.
    rank_total = 7 * rank_sub
    
    U = np.zeros((n*n, rank_total), dtype=int)
    V = np.zeros((n*n, rank_total), dtype=int)
    W = np.zeros((n*n, rank_total), dtype=int)
    
    def get_block_indices(block_r, block_c):
        """
        Returns list of indices (i*n + j) for the block at (block_r, block_c)
        in an n x n matrix, where each block is m x m.
        block_r, block_c are 0 or 1.
        """
        indices = []
        for i in range(m):
            for j in range(m):
                global_i = block_r * m + i
                global_j = block_c * m + j
                indices.append(global_i * n + global_j)
        return indices

    idx_A11 = get_block_indices(0, 0)
    idx_A12 = get_block_indices(0, 1)
    idx_A21 = get_block_indices(1, 0)
    idx_A22 = get_block_indices(1, 1)
    
    idx_B11 = get_block_indices(0, 0)
    idx_B12 = get_block_indices(0, 1)
    idx_B21 = get_block_indices(1, 0)
    idx_B22 = get_block_indices(1, 1)
    
    idx_C11 = get_block_indices(0, 0)
    idx_C12 = get_block_indices(0, 1)
    idx_C21 = get_block_indices(1, 0)
    idx_C22 = get_block_indices(1, 1)
    
    # Helper to add sub-factors
    def add_factors(U_block_idx, V_block_idx, W_block_idx, sign, offset):
        """
        Adds the sub-decomposition for one Strassen multiplication.
        U_block_idx: list of indices in U for the A-part
        V_block_idx: list of indices in V for the B-part
        W_block_idx: list of indices in W for the C-part
        sign: +1 or -1 (applied to W usually, but here we can handle it in W)
        offset: column offset in the total U, V, W
        """
        # Copy U_sub to U[:, offset:offset+rank_sub] for the relevant rows
        # U_sub is (m*m, rank_sub). We need to map its rows to U_block_idx.
        # U_sub row i corresponds to local index i.
        # We need to place U_sub[i, k] into U[U_block_idx[i], offset+k]
        
        for i in range(m*m):
            U[U_block_idx[i], offset:offset+rank_sub] = U_sub[i, :] * sign
            V[V_block_idx[i], offset:offset+rank_sub] = V_sub[i, :] * sign
            W[W_block_idx[i], offset:offset+rank_sub] = W_sub[i, :] * sign

    # Note: The standard Strassen decomposition expresses C as linear combo of Ms.
    # M1 = A11_plus_A22 * B11_plus_B22
    # This is not a single multiplication of (A11+A22) and (B11+B22) in the tensor sense 
    # unless we expand.
    # Actually, the tensor decomposition for M1 is:
    # sum_k (A11+A22)_{...} (B11+B22)_{...}
    # This expands to 4 terms if we don't combine.
    # BUT, Strassen's algorithm works by defining new matrices A' and B' which are linear combos.
    # The rank is 7 * rank(m).
    # The factors for M1 are constructed by summing the factors of A11 and A22, etc.
    
    # Let's refine the `add_factors` logic.
    # For M1: A_part = A11 + A22. B_part = B11 + B22.
    # U for M1 should have 1s where A11 is, and 1s where A22 is.
    # So we can call a function that adds contributions.
    
    def add_contribution(A_blocks, B_blocks, C_blocks, offset):
        """
        A_blocks: list of (block_idx, coeff)
        B_blocks: list of (block_idx, coeff)
        C_blocks: list of (block_idx, coeff)
        """
        for i in range(m*m):
            # This logic is flawed because U_sub is for a specific m x m multiplication.
            # We need to map the local indices of the sub-problem to the global indices.
            # U_sub[i, k] is the coefficient for the i-th element of the A-submatrix.
            # If we are summing A11 and A22, then for each local index i,
            # the global index for A11 is idx_A11[i], and for A22 is idx_A22[i].
            # Both should get the coefficient U_sub[i, k].
            
            for idx, coeff in A_blocks:
                U[idx[i], offset:offset+rank_sub] += coeff * U_sub[i, :]
            for idx, coeff in B_blocks:
                V[idx[i], offset:offset+rank_sub] += coeff * V_sub[i, :]
            for idx, coeff in C_blocks:
                W[idx[i], offset:offset+rank_sub] += coeff * W_sub[i, :]

    # M1 = (A11 + A22)(B11 + B22)
    add_contribution(
        A_blocks=[(idx_A11, 1), (idx_A22, 1)],
        B_blocks=[(idx_B11, 1), (idx_B22, 1)],
        C_blocks=[(idx_C11, 1), (idx_C22, 1)], # C11 += M1, C22 += M1
        offset=0
    )
    
    # M2 = (A21 + A22)B11
    add_contribution(
        A_blocks=[(idx_A21, 1), (idx_A22, 1)],
        B_blocks=[(idx_B11, 1)],
        C_blocks=[(idx_C21, 1), (idx_C22, -1)], # C21 += M2, C22 -= M2
        offset=rank_sub
    )
    
    # M3 = A11(B21 - B22)
    add_contribution(
        A_blocks=[(idx_A11, 1)],
        B_blocks=[(idx_B21, 1), (idx_B22, -1)],
        C_blocks=[(idx_C12, 1), (idx_C22, 1)], # C12 += M3, C22 += M3
        offset=2*rank_sub
    )
    
    # M4 = A22(B12 - B11)
    add_contribution(
        A_blocks=[(idx_A22, 1)],
        B_blocks=[(idx_B12, 1), (idx_B11, -1)],
        C_blocks=[(idx_C11, 1), (idx_C21, 1)], # C11 += M4, C21 += M4
        offset=3*rank_sub
    )
    
    # M5 = (A11 + A12)B22
    add_contribution(
        A_blocks=[(idx_A11, 1), (idx_A12, 1)],
        B_blocks=[(idx_B22, 1)],
        C_blocks=[(idx_C11, -1), (idx_C12, 1)], # C11 -= M5, C12 += M5
        offset=4*rank_sub
    )
    
    # M6 = (A21 - A11)(B11 + B12)
    add_contribution(
        A_blocks=[(idx_A21, 1), (idx_A11, -1)],
        B_blocks=[(idx_B11, 1), (idx_B12, 1)],
        C_blocks=[(idx_C22, 1)], # C22 += M6
        offset=5*rank_sub
    )
    
    # M7 = (A12 - A22)(B21 + B22)
    add_contribution(
        A_blocks=[(idx_A12, 1), (idx_A22, -1)],
        B_blocks=[(idx_B21, 1), (idx_B22, 1)],
        C_blocks=[(idx_C11, 1)], # C11 += M7
        offset=6*rank_sub
    )
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # We need n=m=p=5.
    # We use Strassen for 8x8 and restrict to 5x5.
    N = 8
    U_8, V_8, W_8 = _strassen_factors(N)
    
    # Extract 5x5 sub-blocks
    # Indices for 5x5 within 8x8: i in 0..4, j in 0..4
    idxs = [i * N + j for i in range(5) for j in range(5)]
    
    U = U_8[idxs, :]
    V = V_8[idxs, :]
    W = W_8[idxs, :]
    
    # Remove zero columns to minimize rank?
    # The task says "rank-minimal". We should check if any columns are zero.
    # In Strassen, all columns should be non-zero.
    # However, due to the restriction, some columns might become zero if the
    # corresponding linear combination only involves indices outside 0..4.
    # Given the dense nature of Strassen, this is unlikely for the first 5x5 block.
    
    # Check for zero columns
    nonzero_cols = np.any(U != 0, axis=0) & np.any(V != 0, axis=0) & np.any(W != 0, axis=0)
    U = U[:, nonzero_cols]
    V = V[:, nonzero_cols]
    W = W[:, nonzero_cols]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
