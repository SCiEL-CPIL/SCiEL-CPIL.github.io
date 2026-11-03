import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline to be safe
    
    def get_strassen_2x2_blocks():
        """Generate the 2x2 Strassen decomposition for 6x6 using 2x2 subblocks.
        This gives rank 3*3*7 = 189."""
        block_size = 2
        n_blocks = n // block_size  # 3 blocks
        rank = n_blocks ** 3 * 7
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        
        def u_idx(ib, kb, r, c):
            return (ib * block_size + r) * m + (kb * block_size + c)
        
        def v_idx(kb, jb, r, c):
            return (kb * block_size + r) * p + (jb * block_size + c)
        
        def w_idx(ib, jb, r, c):
            return (jb * block_size + c) * n + (ib * block_size + r)
        
        for ib in range(n_blocks):
            for jb in range(n_blocks):
                for kb in range(n_blocks):
                    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C01 += M1, C10 += M1, C11 += M1
                    U[u_idx(ib, kb, 0, 0), col] += 1
                    U[u_idx(ib, kb, 1, 1), col] += 1
                    V[v_idx(kb, jb, 0, 0), col] += 1
                    V[v_idx(kb, jb, 1, 1), col] += 1
                    W[w_idx(ib, jb, 0, 0), col] += 1
                    W[w_idx(ib, jb, 0, 1), col] += 1
                    W[w_idx(ib, jb, 1, 0), col] += 1
                    W[w_idx(ib, jb, 1, 1), col] += 1
                    col += 1
                    
                    # M2 = (A10+A11)B00 -> C10 += M2
                    U[u_idx(ib, kb, 1, 0), col] += 1
                    U[u_idx(ib, kb, 1, 1), col] += 1
                    V[v_idx(kb, jb, 0, 0), col] += 1
                    W[w_idx(ib, jb, 1, 0), col] += 1
                    col += 1
                    
                    # M3 = A00(B01-B11) -> C01 += M3
                    U[u_idx(ib, kb, 0, 0), col] += 1
                    V[v_idx(kb, jb, 0, 1), col] += 1
                    V[v_idx(kb, jb, 1, 1), col] -= 1
                    W[w_idx(ib, jb, 0, 1), col] += 1
                    col += 1
                    
                    # M4 = A11(B10-B00) -> C11 += M4
                    U[u_idx(ib, kb, 1, 1), col] += 1
                    V[v_idx(kb, jb, 1, 0), col] += 1
                    V[v_idx(kb, jb, 0, 0), col] -= 1
                    W[w_idx(ib, jb, 1, 1), col] += 1
                    col += 1
                    
                    # M5 = (A00+A01)B11 -> C11 += M5
                    U[u_idx(ib, kb, 0, 0), col] += 1
                    U[u_idx(ib, kb, 0, 1), col] += 1
                    V[v_idx(kb, jb, 1, 1), col] += 1
                    W[w_idx(ib, jb, 1, 1), col] += 1
                    col += 1
                    
                    # M6 = (A10-A00)(B00+B01) -> C00 += M6
                    U[u_idx(ib, kb, 1, 0), col] += 1
                    U[u_idx(ib, kb, 0, 0), col] -= 1
                    V[v_idx(kb, jb, 0, 0), col] += 1
                    V[v_idx(kb, jb, 0, 1), col] += 1
                    W[w_idx(ib, jb, 0, 0), col] += 1
                    col += 1
                    
                    # M7 = (A01-A11)(B10+B11) -> C00 += M7
                    U[u_idx(ib, kb, 0, 1), col] += 1
                    U[u_idx(ib, kb, 1, 1), col] -= 1
                    V[v_idx(kb, jb, 1, 0), col] += 1
                    V[v_idx(kb, jb, 1, 1), col] += 1
                    W[w_idx(ib, jb, 0, 0), col] += 1
                    col += 1
        
        return U, V, W

    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        # The tensor T_{i,j,k} = delta_{i,k} (for A_{i,j} * B_{j,k} = C_{i,k})
        # Decomposition: T = sum_r U_{i,r} * V_{j,r} * W_{k,r}
        # Where i indexes A row (0..n*m-1), j indexes B col (0..m*p-1), k indexes C (0..p*n-1)
        
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # The contraction is: sum_r U[i, r] * V[j, r] * W[k, r]
        
        # For matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor form: T[i, j, k] where i = A row index (0..n-1), 
        #              j = (A col / B row) index (0..m-1), 
        #              k = (B col / C col) index (0..p-1)
        # But in the flat indexing:
        #   A index: i_A = i * m + j (i in 0..n-1, j in 0..m-1) -> 0..n*m-1
        #   B index: i_B = j * p + k (j in 0..m-1, k in 0..p-1) -> 0..m*p-1  
        #   C index: i_C = i * p + k (i in 0..n-1, k in 0..p-1) -> 0..p*n-1
        
        # Wait, let me re-read the convention: W[k*n+i, r]
        # So C index = k*n + i where i is row, k is col
        # A index = i*m + j where i is row, j is col
        # B index = j*p + k where j is row, k is col
        
        # T[i_A, i_B, i_C] = 1 if the (i_A row, i_A col) of A, (i_B row, i_B col) of B,
        #   and (i_C row, i_C col) of C are consistent with matrix mult.
        #   i_A row = i, i_A col = j
        #   i_B row = j, i_B col = k  
        #   i_C row = i, i_C col = k
        #   T[i*m+j, j*p+k, k*n+i] = 1
        
        rank = U.shape[1]
        # Compute the decomposed tensor and check
        # T_decomposed[i_A, i_B, i_C] = sum_r U[i_A, r] * V[i_B, r] * W[i_C, r]
        
        # For efficiency, check a random sample of entries
        rng = np.random.default_rng(42)
        n_checks = min(500, n * m * m * p * p * n)
        
        for _ in range(n_checks):
            # Random A index
            i_A = rng.integers(0, n * m)
            i_row, i_col = divmod(i_A, m)
            
            # Random B index  
            i_B = rng.integers(0, m * p)
            j_row, j_col = divmod(i_B, p)
            
            # Random C index
            i_C = rng.integers(0, p * n)
            c_row, c_col = divmod(i_C, n)
            
            # Expected value: 1 if i_row == c_row, i_col == j_row, j_col == c_col
            expected = 1.0 if (i_row == c_row and i_col == j_row and j_col == c_col) else 0.0
            
            # Computed value
            computed = 0.0
            for r in range(rank):
                computed += U[i_A, r] * V[i_B, r] * W[i_C, r]
            
            if abs(computed - expected) > tol:
                return False, (i_A, i_B, i_C, computed, expected)
        
        return True, None

    def reduce_rank_by_column_elimination(U, V, W, tol=1e-10):
        """Try to reduce rank by identifying and eliminating linearly dependent structure.
        
        Strategy: If column r can be expressed as a linear combination of other columns
        in all three matrices simultaneously, we can eliminate it.
        
        More practically: try to find a subset of columns that still spans the same tensor.
        """
        n_rows_U, rank = U.shape
        n_rows_V = V.shape[0]
        n_rows_W = W.shape[0]
        
        # Convert to float for numerical stability
        U_f = U.astype(float)
        V_f = V.astype(float)
        W_f = W.astype(float)
        
        # Try to eliminate columns from the end
        # For each column, check if it's "redundant" given the others
        # A column r is redundant if the tensor it contributes is in the span of others
        
        # Simple approach: iterate and try to remove each column
        # Check if removing column r changes the tensor
        # This is expensive, so we do a greedy approach
        
        # First, let's try a simpler check: are there any zero columns?
        for r in range(rank - 1, -1, -1):
            if np.allclose(U_f[:, r], 0, atol=tol) or \
               np.allclose(V_f[:, r], 0, atol=tol) or \
               np.allclose(W_f[:, r], 0, atol=tol):
                # Remove this column
                mask = np.ones(rank, dtype=bool)
                mask[r] = False
                U_f = U_f[:, mask]
                V_f = V_f[:, mask]
                W_f = W_f[:, mask]
        
        return U_f, V_f, W_f

    def try_block_size(block_size):
        """Try a block decomposition with given block size."""
        if n % block_size != 0 or m % block_size != 0 or p % block_size != 0:
            return None
        
        n_blocks = n // block_size
        m_blocks = m // block_size
        p_blocks = p // block_size
        
        # For square blocks, use Strassen if 2x2, schoolbook otherwise
        if block_size == 2 and n_blocks == m_blocks == p_blocks:
            # 2x2 Strassen: rank = (n_blocks/1)^3 * 7 for 1x1 base
            # Actually for 2x2 blocks: rank = (n_blocks)^3 * 7 where each "block mult" 
            # is done with 7 multiplications of block_size x block_size
            # Total rank = (n//block_size)^3 * 7
            rank = n_blocks ** 3 * 7
            if rank < 189:  # Only worth trying if better than current
                U, V, W = get_strassen_2x2_blocks()
                # Verify
                valid, _ = verify_decomposition(U, V, W, n, m, p)
                if valid:
                    return U, V, W, rank
        else:
            # Schoolbook for other sizes
            rank = n_blocks * m_blocks * p_blocks * (block_size ** 3)
            if rank < 189:
                # Build schoolbook
                U = np.zeros((n * m, rank), dtype=int)
                V = np.zeros((m * p, rank), dtype=int)
                W = np.zeros((p * n, rank), dtype=int)
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i * m + j, col] = 1
                            V[j * p + k, col] = 1
                            W[k * n + i, col] = 1
                            col += 1
                valid, _ = verify_decomposition(U, V, W, n, m, p)
                if valid:
                    return U, V, W, rank
        return None

    # Step 1: Get the baseline Strassen decomposition (rank 189)
    U, V, W = get_strassen_2x2_blocks()
    best_rank = 189
    
    # Verify baseline
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, i * m * p + j * p + k] = 1
                    V[j * p + k, i * m * p + j * p + k] = 1
                    W[k * n + i, i * m * p + j * p + k] = 1
        best_rank = rank
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Step 2: Try to reduce rank via column elimination
    if time.time() < deadline - 20:
        U_reduced, V_reduced, W_reduced = reduce_rank_by_column_elimination(U, V, W)
        new_rank = U_reduced.shape[1]
        if new_rank < best_rank:
            valid, _ = verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p)
            if valid:
                U, V, W = U_reduced, V_reduced, W_reduced
                best_rank = new_rank
    
    # Step 3: Try other block structures
    if time.time() < deadline - 30:
        for bs in [3, 4, 6]:
            if time.time() > deadline - 10:
                break
            result = try_block_size(bs)
            if result is not None:
                U2, V2, W2, rank2 = result
                if rank2 < best_rank:
                    U, V, W = U2, V2, W2
                    best_rank = rank2
    
    # Final verification
    valid, _ = verify_decomposition(U, V, W, n, m, p)
    if not valid:
        # Return the original verified Strassen
        U, V, W = get_strassen_2x2_blocks()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
