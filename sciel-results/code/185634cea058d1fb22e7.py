import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s deadline, 150s kill
    
    n, m, p = DIMENSIONS
    
    # Block decomposition with block size 3
    # 6x6 = 2x2 blocks of 3x3
    block_size = 3
    n_blocks = n // block_size  # 2
    m_blocks = m // block_size  # 2
    p_blocks = p // block_size  # 2
    
    # For 2x2 block multiplication with 3x3 sub-blocks:
    # Use trivial (rank 8) for the 2x2 block structure
    # Each block product is a 3x3 matrix multiplication (rank 9)
    # Total rank = 8 * 9 = 72
    
    rank = (n_blocks * m_blocks) * (m_blocks * p_blocks) * (p_blocks * n_blocks)
    # = 4 * 4 * 4 = 64? No.
    # Actually: for each of the 4 output blocks (2x2), we need 3x3 products.
    # Each output block C_ij = sum_k A_ik * B_kj
    # With block size 3, there are 2 block indices for each dimension.
    # So for each (i,j) pair of output blocks (4 total), we sum over k (2 values).
    # Each A_ik * B_kj is a 3x3 matrix multiplication.
    # A 3x3 matrix multiplication has rank 9 (trivial).
    # So total rank = 4 * 2 * 9 = 72.
    
    rank = 72
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to get index in U for block (ib, kb) and local (r, c)
    # U index: (ib * block_size + r) * m + (kb * block_size + c)
    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)
    
    # Helper to get index in V for block (kb, jb) and local (r, c)
    # V index: (kb * block_size + r) * p + (jb * block_size + c)
    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)
    
    # Helper to get index in W for block (ib, jb) and local (r, c)
    # W index is k*n + i, where k is col of C, i is row of C
    # C has shape (n, p), so W index = (jb * block_size + c) * n + (ib * block_size + r)
    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)
    
    # For each output block (ib, jb) and each intermediate block kb:
    # We need to compute A_{ib,kb} * B_{kb,jb} which is a 3x3 matrix multiplication.
    # The 3x3 multiplication has 9 rank-1 terms (trivial).
    
    for ib in range(n_blocks):
        for jb in range(p_blocks):
            for kb in range(m_blocks):
                # 9 rank-1 terms for 3x3 matrix multiplication
                for r in range(block_size):
                    for c in range(block_size):
                        # U: A_{ib,kb}[r, c]
                        U[u_idx(ib, kb, r, c), col] += 1
                        # V: B_{kb,jb}[c, ?] - wait, we need to be careful.
                        # For matrix multiplication C = A * B, where A is 3x3, B is 3x3:
                        # C[r, s] = sum_c A[r, c] * B[c, s]
                        # Rank-1 term for (r, s): A[r, :] * B[:, s]
                        # So U gets A[r, c] for all c, V gets B[c, s] for all c.
                        # But in our factorization, each rank-1 term is:
                        # U[:, col] has A[r, c] at position (ib, kb, r, c)
                        # V[:, col] has B[c, s] at position (kb, jb, c, s)
                        # W[:, col] has 1 at position (ib, jb, r, s)
                        
                        # Wait, I need to think about this more carefully.
                        # The tensor for 3x3 matrix multiplication is:
                        # T[r, c, s] = A[r, c] * B[c, s]
                        # This can be decomposed as sum over c of rank-1 terms.
                        # For each c, the term is: A[:, c] * B[c, :]
                        # So U gets 1 at (r, c) for all r, V gets 1 at (c, s) for all s.
                        
                        # Let me redo this.
                        # For the 3x3 multiplication:
                        # C[r, s] = sum_{c=0}^{2} A[r, c] * B[c, s]
                        # Rank-1 decomposition: for each c, we have a term
                        # U_term: 1 at position (r, c) for all r in 0..2
                        # V_term: 1 at position (c, s) for all s in 0..2
                        # W_term: 1 at position (r, s) for all r, s in 0..2
                        # But that's not right either. Each rank-1 term should be:
                        # U has 1 at (r, c) only for the specific r, c
                        # No, that's not how it works.
                        
                        # Let me think again. The standard rank-1 decomposition for
                        # matrix multiplication of size n is:
                        # For each (i, j, k) in [0, n)^3, we have a rank-1 term:
                        # U[i, k] = 1, V[k, j] = 1, W[i, j] = 1
                        # So rank = n^3.
                        
                        # For 3x3: rank = 27.
                        # Total rank = 4 * 2 * 27 = 216. That's worse than 189!
                        
                        # I made an error. Let me reconsider.
                        pass
    
    # Let me restart with the correct understanding.
    # The trivial decomposition for n x n matrix multiplication has rank n^3.
    # For block size 3: each 3x3 multiplication has rank 27.
    # Total rank = (2*2) * (2) * 27 = 216. Worse than 189.
    
    # For block size 2: each 2x2 multiplication has rank 8 (trivial) or 7 (Strassen).
    # With Strassen: total rank = 3*3 * 3 * 7 = 189. (current)
    # With trivial: total rank = 3*3 * 3 * 8 = 216.
    
    # So the current approach with block size 2 and Strassen is already good.
    # Can we do better?
    
    # What if we use a mixed approach?
    # Or what if we find a better decomposition for 6x6?
    
    # Actually, let me just return the current best (189) since the proposed
    # block size 3 approach is worse.
    
    # Let me reconstruct the 189-rank decomposition.
    
    block_size = 2
    n_blocks = n // block_size  # 3
    
    rank = n_blocks ** 3 * 7  # 27 * 7 = 189
    
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
                # M1 = (X00 + X11)(Y00 + Y11)
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1
                
                # M2 = (X10 + X11)Y00
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] -= 1
                col += 1
                
                # M3 = X00(Y01 - Y11)
                U[u_idx(ib, kb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1
                
                # M4 = X11(Y10 - Y00)
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                col += 1
                
                # M5 = (X00 + X01)Y11
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                col += 1
                
                # M6 = (X10 - X00)(Y00 + Y01)
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 0, 0), col] -= 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1
                
                # M7 = (X01 - X11)(Y10 + Y11)
                U[u_idx(ib, kb, 0, 1), col] += 1
                U[u_idx(ib, kb, 1, 1), col] -= 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
