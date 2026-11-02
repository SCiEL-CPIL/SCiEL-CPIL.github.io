import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline to be safe
    
    # Fallback solution: Rank 189 (2x2 Strassen on 3x3 blocks)
    def get_fallback():
        block_size = 2
        n_blocks = n // block_size
        rank = n_blocks ** 3 * 7
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        def u_idx(ib, kb, r, c): return (ib * block_size + r) * m + (kb * block_size + c)
        def v_idx(kb, jb, r, c): return (kb * block_size + r) * p + (jb * block_size + c)
        def w_idx(ib, jb, r, c): return (jb * block_size + c) * n + (ib * block_size + r)
        
        for ib in range(n_blocks):
            for jb in range(n_blocks):
                for kb in range(n_blocks):
                    # M1
                    U[u_idx(ib, kb, 0, 0), col] += 1; U[u_idx(ib, kb, 1, 1), col] += 1
                    V[v_idx(kb, jb, 0, 0), col] += 1; V[v_idx(kb, jb, 1, 1), col] += 1
                    W[w_idx(ib, jb, 0, 0), col] += 1; W[w_idx(ib, jb, 1, 1), col] += 1
                    col += 1
                    # M2
                    U[u_idx(ib, kb, 1, 0), col] += 1; U[u_idx(ib, kb, 1, 1), col] += 1
                    V[v_idx(kb, jb, 0, 0), col] += 1
                    W[w_idx(ib, jb, 1, 0), col] += 1; W[w_idx(ib, jb, 1, 1), col] -= 1
                    col += 1
                    # M3
                    U[u_idx(ib, kb, 0, 0), col] += 1
                    V[v_idx(kb, jb, 0, 1), col] += 1; V[v_idx(kb, jb, 1, 1), col] -= 1
                    W[w_idx(ib, jb, 0, 1), col] += 1; W[w_idx(ib, jb, 1, 1), col] += 1
                    col += 1
                    # M4
                    U[u_idx(ib, kb, 1, 1), col] += 1
                    V[v_idx(kb, jb, 1, 0), col] += 1; V[v_idx(kb, jb, 0, 0), col] -= 1
                    W[w_idx(ib, jb, 0, 0), col] += 1; W[w_idx(ib, jb, 1, 0), col] += 1
                    col += 1
                    # M5
                    U[u_idx(ib, kb, 0, 0), col] += 1; U[u_idx(ib, kb, 0, 1), col] += 1
                    V[v_idx(kb, jb, 1, 1), col] += 1
                    W[w_idx(ib, jb, 0, 0), col] -= 1; W[w_idx(ib, jb, 0, 1), col] += 1
                    col += 1
                    # M6
                    U[u_idx(ib, kb, 1, 0), col] += 1; U[u_idx(ib, kb, 0, 0), col] -= 1
                    V[v_idx(kb, jb, 0, 0), col] += 1; V[v_idx(kb, jb, 0, 1), col] += 1
                    W[w_idx(ib, jb, 1, 1), col] += 1
                    col += 1
                    # M7
                    U[u_idx(ib, kb, 0, 1), col] += 1; U[u_idx(ib, kb, 1, 1), col] -= 1
                    V[v_idx(kb, jb, 1, 0), col] += 1; V[v_idx(kb, jb, 1, 1), col] += 1
                    W[w_idx(ib, jb, 0, 0), col] += 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    try:
        if time.time() > deadline:
            return get_fallback()
        
        # Attempt Rank 76 solution: 4 blocks of 3x3, each with rank 19
        # This uses the known minimal rank-19 decomposition for 3x3 matrix multiplication
        # We'll construct the 6x6 decomposition by placing 4 independent rank-19 decompositions
        # in the four 3x3 block positions
        
        # The rank-19 decomposition for 3x3 comes from various sources.
        # One well-known construction uses the following structure.
        
        # Let's use the decomposition from "On the rank of matrix multiplication" 
        # where 3x3 can be done in 19 scalar multiplications.
        
        # Actually, the minimal known rank for 3x3 matrix multiplication is 19.
        # Let's implement a specific rank-19 decomposition.
        
        # Here is one such decomposition (from literature):
        # We need 19 terms, each with U, V, W components for 3x3 matrices.
        
        # For simplicity and correctness, let's use a different approach:
        # Use the fact that we can decompose 6x6 using 2x2 Strassen blocks of size 3,
        # where each 3x3 multiplication uses rank 19.
        
        # But 2x2 Strassen gives 7 multiplications per block position, and we have 4 block positions,
        # so total rank = 4 * 7 * (rank per 3x3 / 1) ... no, that's not right.
        
        # Actually, if we use Strassen at the block level with 3x3 subblocks,
        # each Strassen term involves products of 3x3 blocks, which themselves need rank-19 decompositions.
        # So each Strassen term would need 19 scalar multiplications, giving 7 * 19 = 133 per block position,
        # and 4 block positions, so 4 * 133 = 532. That's worse than 189.
        
        # Let me reconsider. The direct approach is to use 4 independent 3x3 multiplications,
        # each of rank 19, giving 4 * 19 = 76. But this only works if the 6x6 matrix is block-diagonal,
        # which it's not in general.
        
        # Actually, 6x6 matrix multiplication cannot be decomposed into 4 independent 3x3 multiplications
        # because of the cross-block terms.
        
        # Let's try a different strategy: use the rank-19 decomposition for 3x3 and see if we can
        # tile it to get a 6x6 decomposition.
        
        # One approach: decompose the 6x6 tensor as a combination of 3x3 tensor decompositions.
        
        # Actually, I think the safest approach is to stick with the rank-189 solution,
        # but try to see if we can find a lower rank decomposition.
        
        # Let's try to implement a rank-162 or rank-150 solution if possible.
        
        # For now, return the fallback.
        return get_fallback()
        
    except Exception as e:
        return get_fallback()
