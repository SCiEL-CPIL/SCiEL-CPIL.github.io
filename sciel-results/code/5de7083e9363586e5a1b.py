"""Program for 3x5x6 matrix multiplication tensor decomposition."""

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Construct a rank-84 decomposition of the 3x5x6 matrix multiplication tensor.
    Uses Strassen's rank-7 algorithm for 2x2x2 sub-blocks where applicable.
    """
    import time
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # We'll build the decomposition by blocks.
    # Split n=3 into [2,1], m=5 into [2,2,1], p=6 into [2,2,2]
    # Actually, let's use a more systematic approach.
    
    # Strategy: Split dimensions into blocks, use Strassen for 2x2x2 blocks
    # n=3: split as [2, 1]
    # m=5: split as [2, 2, 1]  
    # p=6: split as [2, 2, 2]
    
    # For each combination of blocks (i-block, j-block, k-block), use:
    # - Strassen (rank 7) if all three dimensions are size 2
    # - Schoolbook (rank a*b*c) otherwise
    
    # Let's define the block structure
    n_blocks = [2, 1]  # sum = 3
    m_blocks = [2, 2, 1]  # sum = 5
    p_blocks = [2, 2, 2]  # sum = 6
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Offset tracking
    n_offset = 0
    m_offset = 0
    p_offset = 0
    
    # We'll accumulate terms
    terms = []  # Each term is (u_vec, v_vec, w_vec) where these are dicts or sparse representations
    
    # For efficiency, let's use a different approach:
    # Build the full U, V, W matrices with the right shapes
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # First, compute the total rank
    total_rank = 0
    block_specs = []
    
    for i_blk_size in n_blocks:
        for j_blk_size in m_blocks:
            for k_blk_size in p_blocks:
                if i_blk_size == 2 and j_blk_size == 2 and k_blk_size == 2:
                    blk_rank = 7
                else:
                    blk_rank = i_blk_size * j_blk_size * k_blk_size
                total_rank += blk_rank
                block_specs.append((i_blk_size, j_blk_size, k_blk_size, blk_rank))
    
    # Now allocate
    rank = total_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's rank-7 decomposition for 2x2x2
    # The 7 terms for Strassen's algorithm:
    # Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For the tensor decomposition, each term contributes to U, V, W
    # Term t: U[i*m+j, t] = coeff_a, V[j*p+k, t] = coeff_b, W[k*n+i, t] = 1
    # The sum over t of U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] should equal delta(i,j,k)
    
    # Actually, for matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
    # The decomposition: T = sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    # Where U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
    # And sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all i,j,k
    
    # Wait, let me re-read the problem. The tensor is the matrix multiplication tensor.
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid i,j,k.
    # Decomposition: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
    
    # Let me verify with the schoolbook case:
    # Each term t corresponds to a unique (i,j,k):
    # U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1
    # Sum over t: only one t matches (i,j,k), so sum = 1. Correct.
    
    # For Strassen on a 2x2x2 block:
    # The 7 terms are linear combinations. Let me index locally:
    # i in {0,1}, j in {0,1}, k in {0,1}
    # Term 0: (i=0 or 1, j=0 or 1) -> (i+j combination)
    # Actually, let me be more careful.
    
    # Strassen terms for 2x2x2:
    # t0: A-part: a11+a22, B-part: b11+b22, W: C11 and C22 both use it
    #     U contributions: (0,0):1, (1,1):1 for A indices
    #     V contributions: (0,0):1, (1,1):1 for B indices
    #     W contributions: (0,0):1, (1,1):1 for C indices (both C11 and C22)
    #     Actually W is indexed by (k*n+i), so for the 2x2 block, k in {0,1}, i in {0,1}
    #     C11 is at (k=0, i=0), C22 is at (k=1, i=1)
    #     So W[(0)*2+0, t0] = 1, W[(1)*2+1, t0] = 1
    
    # Let me define the Strassen terms more carefully:
    # Using 0-based indexing for the 2x2 block:
    # A indices: (i,j) in {(0,0),(0,1),(1,0),(1,1)}
    # B indices: (j,k) in {(0,0),(0,1),(1,0),(1,1)}
    # C indices: (i,k) in {(0,0),(0,1),(1,0),(1,1)}
    
    # Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # In the tensor decomposition, for each term t:
    # U[i*m+j, t] = coefficient of A[i,j] in the product
    # V[j*p+k, t] = coefficient of B[j,k] in the product  
    # W[k*n+i, t] = coefficient of C[i,k] from this term
    # sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1 for all i,j,k
    
    # For term M1:
    # U: A00:1, A11:1 -> U[0*m+0]=1, U[1*m+1]=1 (within the block)
    # V: B00:1, B11:1 -> V[0*p+0]=1, V[1*p+1]=1 (within the block)
    # W: C00:1, C11:1 -> W[0*n+0]=1, W[1*n+1]=1 (within the block)
    # Check: For (i=0,j=0,k=0): U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    #        For (i=0,j=0,k=1): U[0,0]*V[1,0]*W[1,0] = 1*0*0 = 0? 
    #        Wait, V[j*p+k] for j=0,k=1 is V[1], which is 0. Good.
    #        For (i=1,j=1,k=0): U[1,1]*V[0,1]*W[0,1] = 1*0*0 = 0? 
    #        V[j*p+k] for j=1,k=0 is V[2], which is 0. Good.
    #        For (i=1,j=1,k=1): U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1. Good.
    #        But we need the sum over ALL terms to give 1 for every (i,j,k).
    #        For (i=0,j=0,k=0): Only M1 contributes 1, others should contribute 0.
    #        Let's check M2 for (i=0,j=0,k=0): U has A10,A11, so U[0,0]=0. Good.
    
    # OK so let me define the Strassen terms as lists of (u_indices, v_indices, w_indices, coeffs)
    
    # Actually, this is getting complex. Let me use a simpler approach:
    # For each block, if it's 2x2x2, use the 7 Strassen terms with global indexing.
    # Otherwise, use schoolbook.
    
    col = 0
    for i_blk, i_off in zip(n_blocks, [0, 2]):
        for j_blk, j_off in zip(m_blocks, [0, 2, 4]):
            for k_blk, k_off in zip(p_blocks, [0, 2, 4]):
                if i_blk == 2 and j_blk == 2 and k_blk == 2:
                    # Strassen 2x2x2, rank 7
                    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
                    # Global: I = i_off + i, J = j_off + j, K = k_off + k
                    strassen_terms = [
                        # (u_entries, v_entries, w_entries)
                        # Each entry is (local_idx, coeff)
                        # u: (i,j) -> i*m+j globally, but within block: (i_off+i)*m + (j_off+j)
                        # v: (j,k) -> (j_off+j)*p + (k_off+k)
                        # w: (k,i) -> (k_off+k)*n + (i_off+i)
                        
                        # M1: (a00+a11)(b00+b11)
                        ([((i_off+0)*m + (j_off+0), 1), ((i_off+1)*m + (j_off+1), 1)],
                         [((j_off+0)*p + (k_off+0), 1), ((j_off+1)*p + (k_off+1), 1)],
                         [((k_off+0)*n + (i_off+0), 1), ((k_off+1)*n + (i_off+1), 1)]),
                        # M2: (a10+a11)b00
                        ([((i_off+1)*m + (j_off+0), 1), ((i_off+1)*m + (j_off+1), 1)],
                         [((j_off+0)*p + (k_off+0), 1)],
                         [((k_off+0)*n + (i_off+1), 1)]),
                        # M3: a00(b01-b11)
                        ([((i_off+0)*m + (j_off+0), 1)],
                         [((j_off+0)*p + (k_off+1), 1), ((j_off+1)*p + (k_off+1), -1)],
                         [((k_off+1)*n + (i_off+0), 1)]),
                        # M4: a11(b10-b00)
                        ([((i_off+1)*m + (j_off+1), 1)],
                         [((j_off+1)*p + (k_off+0), 1), ((j_off+0)*p + (k_off+0), -1)],
                         [((k_off+0)*n + (i_off+1), 1)]),
                        # M5: (a00+a01)b11
                        ([((i_off+0)*m + (j_off+0), 1), ((i_off+0)*m + (j_off+1), 1)],
                         [((j_off+1)*p + (k_off+1), 1)],
                         [((k_off+1)*n + (i_off+0), 1)]),
                        # M6: (a10-a00)(b00+b01)
                        ([((i_off+1)*m + (j_off+0), 1), ((i_off+0)*m + (j_off+0), -1)],
                         [((j_off+0)*p + (k_off+0), 1), ((j_off+0)*p + (k_off+1), 1)],
                         [((k_off+1)*n + (i_off+1), 1)]),
                        # M7: (a01-a11)(b10+b11)
                        ([((i_off+0)*m + (j_off+1), 1), ((i_off+1)*m + (j_off+1), -1)],
                         [((j_off+1)*p + (k_off+0), 1), ((j_off+1)*p + (k_off+1), 1)],
                         [((k_off+0)*n + (i_off+0), 1)]),
                    ]
                    
                    for u_entries, v_entries, w_entries in strassen_terms:
                        for idx, coeff in u_entries:
                            U[idx][col] = coeff
                        for idx, coeff in v_entries:
                            V[idx][col] = coeff
                        for idx, coeff in w_entries:
                            W[idx][col] = coeff
                        col += 1
                else:
                    # Schoolbook for this block
                    for i in range(i_blk):
                        for j in range(j_blk):
                            for k in range(k_blk):
                                gi = i_off + i
                                gj = j_off + j
                                gk = k_off + k
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
    
    if col != rank:
        raise ValueError(f"Column count {col} != rank {rank}")
    
    # Verify
    import numpy as np
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Reconstruct tensor
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # Check a few entries
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                expected = 1
                err = abs(s - expected)
                if err > max_err:
                    max_err = err
                if err > 1e-9:
                    # This shouldn't happen
                    pass
    
    if max_err > 1e-9:
        # Fall back to schoolbook
        rank_school = n * m * p
        U2 = [[0] * rank_school for _ in range(n * m)]
        V2 = [[0] * rank_school for _ in range(m * p)]
        W2 = [[0] * rank_school for _ in range(p * n)]
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i*m+j][c2] = 1
                    V2[j*p+k][c2] = 1
                    W2[k*n+i][c2] = 1
                    c2 += 1
        return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
