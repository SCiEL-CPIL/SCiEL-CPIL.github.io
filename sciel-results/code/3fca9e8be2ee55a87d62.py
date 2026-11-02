"""Bootstrap MODEL-AND-OPTIMIZE approach for 6x6 matrix multiplication tensor decomposition.
Strategy: Start from a known valid decomposition (schoolbook rank 216), then attempt
rank reduction by merging columns where U and V vectors are linearly dependent.
We use a greedy approach: try to combine pairs of columns that have proportional U and V vectors.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 6x6 matrix multiplication.
    Uses a model-and-optimize approach: parameterize the decomposition and attempt
    to reduce rank by finding linear dependencies among the rank-1 components.
    """
    deadline = time.time() + 140  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition (rank 216)
    # Each term: U[row, col] = 1 if row corresponds to (i,j), V[col, col] = 1 if col corresponds to (j,k)
    # W[col, col] = 1 if it maps to output (k,i)
    
    rank_schoolbook = n * m * p  # 216
    
    # Build schoolbook factors
    U = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                # W index: W[k*n+i, r]
                W[k*n + i, col] = 1.0
                col += 1
    
    # Verify correctness of schoolbook
    # C = A @ B, where (A@B)[i,k] = sum_j A[i,j] * B[j,k]
    # Our decomposition: sum_col U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
    # For a single column corresponding to (i',j',k'), this contributes:
    #   1 * 1 * 1 to position (k',i') in W, which maps to output[i',k']
    # So summing over all columns: output[i,k] = sum_{j} A[i,j]*B[j,k] * (sum over matching i',j',k')
    # This is correct.
    
    # Now attempt rank reduction
    # Strategy: Find groups of columns where U vectors and V vectors are linearly dependent
    # If two columns have U[:,c1] = alpha * U[:,c2] and V[:,c1] = beta * V[:,c2],
    # then W[:,c1] must equal alpha*beta * W[:,c2] for them to be mergeable.
    # In that case, we can keep one column and scale W accordingly.
    
    # More generally, if a set of columns has U vectors spanning a d-dimensional space
    # and V vectors spanning a d-dimensional space, we can potentially reduce to d columns.
    
    # Let's use a greedy approach: repeatedly find pairs that can be merged.
    # Two columns c1, c2 can be merged if U[:,c1] and U[:,c2] are linearly dependent
    # AND V[:,c1] and V[:,c2] are linearly dependent.
    
    # For schoolbook, each column has U with a single 1 and V with a single 1.
    # No two columns have dependent U vectors (each has a different position with 1).
    # So schoolbook can't be reduced this way directly.
    
    # Alternative approach: Use block structure with Strassen at the 2x2 level.
    # 6 = 3 * 2, so we have a 3x3 block matrix with 2x2 blocks.
    # Use Strassen (rank 7) for each 2x2 multiplication.
    # Total rank: 9 * 3 * 7 = 189
    
    # But we can do better by using a smarter block decomposition.
    # Let's try 6 = 2 * 3, so 2x2 blocks with 3x3 submatrices? No, that's the same.
    
    # Actually, let's just implement the block-Strassen approach which gives rank 189.
    # Then try to reduce further.
    
    nb = 3  # number of blocks
    bs = 2  # block size
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    strassen_x = [
        [(0,0,1), (1,1,1)],      # M0: A00+A11
        [(1,0,1), (1,1,1)],      # M1: A10+A11
        [(0,0,1)],               # M2: A00
        [(1,1,1)],               # M3: A11
        [(0,0,1), (0,1,1)],      # M4: A00+A01
        [(1,0,1), (0,0,-1)],     # M5: A10-A00
        [(0,1,1), (1,1,-1)],     # M6: A01-A11
    ]
    strassen_y = [
        [(0,0,1), (1,1,1)],      # M0: B00+B11
        [(0,0,1)],               # M1: B00
        [(0,1,1), (1,1,-1)],     # M2: B01-B11
        [(1,0,1), (0,0,-1)],     # M3: B10-B00
        [(1,1,1)],               # M4: B11
        [(0,0,1), (0,1,1)],      # M5: B00+B01
        [(1,0,1), (1,1,1)],      # M6: B10+B11
    ]
    strassen_c = [
        [(0,0,1), (1,1,1)],      # C00 += M0, C11 += M0
        [(1,0,1), (1,1,-1)],     # C10 += M1, C11 -= M1
        [(0,1,1), (1,1,1)],      # C01 += M2, C11 += M2
        [(0,0,1), (1,0,1)],      # C00 += M3, C10 += M3
        [(0,0,-1), (0,1,1)],     # C00 -= M4, C01 += M4
        [(1,1,1)],               # C11 += M5
        [(0,0,1)],               # C00 += M6
    ]
    
    rank_bs = nb * nb * nb * 7  # 9 * 3 * 7 = 189
    
    U_bs = np.zeros((n*m, rank_bs), dtype=np.float64)
    V_bs = np.zeros((m*p, rank_bs), dtype=np.float64)
    W_bs = np.zeros((p*n, rank_bs), dtype=np.float64)
    
    col = 0
    for ib in range(nb):
        for kb in range(nb):
            for jb in range(nb):
                for s in range(7):
                    # A block: rows ib*bs..(ib+1)*bs, cols jb*bs..(jb+1)*bs
                    # B block: rows jb*bs..(jb+1)*bs, cols kb*bs..(kb+1)*bs
                    # C block: rows ib*bs..(ib+1)*bs, cols kb*bs..(kb+1)*bs
                    
                    for (lr, lc, coeff) in strassen_x[s]:
                        gi = ib * bs + lr
                        gj = jb * bs + lc
                        U_bs[gi * m + gj, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_y[s]:
                        gj = jb * bs + lr
                        gk = kb * bs + lc
                        V_bs[gj * p + gk, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_c[s]:
                        gi = ib * bs + lr
                        gk = kb * bs + lc
                        W_bs[gk * n + gi, col] += coeff
                    
                    col += 1
    
    # Now try to reduce rank from 189
    best_U, best_V, best_W = U_bs, V_bs, W_bs
    best_rank = rank_bs
    
    if time.time() > deadline - 15:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Compression strategy: For each group of 7 columns (one per block-triple),
    # check if the U and V vectors have linear dependencies that allow compression.
    # More generally, do a global compression via iterative column elimination.
    
    # Approach: Iteratively try to eliminate one column at a time.
    # Column r can be eliminated if there exist scalars alpha_r, beta_r such that:
    #   U[:,r] = sum_{r' in remaining} alpha_{r,r'} * U[:,r']
    #   V[:,r] = sum_{r' in remaining} beta_{r,r'} * V[:,r']
    #   W[:,r] = sum_{r' in remaining} alpha_{r,r'} * beta_{r,r'} * W[:,r']
    # This is a nonlinear constraint. Instead, use a simpler sufficient condition:
    # If U[:,r] is in the span of other U columns AND V[:,r] is in the span of other V columns,
    # AND the W constraint is satisfied, we can eliminate.
    
    # Even simpler: check if column r is a linear combination of others in U,
    # and the same combination works for V, and W is consistent.
    
    # Let's use a more efficient approach:
    # 1. Compute QR of U_bs to find which columns are "pivot" columns.
    # 2. For non-pivot columns, express them as combinations of pivot columns.
    # 3. Check if the same combination works for V and W.
    
    # Actually, the most effective approach for this structured tensor:
    # Try to find a basis change that reveals redundancy.
    
    # Let's try: for each column r, check if it can be written as a linear
    # combination of at most 2 other columns (in U and V simultaneously).
    # This is expensive, so use a smarter heuristic.
    
    # Heuristic: Group by (ib, kb, jb). Within each group of 7, the U vectors
    # depend only on (ib, jb) and the Strassen pattern. The V vectors depend
    # only on (jb, kb) and the Strassen pattern. So columns with the same (ib,jb)
    # have U vectors that are linear combinations of a small set.
    
    # Better: exploit the Kronecker structure. The full U matrix can be written
    # as a sum over (ib,jb) of block-diagonal contributions. Let's check the
    # numerical rank of U_bs, V_bs, W_bs.
    
    rank_U = np.linalg.matrix_rank(U_bs, tol=1e-10)
    rank_V = np.linalg.matrix_rank(V_bs, tol=1e-10)
    rank_W = np.linalg.matrix_rank(W_bs, tol=1e-10)
    
    # The tensor rank must be at least max(rank_U, rank_V, rank_W)
    # but can be much larger.
    
    # Try iterative column elimination with a linear-algebraic test:
    # For column r, try to express U[:,r] = U_rest @ x  (solve least squares)
    # Then check if V[:,r] ≈ V_rest @ x and W[:,r] ≈ W_rest @ x
    # (same x must work for all three). If so, we can eliminate column r.
    
    remaining = list(range(rank_bs))
    eliminated = set()
    
    # Sort columns by "importance" - eliminate columns that are most redundant first
    # Use the norm of the residual when projecting onto others as a proxy
    
    max_iterations = 50  # limit number of elimination attempts
    iter_count = 0
    
    while remaining and iter_count < max_iterations:
        if time.time() > deadline - 10:
            break
        
        # For each remaining column, try to eliminate it
        # Pick the column with smallest residual norm (most redundant)
        best_col = -1
        best_residual = float('inf')
        best_x = None
        
        for r in remaining:
            if len(remaining) <= 1:
                break
            
            # Build U_rest, V_rest, W_rest excluding column r
            others = [c for c in remaining if c != r]
            
            U_rest = U_bs[:, others]
            V_rest = V_bs[:, others]
            W_rest = W_bs[:, others]
            
            # Solve U_rest @ x = U_bs[:, r]
            try:
                x_u, residuals_u, _, _ = np.linalg.lstsq(U_rest, U_bs[:, r], rcond=None)
            except:
                continue
            
            # Check if same x works for V and W
            v_pred = V_rest @ x_u
            w_pred = W_rest @ x_u
            
            res_v = np.linalg.norm(V_bs[:, r] - v_pred)
            res_w = np.linalg.norm(W_bs[:, r] - w_pred)
            res_u = np.linalg.norm(U_bs[:, r] - U_rest @ x_u)
            
            total_res = res_u + res_v + res_w
            
            if total_res < 1e-8 and total_res < best_residual:
                best_residual = total_res
                best_col = r
                best_x = x_u
        
        if best_col >= 0:
            # Eliminate best_col
            others = [c for c in remaining if c != best_col]
            U_rest = U_bs[:, others]
            V_rest = V_bs[:, others]
            W_rest = W_bs[:, others]
            
            # Update W for the remaining columns: W[:, r] = sum x_i * W[:, i]
            # So we add the contribution to each remaining column
            # Actually, we just remove the column; the decomposition still works
            # because U[:,r]*V[:,r]*W[:,r] = (U_rest@x_u)*(V_rest@x_u)*(W_rest@x_u)
            # Wait, that's not right. The product is element-wise, not a scalar product.
            # 
            # The tensor entry is: sum_r U[i,r]*V[j,r]*W[k,r]
            # If U[:,r] = U_rest @ x, V[:,r] = V_rest @ x, W[:,r] = W_rest @ x
            # Then U[i,r]*V[j,r]*W[k,r] = (U_rest[i,:]@x)*(V_rest[j,:]@x)*(W_rest[k,:]@x)
            # This is NOT equal to sum_s U_rest[i,s]*V_rest[j,s]*W_rest[k,s] in general.
            # So this elimination is only valid if the triple product decomposes.
            
            # The correct condition: we need U[:,r]*V[:,r]*W[:,r] (as a rank-1 tensor)
            # to be in the span of {U[:,s]*V[:,s]*W[:,s] for s != r}.
            # This is much harder to check.
            
            # For now, only do the elimination if x has at most 1 nonzero component
            # (i.e., column r is proportional to a single other column).
            nz = np.nonzero(np.abs(x_u) > 1e-10)[0]
            if len(nz) == 1:
                idx = nz[0]
                scale = x_u[idx]
                # Verify: U[:,r] = scale * U[:, others[idx]]
                # V[:,r] = scale * V[:, others[idx]]  
                # W[:,r] = scale * W[:, others[idx]]
                ref_col = others[idx]
                if (np.allclose(U_bs[:, r], scale * U_bs[:, ref_col], atol=1e-9) and
                    np.allclose(V_bs[:, r], scale * V_bs[:, ref_col], atol=1e-9) and
                    np.allclose(W_bs[:, r], scale * W_bs[:, ref_col], atol=1e-9)):
                    remaining.remove(best_col)
                    eliminated.add(best_col)
                    iter_count += 1
                    continue
            
            # Try 2-nonzero case
            if len(nz) == 2:
                idx1, idx2 = nz[0], nz[1]
                s1, s2 = x_u[idx1], x_u[idx2]
                c1, c2 = others[idx1], others[idx2]
                
                # Check if U[:,r] = s1*U[:,c1] + s2*U[:,c2]
                # AND V[:,r] = s1*V[:,c1] + s2*V[:,c2]
                # AND W[:,r] = s1*W[:,c1] + s2*W[:,c2]
                # Then the rank-1 tensor U[:,r]*V[:,r]*W[:,r] is NOT simply
                # s1*(U[:,c1]*V[:,c1]*W[:,c1]) + s2*(U[:,c2]*V[:,c2]*W[:,c2])
                # because of cross terms. So this doesn't work directly.
                # Skip 2-nonzero case for now.
                pass
        
        # If no column can be eliminated, stop
        if best_col < 0:
            break
        
        iter_count += 1
    
    if len(remaining) < rank_bs:
        new_rank = len(remaining)
        final_U = U_bs[:, remaining]
        final_V = V_bs[:, remaining]
        final_W = W_bs[:, remaining]
        
        if new_rank < best_rank:
            best_U, best_V, best_W = final_U, final_V, final_W
            best_rank = new_rank
    
    # Additional attempt: try a different Strassen variant or mixed block sizes
    # 6 = 2*3: use 2x2 Strassen for some blocks and 3x3 for others?
    # Or: 6 = 1*6, 2*3, 3*2 - mixed approaches
    
    # Try: partition into blocks of size 1, 2, 3 in different ways
    # For 6x6, we can use a (1,2,3) block partition:
    # A is partitioned as [1+2+3] x [1+2+3] = 9 blocks
    # Use schoolbook (rank 1) for 1x1, Strassen (rank 7) for 2x2,
    # and a 3x3 algorithm for 3x3 blocks.
    
    # Known: 3x3 matrix multiplication can be done in rank 23 (Bini et al.)
    # or rank 27 (schoolbook). Let's use a simple 3x3 rank-23 if available.
    # For now, use schoolbook for 3x3 (rank 27) as a baseline.
    
    # Mixed (1,2,3) block partition:
    # Block sizes: A[i,j] block has size (s_i x s_j)
    # s = [1, 2, 3]
    # Number of blocks: 3x3 = 9 blocks for A, B, C
    # For each (ib, jb, kb) triple:
    #   rank contribution = rank_{s_ib x s_jb} * rank_{s_jb x s_kb} ... no
    #   Actually, for block (ib,jb) of A and (jb,kb) of B:
    #   The product is a (s_ib x s_kb) matrix computed from (s_ib x s_jb) and (s_jb x s_kb)
    #   Using schoolbook: s_ib * s_jb * s_kb scalar multiplications
    #   Using Strassen (if s_ib=s_jb=s_kb=2): 7 * (s_ib/2)^2 ... no, Strassen is for 2x2 only.
    
    # For the mixed approach, use schoolbook for all non-2x2 blocks:
    # Total rank = sum over (ib,jb,kb) of s_ib * s_jb * s_kb
    # = (1+2+3)^3 = 6^3 = 216. Same as schoolbook! Not helpful.
    
    # Better: use Strassen for 2x2 sub-blocks within the 3x3 blocks.
    # Partition 3 into 1+2. So block sizes become [1,1,2,2,3] or similar.
    # This gets complex. Let's stick with the current approach.
    
    # Final: try to use a 3x3 rank-23 algorithm for the 3x3 sub-blocks
    # if we partition 6 as 3+3, giving 4 blocks of size 3x3.
    # Rank = 4 * 4 * rank_3x3. With rank_3x3=23: 4*4*23 = 368. Worse!
    # With rank_3x3=27: 4*4*27 = 432. Much worse.
    
    # So the 2x2 Strassen approach (rank 189) is better than 3x3 approaches.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
