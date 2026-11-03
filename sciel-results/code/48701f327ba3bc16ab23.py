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
    # Look for linear dependencies among columns
    # If columns c1..ck have U vectors in a space of dimension d < k,
    # and similarly for V, we might be able to reduce.
    
    # Use SVD-based approach: find the effective rank of U and V submatrices
    # for groups of columns, then attempt to compress.
    
    best_U, best_V, best_W = U_bs, V_bs, W_bs
    best_rank = rank_bs
    
    # Try to identify and merge groups of columns that are linearly dependent
    # For simplicity, let's check if any subset of columns can be compressed
    
    # Time check
    if time.time() > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Attempt: find the numerical rank of U_bs and V_bs
    # If the columns of U are linearly dependent, we can potentially reduce rank
    
    # Compute the rank of U_bs (should be at most n*m = 36)
    # and rank of V_bs (at most m*p = 36)
    # The tensor rank is at least max(rank(U), rank(V), rank(W))
    
    # Let's try a different approach: use the fact that the 6x6 multiplication
    # tensor has known lower bounds and try known algorithms.
    
    # For now, return the block-Strassen result (rank 189)
    # which is better than schoolbook (216)
    
    if time.time() > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to further reduce by checking for mergeable columns
    # Two columns can be merged if U[:,c1] is proportional to U[:,c2]
    # and V[:,c1] is proportional to V[:,c2]
    
    # In our construction, each column has specific non-zero patterns.
    # Let's check for exact proportional relationships.
    
    # Verify the block-Strassen decomposition is correct before any reduction
    def verify_decomposition(U_mat, V_mat, W_mat, n=n, m=m, p=p, atol=1e-9):
        """Verify that sum_r U[:,r] * V[:,r] * W[:,r] gives correct matrix multiplication tensor."""
        rank = U_mat.shape[1]
        # Build the full tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For efficiency, use einsum
        # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
        # We need T[i,j,k] for all i,j,k
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a 3D tensor. Let's build it.
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U_mat[:, r]
            v_col = V_mat[:, r]
            w_col = W_mat[:, r]
            # u_col has non-zeros at positions i*m+j
            # v_col has non-zeros at positions j*p+k
            # w_col has non-zeros at positions k*n+i
            # For each (i,j) where u_col is non-zero, and (j,k) where v_col is non-zero,
            # and (k,i) where w_col is non-zero, add the product.
            # More efficiently:
            u_mat_2d = u_col.reshape(n, m)
            v_mat_2d = v_col.reshape(m, p)
            w_mat_2d = w_col.reshape(p, n)
            # T += u_mat_2d[i,j] * v_mat_2d[j,k] * w_mat_2d[k,i] for all i,j,k
            # This is: T[i,j,k] += u_mat_2d[i,j] * v_mat_2d[j,k] * w_mat_2d[k,i]
            # Use einsum
            T += np.einsum('ij,jk,ki->ijk', u_mat_2d, v_mat_2d, w_mat_2d)
        # Check against schoolbook: T[i,j,k] should be 1 if i,j,k form a valid multiplication path
        # Actually, the tensor for matrix multiplication is: T[i,j,k] = delta_{i,j,k} where
        # the multiplication is C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T[i,j,k] = 1 for all i,j,k (each term A[i,j]*B[j,k] contributes to C[i,k])
        # Wait, no. The tensor is defined such that C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all i,j,k.
        expected = np.ones((n, m, p))
        if not np.allclose(T, expected, atol=atol):
            return False
        return True
    
    if not verify_decomposition(U_bs, V_bs, W_bs):
        # Fallback to schoolbook if block-Strassen is wrong
        U = np.zeros((n*m, n*m*p), dtype=np.float64)
        V = np.zeros((m*p, n*m*p), dtype=np.float64)
        W = np.zeros((p*n, n*m*p), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now attempt rank reduction with improved group-based merge logic
    # First pass: pairwise proportional merges
    reduced_U = [U_bs[:, c].copy() for c in range(rank_bs)]
    reduced_V = [V_bs[:, c].copy() for c in range(rank_bs)]
    reduced_W = [W_bs[:, c].copy() for c in range(rank_bs)]
    
    used = [True] * rank_bs
    merged = 0
    
    def find_ratio(vec_a, vec_b, tol=1e-9):
        """Find scalar r such that vec_b = r * vec_a, or None if no such r exists."""
        r = None
        for idx in range(len(vec_a)):
            if abs(vec_a[idx]) > 1e-12:
                if abs(vec_b[idx]) > 1e-12:
                    candidate = vec_b[idx] / vec_a[idx]
                    if r is None:
                        r = candidate
                    elif abs(candidate - r) > tol:
                        return None
                else:
                    if r is not None and abs(r) > tol:
                        return None
                    r = 0.0
        if r is not None:
            if not np.allclose(vec_b, r * vec_a, atol=tol):
                return None
        return r
    
    for i in range(rank_bs):
        if not used[i]:
            continue
        for j in range(i+1, rank_bs):
            if not used[j]:
                continue
            if time.time() > deadline - 15:
                break
            
            ui = reduced_U[i]
            uj = reduced_U[j]
            vi = reduced_V[i]
            vj = reduced_V[j]
            
            ratio_u = find_ratio(ui, uj)
            if ratio_u is None:
                continue
            
            ratio_v = find_ratio(vi, vj)
            if ratio_v is None:
                continue
            
            wi = reduced_W[i]
            wj = reduced_W[j]
            reduced_W[i] = wi + ratio_u * ratio_v * wj
            used[j] = False
            merged += 1
    
    # Second pass: find groups of columns where U vectors span a lower-dimensional space
    # and V vectors span a lower-dimensional space, then compress
    if time.time() < deadline - 20:
        # For each group of columns that share the same support pattern in U and V,
        # check if they can be compressed
        active_indices = [i for i in range(rank_bs) if used[i]]
        
        # Group columns by their U support pattern (which positions are non-zero)
        u_patterns = {}
        for idx in active_indices:
            u_col = reduced_U[idx]
            # Create a hashable key from the support pattern
            support = tuple(np.where(np.abs(u_col) > 1e-12)[0])
            u_patterns.setdefault(support, []).append(idx)
        
        # For each group, check if V vectors are also linearly dependent
        for support, indices in u_patterns.items():
            if len(indices) <= 1:
                continue
            if time.time() > deadline - 15:
                break
            
            # Check if V vectors in this group are linearly dependent
            v_sub = np.column_stack([reduced_V[idx] for idx in indices])
            # Compute rank of v_sub
            try:
                v_rank = np.linalg.matrix_rank(v_sub, tol=1e-9)
            except:
                continue
            
            if v_rank < len(indices):
                # Try to compress: express each V vector as linear combination of basis
                # and adjust W accordingly
                # For simplicity, just try pairwise merges within this group
                for ii in range(len(indices)):
                    for jj in range(ii+1, len(indices)):
                        idx_i = indices[ii]
                        idx_j = indices[jj]
                        if not used[idx_i] or not used[idx_j]:
                            continue
                        
                        ui = reduced_U[idx_i]
                        uj = reduced_U[idx_j]
                        vi = reduced_V[idx_i]
                        vj = reduced_V[idx_j]
                        
                        ratio_u = find_ratio(ui, uj)
                        if ratio_u is None:
                            continue
                        
                        ratio_v = find_ratio(vi, vj)
                        if ratio_v is None:
                            continue
                        
                        wi = reduced_W[idx_i]
                        wj = reduced_W[idx_j]
                        reduced_W[idx_i] = wi + ratio_u * ratio_v * wj
                        used[idx_j] = False
                        merged += 1
    
    if merged > 0:
        new_rank = sum(used)
        final_U = np.column_stack([reduced_U[i] for i in range(rank_bs) if used[i]])
        final_V = np.column_stack([reduced_V[i] for i in range(rank_bs) if used[i]])
        final_W = np.column_stack([reduced_W[i] for i in range(rank_bs) if used[i]])
        
        if new_rank < best_rank:
            if verify_decomposition(final_U, final_V, final_W):
                best_U, best_V, best_W = final_U, final_V, final_W
                best_rank = new_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
