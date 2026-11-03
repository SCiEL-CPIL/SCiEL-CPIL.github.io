import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    deadline = time.time() + 145
    
    # Build the rank-36 Strassen solution
    U36, V36, W36 = _build_strassen_36(n, m, p)
    U36 = np.array(U36, dtype=np.float64)
    V36 = np.array(V36, dtype=np.float64)
    W36 = np.array(W36, dtype=np.float64)
    R36 = 36
    
    best_rank = R36
    best_U, best_V, best_W = U36.copy(), V36.copy(), W36.copy()
    
    # Check for linear dependencies among the rank-1 terms
    # Vectorize each rank-1 term T_r = u_r \otimes v_r \otimes w_r
    # T_r[i*m+j, j*p+k, k*n+i] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Flatten this tensor into a vector of size n*m*p
    
    num_terms = n * m * p
    # M will have shape (num_terms, R36)
    M = np.zeros((num_terms, R36), dtype=np.float64)
    
    for r in range(R36):
        u_r = U36[:, r]
        v_r = V36[:, r]
        w_r = W36[:, r]
        
        # Construct the rank-1 tensor
        # We can use einsum or broadcasting
        # T[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
        # Let's iterate or use vectorized ops
        # For efficiency, we can pre-compute the Kronecker products or use loops since dimensions are small
        # n=2, m=4, p=5. 40 elements per term. 36 terms. 1440 operations. Very fast.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * (m * p) + j * p + k
                    M[idx, r] = u_r[i * m + j] * v_r[j * p + k] * w_r[k * n + i]
                    
    # Compute SVD to find rank
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback to random search if SVD fails (unlikely)
        pass
    else:
        # Determine numerical rank
        tol = 1e-10
        rank_num = np.sum(s > tol)
        
        if rank_num < R36:
            # The terms are linearly dependent.
            # We can reduce the rank.
            # Find the null space vectors (dependencies)
            # The last (R36 - rank_num) columns of Vt (rows of V) span the null space
            # Actually, Vt is (R36, R36). The last rows correspond to zero singular values.
            # Let's identify the dependent columns.
            
            # A simpler approach: Find a set of rank_num linearly independent columns.
            # We can use QR decomposition with column pivoting or just check subsets.
            # Given small size, we can find the pivot columns of the SVD.
            
            # The basis vectors for the column space are the columns of M corresponding to the non-zero singular values?
            # No, the columns of U_svd (first rank_num) are the basis for the row space of M (which is the space of the tensors).
            # We want to express the original terms as linear combinations of a basis.
            
            # Let's find the rank_num linearly independent columns of M.
            # Use QR with pivoting
            Q, R_qr, piv = np.linalg.qr(M, mode='complete')
            # piv is the permutation of columns
            # The first rank_num columns of Q are the basis.
            # The coefficients are in the first rank_num rows of R_qr (permuted).
            
            # Actually, we just need to find one column that is a linear combination of the others.
            # If rank < R36, at least one column is dependent.
            
            # Let's try to drop the last column (index R36-1) and see if it's in the span of the others.
            # If yes, we can express T_{R36-1} = sum c_i T_i.
            # Then T = sum_{i < R36-1} T_i + T_{R36-1} = sum_{i < R36-1} (1 + c_i) T_i.
            # This gives a decomposition of rank R36-1.
            
            # To find c, solve M[:, :R36-1] @ c = M[:, R36-1]
            # If solvable, we have the coefficients.
            
            M_sub = M[:, :R36-1]
            b = M[:, R36-1]
            
            try:
                c, residuals, rank_sol, sv = np.linalg.lstsq(M_sub, b, rcond=None)
            except:
                c = None
            
            if c is not None and residuals is not None and residuals[0] < 1e-8:
                # We can drop the last term.
                # New U, V, W have R36-1 columns.
                # For i < R36-1, U_new[:, i] = U36[:, i] * (1 + c[i])
                # V_new[:, i] = V36[:, i]
                # W_new[:, i] = W36[:, i]
                # Wait, the scaling can be distributed.
                # T_i' = (1+c_i) T_i = ((1+c_i) u_i) \otimes v_i \otimes w_i
                # So we just scale U.
                
                best_rank = R36 - 1
                best_U = U36[:, :R36-1].copy()
                best_V = V36[:, :R36-1].copy()
                best_W = W36[:, :R36-1].copy()
                
                # Apply scaling
                for i in range(R36 - 1):
                    scale = 1.0 + c[i]
                    best_U[:, i] *= scale
                
                # Verify the solution
                # Check if the new decomposition is valid
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i * (m * p) + j * p + k
                            sum_val = 0.0
                            for r in range(best_rank):
                                sum_val += best_U[i * m + j, r] * best_V[j * p + k, r] * best_W[k * n + i, r]
                            if abs(sum_val - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    # Check if we can reduce further
                    # Repeat the process?
                    # For now, return this.
                    pass
                else:
                    # Fallback to original
                    best_rank = R36
                    best_U, best_V, best_W = U36.copy(), V36.copy(), W36.copy()

    # If no reduction found or validation failed, try random search for rank 35 as before?
    # The SVD check is deterministic and fast. If it fails, rank is likely 36.
    # But what if the dependency involves a different column?
    # We assumed column R36-1 is dependent. It might not be.
    # Let's try all columns.
    
    if best_rank == R36:
        for col_to_drop in range(R36):
            if time.time() > deadline:
                break
            
            M_sub = np.delete(M, col_to_drop, axis=1)
            b = M[:, col_to_drop]
            
            try:
                c, residuals, rank_sol, sv = np.linalg.lstsq(M_sub, b, rcond=None)
            except:
                continue
            
            if residuals is None or len(residuals) == 0:
                continue
                
            if residuals[0] < 1e-8:
                # Try to construct the solution
                new_rank = R36 - 1
                new_U = np.zeros((n * m, new_rank))
                new_V = np.zeros((m * p, new_rank))
                new_W = np.zeros((p * n, new_rank))
                
                idx = 0
                for i in range(R36):
                    if i == col_to_drop:
                        continue
                    scale = 1.0 + c[idx]
                    new_U[:, idx] = U36[:, i] * scale
                    new_V[:, idx] = V36[:, i]
                    new_W[:, idx] = W36[:, i]
                    idx += 1
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_t = i * (m * p) + j * p + k
                            sum_val = 0.0
                            for r in range(new_rank):
                                sum_val += new_U[i * m + j, r] * new_V[j * p + k, r] * new_W[k * n + i, r]
                            if abs(sum_val - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    best_rank = new_rank
                    best_U, best_V, best_W = new_U, new_V, new_W
                    break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _build_strassen_36(n, m, p):
    """Build Strassen rank-36 decomposition for 2x4 by 4x5 matrix multiplication."""
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        nonlocal col
        c = col_start
        U[i0 * m + j0, c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[j0 * p + k0, c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        col += 1

        c = col_start + 1
        U[(i0 + 1) * m + j0, c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[j0 * p + k0, c] += 1
        W[k0 * n + (i0 + 1), c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] -= 1
        col += 1

        c = col_start + 2
        U[i0 * m + j0, c] += 1
        V[j0 * p + (k0 + 1), c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] -= 1
        W[(k0 + 1) * n + i0, c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        col += 1

        c = col_start + 3
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[(j0 + 1) * p + k0, c] += 1
        V[j0 * p + k0, c] -= 1
        W[k0 * n + i0, c] += 1
        W[k0 * n + (i0 + 1), c] += 1
        col += 1

        c = col_start + 4
        U[i0 * m + j0, c] += 1
        U[i0 * m + (j0 + 1), c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] -= 1
        W[(k0 + 1) * n + i0, c] += 1
        col += 1

        c = col_start + 5
        U[(i0 + 1) * m + j0, c] += 1
        U[i0 * m + j0, c] -= 1
        V[j0 * p + k0, c] += 1
        V[j0 * p + (k0 + 1), c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        col += 1

        c = col_start + 6
        U[i0 * m + (j0 + 1), c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] -= 1
        V[(j0 + 1) * p + k0, c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] += 1
        col += 1

        return col_start + 7

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj), c] = 1
                V[(j0 + dj) * p + k0, c] = 1
                W[k0 * n + (i0 + di), c] = 1
                c += 1
        return c

    col = add_strassen_222(0, 0, 0, 0)
    col = add_strassen_222(0, 2, 0, col)
    col = add_strassen_222(0, 0, 2, col)
    col = add_strassen_222(0, 2, 2, col)
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W
