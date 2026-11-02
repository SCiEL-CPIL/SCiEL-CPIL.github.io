import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # ... (same Strassen construction as before) ...
    
    # Convert to numpy arrays for SVD
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Vectorize each term
    # Term r is u_r (4,) x v_r (12,) x w_r (12,)
    # Flatten to 576 vector
    # Index mapping: (i,j) -> i*m+j, (j,k) -> j*p+k, (k,i) -> k*n+i
    # The tensor T is in R^(4*12*12)
    # T[r] = vec(u_r) \otimes vec(v_r) \otimes vec(w_r) ?
    # No, the tensor component T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is not a simple Kronecker product of the vectors in the standard basis of R^4, R^12, R^12.
    # Wait, U is 4xR, V is 12xR, W is 12xR.
    # The tensor T is a 4x12x12 array?
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is a 3-mode tensor of size 4x12x12?
    # No, i is 0..3 (4 values), j is 0..11 (12 values), k is 0..11 (12 values).
    # But i,j,k are not independent.
    # i = i_row * m + i_col? No, U index is i*m+j.
    # So U index is 0..3.
    # V index is j*p+k. j is 0..1, k is 0..5. So 0..11.
    # W index is k*n+i. k is 0..5, i is 0..1. So 0..11.
    # The tensor T has indices (U_idx, V_idx, W_idx)?
    # No, the decomposition is sum_r u_r v_r w_r.
    # u_r is a vector in R^4.
    # v_r is a vector in R^12.
    # w_r is a vector in R^12.
    # The product u_r v_r w_r is a tensor of size 4x12x12.
    # T[a,b,c] = sum_r u_r[a] v_r[b] w_r[c].
    # The target tensor T_target is defined by the matrix multiplication identity.
    # T_target[a,b,c] = 1 if a,b,c correspond to a valid multiplication term, else 0.
    # What is the mapping?
    # a = i*m+j
    # b = j*p+k
    # c = k*n+i
    # For a valid term, we need consistency of i,j,k.
    # i = a // m, j = a % m
    # j' = b // p, k = b % p
    # k' = c // n, i' = c % n
    # We need i==i', j==j', k==k'.
    # So T_target[a,b,c] = 1 if (a//m == c%n) and (a%m == b//p) and (b%p == c//n) else 0.
    
    # Let's construct T_target as a 4x12x12 array.
    T_target = np.zeros((n*m, m*p, p*n))
    for a in range(n*m):
        i = a // m
        j = a % m
        for b in range(m*p):
            j_b = b // p
            k = b % p
            if j_b != j: continue
            for c in range(p*n):
                k_c = c // n
                i_c = c % n
                if k_c != k or i_c != i: continue
                T_target[a,b,c] = 1.0

    # Vectorize T_target to length 4*12*12 = 576
    T_vec = T_target.flatten()
    
    # Construct matrix M of size 576 x 21
    # M[:, r] = vec(u_r \otimes v_r \otimes w_r)
    # u_r = U_np[:, r]
    # v_r = V_np[:, r]
    # w_r = W_np[:, r]
    # The tensor u_r \otimes v_r \otimes w_r has components u_r[a] v_r[b] w_r[c].
    # This is exactly the outer product of the three vectors.
    # We can compute this as np.kron(np.kron(u_r, v_r), w_r)
    # But we need to be careful with the flattening order.
    # T_target.flatten() uses C-order (last index varies fastest).
    # T[a,b,c] -> index a*(12*12) + b*12 + c.
    # np.kron(u, v) gives (u[0]*v, u[1]*v, ...).
    # This matches the structure.
    
    M = np.zeros((n*m * m*p * p*n, rank))
    for r in range(rank):
        u_r = U_np[:, r]
        v_r = V_np[:, r]
        w_r = W_np[:, r]
        # Compute outer product
        # u_r is 4x1, v_r is 12x1, w_r is 12x1
        # Outer product is 4x12x12
        # We can use np.einsum or kron
        # np.kron(np.kron(u_r, v_r), w_r)
        # u_r (4), v_r (12) -> 48
        # 48 x 12 -> 576
        M[:, r] = np.kron(np.kron(u_r, v_r), w_r)
        
    # Compute SVD
    try:
        U_svd, S, Vt_svd = np.linalg.svd(M, full_matrices=False)
    except:
        # Fallback to rank 21
        return {"U": U, "V": V, "W": W}
        
    # Determine rank
    tol = 1e-9
    effective_rank = np.sum(S > tol)
    
    if effective_rank < rank:
        # Select pivot columns
        # Use QR decomposition with column pivoting to find independent columns
        # Or just take the first 'effective_rank' columns of Vt_svd?
        # Vt_svd is 21 x 21 (or 21 x 576 if full_matrices=True, but we used False so 21x21)
        # The columns of M are linear combinations of the left singular vectors.
        # We want to express T_vec in terms of a subset of the original columns M[:, r].
        # This is equivalent to finding a sparse representation.
        # However, if we just pick 'effective_rank' columns that form a basis, we can solve for the coefficients.
        
        # Find a basis of columns
        # We can use the fact that the first 'effective_rank' right singular vectors span the row space.
        # But we want to use the original columns.
        # Let's pick 'effective_rank' columns from M that are linearly independent.
        # We can do this by Gaussian elimination on M.T (21x576) or just checking ranks.
        
        basis_cols = []
        current_rank = 0
        for r in range(rank):
            if len(basis_cols) == effective_rank:
                break
            # Check if M[:, r] is in the span of M[:, basis_cols]
            # We can use np.linalg.matrix_rank
            if basis_cols:
                sub_M = M[:, basis_cols + [r]]
                if np.linalg.matrix_rank(sub_M) > current_rank:
                    basis_cols.append(r)
                    current_rank += 1
            else:
                if np.linalg.norm(M[:, r]) > tol:
                    basis_cols.append(r)
                    current_rank += 1
                    
        if len(basis_cols) < effective_rank:
             # Fallback
             return {"U": U, "V": V, "W": W}

        # Solve M[:, basis_cols] x = T_vec
        M_sub = M[:, basis_cols]
        # Least squares solution
        x, residuals, rank_sub, s = np.linalg.lstsq(M_sub, T_vec, rcond=tol)
        
        # Check if the solution is valid (residuals close to 0)
        if np.linalg.norm(M_sub @ x - T_vec) < 1e-6:
            # Construct new U, V, W
            new_rank = len(basis_cols)
            U_new = [[0] * new_rank for _ in range(n * m)]
            V_new = [[0] * new_rank for _ in range(m * p)]
            W_new = [[0] * new_rank for _ in range(p * n)]
            
            for new_r, old_r in enumerate(basis_cols):
                coeff = x[new_r]
                for i in range(n * m):
                    U_new[i][new_r] = U[i][old_r] * coeff
                for j in range(m * p):
                    V_new[j][new_r] = V[j][old_r] * coeff
                for k in range(p * n):
                    W_new[k][new_r] = W[k][old_r] * coeff
                    
            # Verify
            # (Optional, but good for safety)
            return {"U": U_new, "V": V_new, "W": W_new}
            
    return {"U": U, "V": V, "W": W}
