"""Refined program using iterative CP-ALS for rank minimization."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Compute the reconstructed tensor
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want T == 1 for all i,j,k
    
    # Efficient check using einsum
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Sum over rank dimension
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Note: W is indexed by (k, i) in the tensor definition W[k*n+i, c]
    # So W_r[k, i, c] corresponds to W[k*n+i, c]
    
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    # Check if T is close to 1
    return np.allclose(T, 1.0, atol=tol, rtol=0.0)

def _als_update(U, V, W, n, m, p, max_iter=20, tol=1e-12):
    """
    Perform CP-ALS updates to minimize the residual || T - sum U_c o V_c o W_c ||_F.
    Since T is the all-ones tensor, we are fitting the all-ones tensor.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    # Precompute slices for efficiency if needed, but for small rank/dims, direct computation is fine.
    
    for _ in range(max_iter):
        # Update U
        U_new = np.zeros_like(U)
        for c in range(rank):
            # We want to find U[:,c] such that U[:,c] o V[:,c] o W[:,c] approximates the residual
            # Residual R = T - sum_{c'!=c} U_{c'} o V_{c'} o W_{c'}
            # T is all ones.
            
            # Let's compute the contribution of all other terms to the tensor T_other
            # T_other[i,j,k] = sum_{c'!=c} U[i,j,c'] * V[j,k,c'] * W[k,i,c']
            
            # Efficient way:
            # Let A = V[:,c] o W[:,c] (this is not quite right for ALS, we need to solve for U)
            
            # Standard ALS for CP:
            # Minimize || T - U_c o V_c o W_c - sum_{c'!=c} U_{c'} o V_{c'} o W_{c'} ||^2
            # This is equivalent to finding U_c such that U_c is the projection of (T - T_other) onto the subspace spanned by V_c o W_c.
            
            # Let's construct the matrix for the linear system.
            # For each (i,j), we have variables U[i,j,c].
            # The equation is: sum_{j,k} (V[j,k,c] * W[k,i,c]) * U[i',j',c] ... wait, indices are tricky.
            
            # Let's use the matrix unfolding approach.
            # Fix V and W. We want to update U.
            # T_{ijk} = U_{ij,c} V_{jk,c} W_{ki,c}
            # This can be seen as: For a fixed c, we want U[:,c] to satisfy:
            # sum_{j,k} V_{jk,c} W_{ki,c} U_{i'j',c} ... No, the indices i,j,k are coupled.
            
            # Let's stick to the direct residual computation for robustness given small sizes.
            # n=3, m=4, p=8. Total elements 96.
            # Rank is at most 96.
            
            # Compute T_other
            T_other = np.zeros((n, m, p))
            for c_prime in range(rank):
                if c_prime == c:
                    continue
                # T_other += U[:,c'] o V[:,c'] o W[:,c']
                # U[:,c'] is (n*m), reshape to (n,m)
                # V[:,c'] is (m*p), reshape to (m,p)
                # W[:,c'] is (p*n), reshape to (p,n)
                
                U_c = U[:, c_prime].reshape(n, m)
                V_c = V[:, c_prime].reshape(m, p)
                W_c = W[:, c_prime].reshape(p, n)
                
                # Outer product sum
                # T_other[i,j,k] += U_c[i,j] * V_c[j,k] * W_c[k,i]
                # Use einsum
                T_other += np.einsum('ij,jk,ki->ijk', U_c, V_c, W_c)
            
            # Target for U_c is T - T_other = 1 - T_other
            target = np.ones((n, m, p)) - T_other
            
            # Now solve for U_c.
            # We want to minimize || target - U_c o V_c o W_c ||_F^2
            # Let A be a matrix where A[(i,j), (j,k,i)] = V[j,k,c] * W[k,i,c]
            # Then A * U_c.flatten() = target.flatten()
            
            # Construct A
            # Rows correspond to (i,j) in U. Size n*m.
            # Columns correspond to (j,k,i) in target. Size m*p*n.
            # But wait, the variable is U[i,j]. The term in the sum for a specific target element (i,j,k) is U[i,j] * V[j,k] * W[k,i].
            # So for a fixed target element (i,j,k), it only depends on U[i,j].
            # This means the system is decoupled for each (i,j)?
            # No, because U[i,j] appears in multiple target elements (for different k).
            # Specifically, U[i,j] appears in target[i,j,k] for all k.
            # So we have a least squares problem for each (i,j):
            # Minimize over u = U[i,j]: sum_k (target[i,j,k] - u * V[j,k] * W[k,i])^2
            # This is a simple 1-variable least squares!
            
            # Solution: u = sum_k target[i,j,k] * V[j,k] * W[k,i] / sum_k (V[j,k] * W[k,i])^2
            
            denom = np.zeros((n, m))
            numer = np.zeros((n, m))
            
            # V_c is (m,p), W_c is (p,n)
            # For each i, j:
            # V_j = V_c[j, :] (length p)
            # W_i = W_c[:, i] (length p)
            # term_k = V_j[k] * W_i[k]
            # denom[i,j] = sum_k term_k^2
            # numer[i,j] = sum_k target[i,j,k] * term_k
            
            # Vectorized:
            # term is (m, p, n) -> V_c[j,k] * W_c[k,i]
            # V_c shape (m, p), W_c shape (p, n)
            # term = np.einsum('jk,ki->jki', V_c, W_c)
            
            term = np.einsum('jk,ki->jki', V_c, W_c)
            
            # denom[i,j] = sum_k term[j,k,i]^2
            denom = np.sum(term**2, axis=1) # (m, n) -> wait, term is (j, k, i). Sum over k. Result (m, n).
            # We need denom[i,j]. So transpose.
            denom = denom.T # (n, m)
            
            # numer[i,j] = sum_k target[i,j,k] * term[j,k,i]
            # target is (n, m, p). term is (m, p, n).
            # We need to align indices.
            # target[i, j, k] * term[j, k, i]
            # Let's transpose target to (j, k, i) to match term?
            # target.T (1,0,2) -> (m, n, p). No.
            # target axes: i, j, k.
            # term axes: j, k, i.
            # We want sum_k target[i,j,k] * term[j,k,i].
            # This is a contraction over k.
            # Let's reshape/transpose to align.
            # target_ijk. term_jki.
            # We can do: np.einsum('ijk,jki->ij', target, term)
            
            numer = np.einsum('ijk,jki->ij', target, term)
            
            # Avoid division by zero
            mask = denom > 1e-15
            U_new[:, c] = np.where(mask, numer / np.where(mask, denom, 1.0), 0.0).flatten()
            
        U = U_new
        
        # Update V
        V_new = np.zeros_like(V)
        for c in range(rank):
            T_other = np.zeros((n, m, p))
            for c_prime in range(rank):
                if c_prime == c:
                    continue
                U_c = U[:, c_prime].reshape(n, m)
                V_c = V[:, c_prime].reshape(m, p)
                W_c = W[:, c_prime].reshape(p, n)
                T_other += np.einsum('ij,jk,ki->ijk', U_c, V_c, W_c)
            
            target = np.ones((n, m, p)) - T_other
            
            # Minimize over v = V[j,k]: sum_i (target[i,j,k] - U[i,j] * v * W[k,i])^2
            # v = sum_i target[i,j,k] * U[i,j] * W[k,i] / sum_i (U[i,j] * W[k,i])^2
            
            # U_c is (n,m), W_c is (p,n)
            # term[i,j,k] = U[i,j] * W[k,i]
            # term shape (n, m, p)
            term = np.einsum('ij,ki->ijk', U_c, W_c)
            
            # denom[j,k] = sum_i term[i,j,k]^2
            denom = np.sum(term**2, axis=0) # (m, p)
            
            # numer[j,k] = sum_i target[i,j,k] * term[i,j,k]
            numer = np.einsum('ijk,ijk->jk', target, term)
            
            mask = denom > 1e-15
            V_new[:, c] = np.where(mask, numer / np.where(mask, denom, 1.0), 0.0).flatten()
            
        V = V_new
        
        # Update W
        W_new = np.zeros_like(W)
        for c in range(rank):
            T_other = np.zeros((n, m, p))
            for c_prime in range(rank):
                if c_prime == c:
                    continue
                U_c = U[:, c_prime].reshape(n, m)
                V_c = V[:, c_prime].reshape(m, p)
                W_c = W[:, c_prime].reshape(p, n)
                T_other += np.einsum('ij,jk,ki->ijk', U_c, V_c, W_c)
            
            target = np.ones((n, m, p)) - T_other
            
            # Minimize over w = W[k,i]: sum_j (target[i,j,k] - U[i,j] * V[j,k] * w)^2
            # w = sum_j target[i,j,k] * U[i,j] * V[j,k] / sum_j (U[i,j] * V[j,k])^2
            
            # U_c is (n,m), V_c is (m,p)
            # term[i,j,k] = U[i,j] * V[j,k]
            # term shape (n, m, p)
            term = np.einsum('ij,jk->ijk', U_c, V_c)
            
            # denom[k,i] = sum_j term[i,j,k]^2
            # term axes: i, j, k. Sum over j (axis 1). Result (n, p).
            # We need denom for W[k,i], so indices (k, i).
            denom = np.sum(term**2, axis=1) # (n, p) -> this is (i, k).
            denom = denom.T # (k, n) -> (p, n). This matches W shape (p, n).
            
            # numer[k,i] = sum_j target[i,j,k] * term[i,j,k]
            # target axes: i, j, k. term axes: i, j, k.
            # Contract over j. Result (i, k).
            numer = np.einsum('ijk,ijk->ik', target, term)
            numer = numer.T # (k, i) -> (p, n).
            
            mask = denom > 1e-15
            W_new[:, c] = np.where(mask, numer / np.where(mask, denom, 1.0), 0.0).flatten()
            
        W = W_new
        
        # Check convergence (optional, but good for speed)
        # Compute residual norm
        # T_recon = np.einsum('ijc,jkc,kic->ijk', U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank))
        # err = np.linalg.norm(T_recon - 1.0)
        # if err < tol: break

    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # 1. Initialize with Schoolbook Decomposition (Rank 96)
    def _init_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
        return U, V, W

    U, V, W = _init_schoolbook()
    current_rank = n * m * p
    
    # 2. Iterative Pruning and ALS
    # We will try to reduce the rank by removing columns one by one.
    # Strategy:
    # a) Pick a column to remove (randomly or based on some heuristic).
    # b) Remove it from U, V, W.
    # c) Run ALS to see if the remaining columns can fit the tensor.
    # d) If the residual is small (exact decomposition), keep the reduction.
    # e) If not, revert and try another column.
    
    best_U, best_V, best_W = U, V, W
    best_rank = current_rank
    
    # To speed up, we can try removing columns in a specific order or randomly.
    # Since we have time, we can do multiple passes.
    
    # Helper to check if a set of columns forms an exact decomposition
    def _is_exact(U, V, W, tol=1e-8):
        return _verify(U, V, W, n, m, p, tol)
    
    # We will attempt to reduce rank from 96 down to as low as possible.
    # We'll use a "greedy" approach with randomization to avoid local minima.
    
    # Since ALS is expensive, we'll limit the number of iterations.
    # Also, we can use a cheaper check first:
    # Check if the rank of the Khatri-Rao product of the factor matrices is sufficient?
    # Or just run ALS and check residual.
    
    # Let's try a few random restarts with different pruning orders.
    # But first, let's try a systematic pruning.
    
    # Note: ALS might converge to a local minimum where the residual is not zero.
    # We need to be careful.
    
    # Alternative: Use a "Null Space" approach?
    # If we have rank R, and we remove one column, we have R-1 columns.
    # Can we adjust the remaining R-1 columns to exactly represent T?
    # This is a linear feasibility problem if we fix the "directions" of the vectors?
    # No, the directions change.
    
    # Let's stick to ALS.
    
    # To improve success rate, we can add random noise to the initial guess before ALS.
    
    attempts = 0
    max_attempts = 500 # Adjust based on time
    
    while time.time() < deadline - 10 and attempts < max_attempts:
        attempts += 1
        
        # Try to remove a random column
        if current_rank <= 1:
            break
            
        col_to_remove = np.random.randint(current_rank)
        
        # Remove column
        U_trial = np.delete(U, col_to_remove, axis=1)
        V_trial = np.delete(V, col_to_remove, axis=1)
        W_trial = np.delete(W, col_to_remove, axis=1)
        
        # Run ALS on the trial
        # Use fewer iterations for speed during search
        U_als, V_als, W_als = _als_update(U_trial, V_trial, W_trial, n, m, p, max_iter=10)
        
        # Check if exact
        if _is_exact(U_als, V_als, W_als, tol=1e-7):
            # Success! Keep the reduction
            U, V, W = U_als, V_als, W_als
            current_rank -= 1
            best_U, best_V, best_W = U, V, W
            best_rank = current_rank
        else:
            # Failure. Revert.
            # Optionally, we could keep the ALS result if it's "close" and try to polish later,
            # but for exact decomposition, we need to be strict.
            pass
        
        # If we've reduced rank significantly, we might want to randomize the starting point
        # to escape local minima.
        # But since we are pruning, the structure is fixed.
        
    # Final Polish: Run more ALS iterations on the best found solution to ensure precision
    if best_rank < n * m * p:
        U_polished, V_polished, W_polished = _als_update(best_U, best_V, best_W, n, m, p, max_iter=50)
        if _is_exact(U_polished, V_polished, W_polished, tol=1e-8):
            best_U, best_V, best_W = U_polished, V_polished, W_polished
    
    # Convert to list of lists for interface
    def _to_list(arr):
        return arr.tolist()
        
    return {
        "U": _to_list(best_U),
        "V": _to_list(best_V),
        "W": _to_list(best_W)
    }

# EVOLVE-BLOCK-END
