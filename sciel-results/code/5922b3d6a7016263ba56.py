"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import sys
import random

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Compute the tensor T from the decomposition
    # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    # Using einsum for speed and clarity
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Sum over rank dimension
    # T[i, j, k] = sum_c U_t[i, j, c] * V_t[j, k, c] * W_t[k, i, c]
    # We can do this by iterating over c or using tensordot tricks.
    # For small dims, explicit loop or einsum is fine.
    # Let's use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    # The target tensor for matrix multiplication A(n,m) * B(m,p) = C(n,p) is 1 everywhere?
    # No, the tensor T_MMM[i,j,k] is 1 if the indices correspond to a valid multiplication term.
    # Specifically, C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T[i,j,k] represents the coefficient of A[i,j]B[j,k] in C[i,k].
    # Since every pair (i,j) and (j,k) contributes exactly once to C[i,k], the coefficient is 1 for all valid i,j,k.
    # So the target tensor is indeed all ones of shape (n, m, p).
    
    target = np.ones((n, m, p))
    
    if not np.allclose(T, target, atol=tol, rtol=0):
        return False
    return True

def _strassen_block(U, V, W, i0, j0, k0, col_start, n, m, p):
    terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
    ]
    col = col_start
    for t in range(7):
        a_coeffs = terms[t][0]
        b_coeffs = terms[t][1]
        c_coeffs = terms[t][2]
        for di in range(2):
            for dj in range(2):
                U[(i0+di)*m + (j0+dj)][col] = a_coeffs[di*2+dj]
        for dj in range(2):
            for dk in range(2):
                V[(j0+dj)*p + (k0+dk)][col] = b_coeffs[dj*2+dk]
        for di in range(2):
            for dk in range(2):
                W[(k0+dk)*n + (i0+di)][col] = c_coeffs[di*2+dk]
        col += 1
    return col

def _schoolbook_block(U, V, W, i0, j0, k0, col_start, n, m, p):
    col = col_start
    for dj in range(2):
        for dk in range(2):
            U[i0*m + (j0+dj)][col] = 1
            V[(j0+dj)*p + (k0+dk)][col] = 1
            W[(k0+dk)*n + i0][col] = 1
            col += 1
    return col

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Construct the baseline Strassen+Schoolbook decomposition (Rank 88)
    base_rank = 88
    U_base = [[0] * base_rank for _ in range(n * m)]
    V_base = [[0] * base_rank for _ in range(m * p)]
    W_base = [[0] * base_rank for _ in range(p * n)]
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # Strassen block for i={0,1}
            col = _strassen_block(U_base, V_base, W_base, 0, j0, k0, col, n, m, p)
            # Schoolbook for i=2
            col = _schoolbook_block(U_base, V_base, W_base, 2, j0, k0, col, n, m, p)
    
    # Convert to numpy for processing
    U = np.array(U_base, dtype=float)
    V = np.array(V_base, dtype=float)
    W = np.array(W_base, dtype=float)
    
    if _verify(U, V, W, n, m, p):
        best_solution = (U, V, W, base_rank)
    else:
        # Fallback to pure schoolbook if baseline fails (shouldn't happen)
        best_solution = None
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        best_solution = (U, V, W, rank)

    # 2. Attempt Rank Reduction via CP-ALS (Alternating Least Squares)
    # Strategy:
    # We try to reduce the rank from 88 down to, say, 80.
    # We initialize with the 88-rank solution, zero out 'delta' columns, 
    # and run ALS to see if the remaining columns can adjust to fit the tensor.
    
    def run_als(U_init, V_init, W_init, max_iter=100, tol=1e-9):
        """
        Run CP-ALS to refine the factor matrices.
        U, V, W are numpy arrays of shape (size, rank).
        """
        n, m, p = DIMENSIONS
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        rank = U.shape[1]
        
        # Pre-compute target tensor (all ones)
        T = np.ones((n, m, p))
        
        # Reshape for easier manipulation
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        for it in range(max_iter):
            # Update U
            # We want to minimize || T - sum_c U_c \otimes V_c \otimes W_c ||
            # For each c, we fix V_c, W_c and solve for U_c.
            # This is a linear least squares problem for each c.
            # Let's vectorize the update for efficiency.
            
            # Compute the "effective" tensors for V and W
            # V_W_c = V_c \otimes W_c ? No, it's a tensor product of vectors.
            # The term is U_c[i,j] * V_c[j,k] * W_c[k,i].
            # This can be seen as:
            # T[i,j,k] = sum_c U_c[i,j] * (V_c[j,k] * W_c[k,i])
            # Let X_c[j,k,i] = V_c[j,k] * W_c[k,i]. This is a tensor of shape (m, p, n).
            # We want U_c to fit the residual R_c = T - sum_{c'!=c} U_c' * X_c'.
            
            # Efficient ALS update:
            # 1. Compute the "observation matrix" for U.
            #    The equation is: sum_c U_c[i,j] * X_c[j,k,i] = T[i,j,k]
            #    This is a 3-way tensor equation.
            #    We can solve for all U_c simultaneously using Kronecker products?
            #    Or iterate c?
            
            # Let's use the standard CP-ALS update rule which is efficient.
            # U[:, c] = (T \times_1 (V[:,c] \otimes W[:,c])) / ||V[:,c] \otimes W[:,c]||^2 ?
            # No, that's only if V and W are orthonormal.
            
            # Correct ALS:
            # Fix V, W.
            # For each c, we want to minimize || T - sum_{c'} U_{c'} \otimes V_{c'} \otimes W_{c'} ||_F^2
            # This is equivalent to solving:
            # (V_c \otimes W_c)^T (V_c \otimes W_c) U_c = (V_c \otimes W_c)^T (T - sum_{c'!=c} U_{c'} \otimes V_{c'} \otimes W_{c'})
            
            # Let's compute the residual tensor R = T - current_est
            current_est = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
            R = T - current_est
            
            # Now, for each c, we want to update U_c such that:
            # sum_{j,k} U_c[i,j] * V_c[j,k] * W_c[k,i] approx R[i,j,k] + sum_{c'!=c} ... (wait, this is getting circular)
            
            # Standard ALS step for U:
            # 1. Compute M_c = V_c \otimes W_c ? No.
            # Let's flatten the dimensions j and k into a single index l = j*p + k.
            # Then V_c is a vector of length m*p. W_c is a vector of length p*n.
            # This doesn't directly form a matrix product for U.
            
            # Let's use the "slice" method.
            # For each i, we have a matrix equation:
            # sum_c U[i,j,c] * (V_c[j,:] * W_c[:,i]) = T[i,j,:]
            # Let's define for each c and i:
            # B_c[i] = W_c[:, i] (vector of length p)
            # A_c[j] = V_c[j, :] (vector of length p)
            # Then the term is U[i,j,c] * dot(A_c[j], B_c[i])
            
            # This is still coupled.
            
            # Let's try a simpler, iterative approach:
            # For each c:
            #   1. Compute residual R_c = T - sum_{c'!=c} U_{c'} \otimes V_{c'} \otimes W_{c'}
            #   2. Solve for U_c: minimize || R_c - U_c \otimes V_c \otimes W_c ||
            #      This is a linear system for U_c.
            #      Let's construct the matrix A_c such that A_c * U_c.flatten() = R_c.flatten()
            #      A_c is of size (n*m*p) x (n*m).
            #      Row index: (i, j, k). Col index: (i', j').
            #      A_c[(i,j,k), (i',j')] = 1 if i=i' and j=j' else 0? No.
            #      The term is U[i',j',c] * V[j',k,c] * W[k,i',c].
            #      So for a fixed (i,j,k), the coefficient of U[i',j',c] is:
            #      delta(i,i') * V[j',k,c] * W[k,i',c]
            #      Since i=i', it's V[j',k,c] * W[k,i,c].
            #      So the row (i,j,k) of A_c has non-zeros only at columns (i,j').
            #      Specifically, A_c[(i,j,k), (i,j')] = V[j',k,c] * W[k,i,c].
            #      This matrix is block-diagonal with respect to i.
            #      For each i, we have a system of size (m*p) x m.
            #      Let's solve for each i separately.
            
            for c in range(rank):
                # Compute R_c
                # R_c = T - sum_{c'!=c} ...
                # To avoid O(rank^2) cost, we can use the fact that we just computed current_est.
                # But current_est includes U_c.
                # R_c = T - current_est + U_c \otimes V_c \otimes W_c
                
                # Let's compute the contribution of term c
                term_c = np.einsum('ijc,jkc,kic->ijk', U_t[:,:,c:c+1], V_t[:,:,c:c+1], W_t[:,:,c:c+1])
                R_c = T - current_est + term_c
                
                # Now solve for U_c
                # For each i:
                #   Minimize || R_c[i,:,:] - U_c[i,:] \otimes (V_c[:,i] \otimes W_c[:,i] ??) ||
                #   Wait, the structure is:
                #   R_c[i,j,k] = sum_{j'} U_c[i,j'] * V_c[j',k] * W_c[k,i]
                #   Let's define for fixed i and c:
                #   b_k = W_c[k,i] (vector of length p)
                #   M[j', k] = V_c[j', k]
                #   Then R_c[i,j,k] = sum_{j'} U_c[i,j'] * M[j',k] * b_k
                #   This is a weighted least squares problem for U_c[i,:].
                #   Let y_k = R_c[i,:,k] (vector of length m? No, R_c[i,j,k] is scalar)
                #   Let's fix k.
                #   R_c[i,j,k] = (sum_{j'} U_c[i,j'] * V_c[j',k]) * W_c[k,i]
                #   Let S[j] = sum_{j'} U_c[i,j'] * V_c[j',k]
                #   Then R_c[i,j,k] = S[j] * W_c[k,i]
                #   If W_c[k,i] != 0, then S[j] = R_c[i,j,k] / W_c[k,i]
                #   So we have:
                #   sum_{j'} U_c[i,j'] * V_c[j',k] = R_c[i,j,k] / W_c[k,i]  for all j
                #   This is a linear system for U_c[i,:].
                #   Let V_k be the column k of V_c (vector of length m).
                #   Let r_k be the vector of length m with entries r_k[j] = R_c[i,j,k] / W_c[k,i]
                #   Then V_k^T * U_c[i,:]^T = r_k ? No.
                #   U_c[i,:] is a row vector. V_c is m x p.
                #   (U_c[i,:] * V_c) [k] = sum_{j'} U_c[i,j'] * V_c[j',k]
                #   So we want U_c[i,:] * V_c = r_row, where r_row[k] = R_c[i,:,k] / W_c[k,i] ... wait, r_row depends on k?
                #   No, r_row[k] is a scalar for each k.
                #   So we have a system:
                #   U_c[i,:] * V_c = r_vec
                #   Where r_vec[k] = ?
                #   From R_c[i,j,k] = (U_c[i,:] * V_c[:,k]) * W_c[k,i]
                #   We need this to hold for all j.
                #   This implies that for a given k, the vector R_c[i,:,k] must be proportional to the vector (U_c[i,:] * V_c[:,k])?
                #   No, (U_c[i,:] * V_c[:,k]) is a scalar S_k.
                #   So R_c[i,j,k] = S_k * W_c[k,i].
                #   This means R_c[i,:,k] is a constant vector (same for all j) multiplied by W_c[k,i]?
                #   No, R_c[i,j,k] depends on j.
                #   The equation is:
                #   For all j, k:
                #   R_c[i,j,k] = sum_{j'} U_c[i,j'] V_c[j',k] W_c[k,i]
                #   Let's fix i.
                #   This is a linear system for U_c[i,:] (length m).
                #   Variables: u_j' for j'=0..m-1.
                #   Equations: for each j, k:
                #   sum_{j'} u_j' V_c[j',k] W_c[k,i] = R_c[i,j,k]
                #   Let's define a matrix A of size (m*p) x m.
                #   Row index: (j, k). Col index: j'.
                #   A[(j,k), j'] = V_c[j',k] * W_c[k,i]
                #   Then A * u = r, where r[(j,k)] = R_c[i,j,k]
                #   This is an overdetermined system (m*p equations, m unknowns).
                #   We can solve using least squares: u = lstsq(A, r)
                
                A = np.zeros((m * p, m))
                r = R_c[i, :, :].flatten() # Shape (m*p,)
                
                for j in range(m):
                    for k in range(p):
                        # A[(j,k), j'] = V_c[j', k] * W_c[k, i]
                        # V_c is (m, p). So V_c[j', k] is the element at row j', col k.
                        # W_c is (p, n). So W_c[k, i] is the element at row k, col i.
                        A[j*p+k, :] = V_c[:, k] * W_c[k, i]
                
                # Solve A u = r
                try:
                    u_sol, residuals, rank_A, s = np.linalg.lstsq(A, r, rcond=None)
                    U_t[i, :, c] = u_sol
                except:
                    pass

            # Update V and W similarly?
            # Due to time constraints and complexity, let's just do one full pass of U updates
            # and check if the error decreases. If it does, we might converge.
            # However, ALS usually cycles U, V, W.
            # Let's implement V and W updates similarly for completeness if time permits.
            
            # Update V
            for c in range(rank):
                term_c = np.einsum('ijc,jkc,kic->ijk', U_t[:,:,c:c+1], V_t[:,:,c:c+1], W_t[:,:,c:c+1])
                R_c = T - current_est + term_c
                
                for j in range(m):
                    # Fix j.
                    # R_c[i,j,k] = sum_{k'} U_c[i,j] V_c[j,k'] W_c[k',i] ... wait, indices are tricky.
                    # R_c[i,j,k] = U_c[i,j] V_c[j,k] W_c[k,i]
                    # For fixed j and c, we want to find V_c[j,:] (length p).
                    # Equations for all i, k:
                    # R_c[i,j,k] = U_c[i,j] V_c[j,k] W_c[k,i]
                    # Let v_k = V_c[j,k].
                    # R_c[i,j,k] = U_c[i,j] W_c[k,i] v_k
                    # This is a system for v (length p).
                    # Variables: v_k for k=0..p-1.
                    # Equations: for each i, k:
                    # U_c[i,j] W_c[k,i] v_k = R_c[i,j,k]
                    # This is a diagonal system!
                    # For each k, we have multiple equations (one for each i):
                    # U_c[i,j] W_c[k,i] v_k = R_c[i,j,k]
                    # So v_k = R_c[i,j,k] / (U_c[i,j] W_c[k,i])
                    # We can solve by least squares over i for each k.
                    
                    V_row = np.zeros(p)
                    for k in range(p):
                        # Gather equations for this k
                        # For all i:
                        # U_c[i,j] W_c[k,i] v_k = R_c[i,j,k]
                        # Let's form a small system or just take the mean if consistent.
                        # Actually, this is 1 variable v_k, n equations.
                        # A_i = U_c[i,j] * W_c[k,i]
                        # b_i = R_c[i,j,k]
                        # Minimize sum_i (A_i v_k - b_i)^2
                        # v_k = sum(A_i b_i) / sum(A_i^2)
                        
                        A_vals = U_c[:, j] * W_c[k, :] # Shape (n,)
                        b_vals = R_c[:, j, k] # Shape (n,)
                        
                        denom = np.dot(A_vals, A_vals)
                        if denom > 1e-10:
                            V_row[k] = np.dot(A_vals, b_vals) / denom
                        else:
                            V_row[k] = 0
                    V_t[j, :, c] = V_row

            # Update W
            for c in range(rank):
                term_c = np.einsum('ijc,jkc,kic->ijk', U_t[:,:,c:c+1], V_t[:,:,c:c+1], W_t[:,:,c:c+1])
                R_c = T - current_est + term_c
                
                for k in range(p):
                    # Fix k.
                    # R_c[i,j,k] = U_c[i,j] V_c[j,k] W_c[k,i]
                    # For fixed k and c, we want to find W_c[k,:] (length n).
                    # Equations for all i, j:
                    # R_c[i,j,k] = U_c[i,j] V_c[j,k] W_c[k,i]
                    # Let w_i = W_c[k,i].
                    # R_c[i,j,k] = (U_c[i,j] V_c[j,k]) w_i
                    # Variables: w_i for i=0..n-1.
                    # Equations: for each i, j:
                    # U_c[i,j] V_c[j,k] w_i = R_c[i,j,k]
                    # For each i, we have m equations.
                    # A_j = U_c[i,j] * V_c[j,k]
                    # b_j = R_c[i,j,k]
                    # w_i = sum(A_j b_j) / sum(A_j^2)
                    
                    W_row = np.zeros(n)
                    for i in range(n):
                        A_vals = U_c[i, :] * V_c[:, k] # Shape (m,)
                        b_vals = R_c[i, :, k] # Shape (m,)
                        
                        denom = np.dot(A_vals, A_vals)
                        if denom > 1e-10:
                            W_row[i] = np.dot(A_vals, b_vals) / denom
                        else:
                            W_row[i] = 0
                    W_t[k, :, c] = W_row

            # Check convergence
            current_est = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
            err = np.linalg.norm(T - current_est)
            if err < tol:
                break
            
            if time.time() > deadline - 5:
                break

        return U_t.reshape(n*m, rank), V_t.reshape(m*p, rank), W_t.reshape(p*n, rank)

    # Try to reduce rank from 88 to 80, 75, etc.
    # We will try a few target ranks.
    target_ranks = [85, 82, 80]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
            
        # Initialize with the first 'target_rank' columns of the base solution
        # and zero out the rest.
        U_try = U[:, :target_rank].copy()
        V_try = V[:, :target_rank].copy()
        W_try = W[:, :target_rank].copy()
        
        # Add some noise to help escape local minima? Or just start from base.
        # Let's start from base.
        
        U_ref, V_ref, W_ref = run_als(U_try, V_try, W_try, max_iter=50)
        
        if _verify(U_ref, V_ref, W_ref, n, m, p, tol=1e-5):
            best_solution = (U_ref, V_ref, W_ref, target_rank)
            # If we found a valid solution, we can try to go lower immediately
            # but we must respect the time limit.
            # Let's try the next lower target.
            continue
        else:
            # If it fails, we can try a random restart with the same rank
            # but with random initialization of the "free" parameters?
            # For now, we just move to the next target rank.
            pass

    # Final check
    if best_solution is None:
        # Return the base solution
        U_final, V_final, W_final, rank_final = U, V, W, base_rank
    else:
        U_final, V_final, W_final, rank_final = best_solution

    # Convert back to list of lists if required, or return numpy arrays?
    # The interface says: {'U', 'V', 'W'}: factor matrices of shapes...
    # It doesn't specify type, but numpy is accepted in the seed.
    
    return {"U": U_final, "V": V_final, "W": W_final}

# EVOLVE-BLOCK-END
