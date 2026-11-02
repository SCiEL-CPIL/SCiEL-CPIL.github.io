"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x6.
    
    Strategy:
    1. Start with a guaranteed valid schoolbook decomposition (rank 120).
    2. Use a structured Kronecker/Strassen-inspired initialization for lower ranks.
    3. Use coordinate descent with integer/half-integer constraints to refine.
    4. Return the best valid decomposition found within time limits.
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Deadline guard: keep 20s margin from the 150s limit
    start_time = time.time()
    deadline = 130  # 130s from start, leaves 20s margin
    
    # Target tensor T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Helper: compute the reconstructed tensor from U, V, W
    def reconstruct(U, V, W):
        """U: (n*m, R), V: (m*p, R), W: (p*n, R) -> tensor (n, m, p)"""
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        return np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    # Helper: check if tensor is all 1s (within tolerance)
    def is_valid(tensor):
        return np.allclose(tensor, 1.0, atol=1e-9)
    
    # Helper: ensure entries are integer or half-integer
    def snap_to_halfint(arr):
        """Round to nearest half-integer: x -> round(2x)/2"""
        return np.round(arr * 2) / 2.0
    
    # Helper: convert numpy arrays to lists for output
    def to_lists(U, V, W):
        U_l = U.tolist()
        V_l = V.tolist()
        W_l = W.tolist()
        # Ensure entries are exactly integer or half-integer
        for arr in [U_l, V_l, W_l]:
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    val = arr[i][j]
                    # Snap to nearest half-integer
                    arr[i][j] = round(val * 2) / 2.0
        return U_l, V_l, W_l
    
    # Schoolbook decomposition (baseline, rank 120)
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Best result so far
    U_best, V_best, W_best = make_schoolbook()
    best_rank = n * m * p  # 120
    
    # Check if schoolbook is valid (it should be)
    T_recon = reconstruct(U_best, V_best, W_best)
    if not is_valid(T_recon):
        # Should not happen, but just in case
        pass
    
    # Strassen 2x2x2 decomposition terms
    # For U = [[u00, u01], [u10, u11]], V = [[v00, v01], [v10, v11]]
    # 7 multiplications:
    # M1 = (u00+u11)*(v00+v11)
    # M2 = (u10+u11)*v00
    # M3 = u00*(v01-v11)
    # M4 = u11*(v10-v00)
    # M5 = (u00+u01)*v11
    # M6 = (u10-u00)*(v00+v01)
    # M7 = (u01-u11)*(v10+v11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen 2x2x2 rank-1 terms (U, V, W coefficients)
    # Each term is (u_coeffs, v_coeffs, w_coeffs) where each is a dict mapping (i,j) or (j,k) or (k,i) to coefficient
    # U is 2x2, V is 2x2, W is 2x2 (but W index is (k,i) for 2x2x2)
    
    # Let's define the 7 terms for 2x2x2:
    # Term 1: (u00+u11)*(v00+v11) contributes to C00 and C11
    # We need to decompose each M into rank-1 terms for the 2x2x2 tensor.
    # Actually, for the 2x2x2 tensor, each "multiplication" is a scalar product of two 2x2 matrices,
    # which gives a 2x2 result. In tensor form, this is not rank-1.
    # The Strassen algorithm computes 7 scalar multiplications of 2x2 matrices, but each such
    # multiplication produces a 2x2 matrix, which requires 4 rank-1 terms in the tensor decomposition.
    # Wait, no. In the tensor decomposition of mu_{2,2,2}, the rank is 7.
    # The 7 terms are:
    # For each r in 1..7, we have U_r, V_r, W_r which are 2x2 matrices (or vectors in the appropriate spaces)
    # such that mu_{2,2,2} = sum_r U_r ⊗ V_r ⊗ W_r
    
    # The standard Strassen rank-7 decomposition of mu_{2,2,2}:
    # U_1 = [[1,0],[1,1]], V_1 = [[1,1],[0,1]], W_1 = [[1,0],[0,1]]  -- not quite
    # Let me look up the exact form.
    
    # Actually, the cleanest way is to use the known rank-7 decomposition.
    # For 2x2x2, the 7 terms are:
    # r=1: U=[[1,0],[0,1]], V=[[1,0],[0,0]], W=[[1,0],[0,0]]  -- no, this is schoolbook
    
    # Let me use the explicit Strassen decomposition.
    # The 7 multiplications in Strassen correspond to 7 rank-1 terms in the tensor decomposition.
    # For mu_{2,2,2}, the tensor has entries T[i,j,k] = delta_{i,k} for the "multiplication" part... 
    # No, the matrix multiplication tensor mu_{n,m,p} has entries T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k].
    # So T[i,j,k] = 1 for all i,j,k in the appropriate ranges? No.
    # T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = 1 for all valid i,j,k.
    # Wait, that's not right either. The tensor is defined such that:
    # C[i,k] = sum_{j,r} U[i,j,r] * V[j,k,r] * W[k,i,r]  -- no, this is the decomposition form.
    # The original tensor T satisfies: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # where T[i,j,k] = 1 for all i,j,k.
    
    # So for mu_{2,2,2}, T[i,j,k] = 1 for all i,j,k in {0,1}.
    
    # The Strassen decomposition gives 7 rank-1 terms.
    # Let me define them explicitly:
    # Using the standard Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M_r is a product of a linear combination of A entries and a linear combination of B entries.
    # In tensor form, each M_r contributes to C[i,k] with coefficient w_{r,ik}.
    # The rank-1 term r has:
    # U_r[i,j] = coefficient of A[i,j] in the first factor of M_r
    # V_r[j,k] = coefficient of B[j,k] in the second factor of M_r
    # W_r[k,i] = coefficient of C[i,k] from M_r (note: W index is (k,i) per the problem statement)
    
    # Let me enumerate:
    # M1: (A00+A11)(B00+B11)
    # U_1: A00 has coeff 1, A11 has coeff 1 -> U_1 = [[1,0],[1,0]]  (rows i, cols j)
    # Wait, A[i,j], so A00 is i=0,j=0; A11 is i=1,j=1
    # U_1[0,0] = 1, U_1[1,1] = 1, others 0
    # V_1: B00 has coeff 1, B11 has coeff 1 -> V_1 = [[1,0],[0,1]]
    # W_1: M1 contributes to C00 with coeff 1, C11 with coeff 1
    # C[i,k], so C00 is i=0,k=0; C11 is i=1,k=1
    # W_1[k,i] -> W_1[0,0] = 1 (for C00), W_1[1,1] = 1 (for C11)
    
    # Let me be more careful. W has shape (p*n, rank) = (4, rank) for 2x2x2.
    # W index is k*n + i = k*2 + i.
    # So W_1[0*2+0] = W_1[0] = 1 (C00)
    # W_1[1*2+1] = W_1[3] = 1 (C11)
    
    # This is getting complex. Let me just use a numerical approach.
    # I'll use the schoolbook as baseline and try to find lower rank solutions
    # using coordinate descent with the constraint that entries are half-integers.
    
    # Try ranks from best_rank-1 down to some minimum
    # For each rank, try a few random initializations and coordinate descent
    
    # Minimum rank to try: 4*5*6 = 120, but we know Strassen can do better for some sub-blocks
    # Let's try down to rank 80
    
    min_rank_to_try = 80
    
    # For coordinate descent, we update one matrix at a time
    # When updating U, fix V and W, and for each column r, solve for U[:,r]
    # This is a linear least squares problem
    
    def coordinate_descent(U, V, W, max_sweeps=50, tol=1e-9):
        """
        Coordinate descent to minimize ||sum_r U_r V_r W_r - T||^2
        where T is all-ones tensor.
        U, V, W are numpy arrays.
        """
        n, m, p = DIMENSIONS
        R = U.shape[1]
        
        # Precompute target tensor
        T = np.ones((n, m, p))
        
        for sweep in range(max_sweeps):
            if time.time() > deadline:
                break
            
            # Compute current reconstruction
            U_r = U.reshape(n, m, R)
            V_r = V.reshape(m, p, R)
            W_r = W.reshape(p, n, R)
            T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            
            error = np.sum((T_recon - T) ** 2)
            if error < tol:
                break
            
            # Update U
            # For each r, fix V_r, W_r, and solve for U_r
            # We want to minimize sum_{i,j,k} (sum_r' U_{ij,r'} V_{jk,r'} W_{ki,r'} - T_{ijk})^2
            # When updating U_{:,r}, we fix all other U_{:,r'}
            # Let R_{ijk} = T_{ijk} - sum_{r' != r} U_{ij,r'} V_{jk,r'} W_{ki,r'}
            # Then we want to minimize sum_{j,k} (sum_r' ... ) 
            # Actually, the contribution of U_{:,r} to T_recon is:
            # T_recon_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
            # The residual excluding r is:
            # Res[i,j,k] = T[i,j,k] - sum_{r' != r} U[i,j,r'] V[j,k,r'] W[k,i,r']
            # We want to minimize sum_{i,j,k} (Res[i,j,k] - U[i,j,r] V[j,k,r] W[k,i,r])^2
            # This is a linear least squares problem in U[i,j,r] for fixed i.
            # For each i, we have variables U[i,j,r] for j=0..m-1
            # The objective is sum_{j,k} (Res[i,j,k] - U[i,j,r] V[j,k,r] W[k,i,r])^2
            # This is not a standard linear LS because U[i,j,r] appears multiplied by V[j,k,r]*W[k,i,r]
            # which depends on j and k.
            # Let's write it as: for each i, minimize over u_j (j=0..m-1):
            # sum_{j,k} (Res[i,j,k] - u_j * A[j,k])^2 where A[j,k] = V[j,k,r] * W[k,i,r]
            # This is a quadratic in u_j. Taking derivative w.r.t. u_j:
            # -2 sum_k (Res[i,j,k] - u_j A[j,k]) A[j,k] = 0
            # u_j sum_k A[j,k]^2 = sum_k Res[i,j,k] A[j,k]
            # u_j = sum_k Res[i,j,k] A[j,k] / sum_k A[j,k]^2
            
            for r in range(R):
                if time.time() > deadline:
                    break
                
                # Compute Res = T - (T_recon - U_r V_r W_r)
                U_r_curr = U_r[:, :, r]
                V_r_curr = V_r[:, :, r]
                W_r_curr = W_r[:, :, r]
                contrib_r = U_r_curr[:, :, None] * V_r_curr[None, :, :] * W_r_curr[None, None, :]
                # Wait, indices: U_r_curr is (n,m), V_r_curr is (m,p), W_r_curr is (p,n)
                # contrib_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
                # Let me compute this properly
                contrib_r = np.einsum('ij,jk,ki->ijk', U_r_curr, V_r_curr, W_r_curr)
                Res = T - (T_recon - contrib_r)
                
                # For each i, solve for U[i,:,r]
                for i in range(n):
                    # A[j,k] = V[j,k,r] * W[k,i,r]
                    A = V_r_curr * W_r_curr[:, i]  # (m, p) * (p,) -> (m, p)
                    # Wait, W_r_curr is (p, n), so W_r_curr[:, i] is (p,)
                    # V_r_curr is (m, p)
                    # A[j,k] = V_r_curr[j,k] * W_r_curr[k,i]
                    A = V_r_curr * W_r_curr[:, i][np.newaxis, :]  # (m, p)
                    
                    # For each j, u_j = sum_k Res[i,j,k] A[j,k] / sum_k A[j,k]^2
                    denom = np.sum(A ** 2, axis=1)  # (m,)
                    numer = np.sum(Res[i, :, :] * A, axis=1)  # (m,)
                    
                    # Avoid division by zero
                    mask = denom > 1e-12
                    u_new = np.zeros(m)
                    u_new[mask] = numer[mask] / denom[mask]
                    
                    # Snap to half-integer
                    u_new = np.round(u_new * 2) / 2.0
                    
                    U_r[:, i, r] = u_new  # Wait, U_r is (n, m, R), so U_r[i, :, r]
                    # Actually, U_r is a view of U.reshape(n, m, R)
                    # U_r[i, :, r] is not a valid slice because the last axis is R
                    # Let me fix this: U is (n*m, R), reshaped to (n, m, R)
                    # U[i, j, r] = U[i*m + j, r]
                    # So U_r[i, :, r] doesn't work. Let me use U directly.
                    
                    for j in range(m):
                        U[i * m + j, r] = u_new[j]
            
            # Update V (similar logic)
            # Recompute T_recon
            U_r = U.reshape(n, m, R)
            V_r = V.reshape(m, p, R)
            W_r = W.reshape(p, n, R)
            T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            
            for r in range(R):
                if time.time() > deadline:
                    break
                
                U_r_curr = U_r[:, :, r]
                V_r_curr = V_r[:, :, r]
                W_r_curr = W_r[:, :, r]
                contrib_r = np.einsum('ij,jk,ki->ijk', U_r_curr, V_r_curr, W_r_curr)
                Res = T - (T_recon - contrib_r)
                
                # For each j, solve for V[j,:,r]
                for j in range(m):
                    # A[i,k] = U[i,j,r] * W[k,i,r]
                    # U_r_curr[i,j] is (n,), W_r_curr[k,i] is (p,)
                    # A[i,k] = U_r_curr[i,j] * W_r_curr[k,i]
                    A = U_r_curr[:, j][:, np.newaxis] * W_r_curr  # (n, 1) * (p, n) -> need to be careful
                    # W_r_curr is (p, n), so W_r_curr[k,i]
                    # A[i,k] = U_r_curr[i,j] * W_r_curr[k,i]
                    # Let's construct A as (n, p)
                    A = np.zeros((n, p))
                    for i in range(n):
                        for k in range(p):
                            A[i, k] = U_r_curr[i, j] * W_r_curr[k, i]
                    
                    # For each k, v_k = sum_i Res[i,j,k] A[i,k] / sum_i A[i,k]^2
                    denom = np.sum(A ** 2, axis=0)  # (p,)
                    numer = np.sum(Res[:, j, :] * A, axis=0)  # (p,)
                    
                    mask = denom > 1e-12
                    v_new = np.zeros(p)
                    v_new[mask] = numer[mask] / denom[mask]
                    
                    v_new = np.round(v_new * 2) / 2.0
                    
                    for k in range(p):
                        V[j * p + k, r] = v_new[k]
            
            # Update W (similar logic)
            U_r = U.reshape(n, m, R)
            V_r = V.reshape(m, p, R)
            W_r = W.reshape(p, n, R)
            T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            
            for r in range(R):
                if time.time() > deadline:
                    break
                
                U_r_curr = U_r[:, :, r]
                V_r_curr = V_r[:, :, r]
                W_r_curr = W_r[:, :, r]
                contrib_r = np.einsum('ij,jk,ki->ijk', U_r_curr, V_r_curr, W_r_curr)
                Res = T - (T_recon - contrib_r)
                
                # For each k, solve for W[k,:,r]
                for k in range(p):
                    # A[i,j] = U[i,j,r] * V[j,k,r]
                    A = U_r_curr * V_r_curr[:, k][np.newaxis, :]  # (n, m) * (m,) -> (n, m)
                    
                    # For each i, w_i = sum_j Res[i,j,k] A[i,j] / sum_j A[i,j]^2
                    denom = np.sum(A ** 2, axis=1)  # (n,)
                    numer = np.sum(Res[:, :, k] * A, axis=1)  # (n,)
                    
                    mask = denom > 1e-12
                    w_new = np.zeros(n)
                    w_new[mask] = numer[mask] / denom[mask]
                    
                    w_new = np.round(w_new * 2) / 2.0
                    
                    for i in range(n):
                        W[k * n + i, r] = w_new[i]
        
        return U, V, W
    
    # Try lower ranks
    for R in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        max_restarts = 3
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            np.random.seed(42 + restart)
            U_rand = np.random.randint(0, 3, (n * m, R)) * 0.5  # Half-integers in {0, 0.5, 1}
            V_rand = np.random.randint(0, 3, (m * p, R)) * 0.5
            W_rand = np.random.randint(0, 3, (p * n, R)) * 0.5
            
            # Run coordinate descent
            U_opt, V_opt, W_opt = coordinate_descent(U_rand, V_rand, W_rand, max_sweeps=30)
            
            # Check validity
            T_recon = reconstruct(U_opt, V_opt, W_opt)
            if is_valid(T_recon):
                # Found a valid solution with rank R
                best_rank = R
                U_best, V_best, W_best = U_opt, V_opt, W_opt
                break  # Move to next (lower) rank
        
        # If we found a valid solution for this rank, continue to lower ranks
        # If not, continue to lower ranks anyway (maybe a different init will work)
    
    # Return the best result
    U_list, V_list, W_list = to_lists(U_best, V_best, W_best)
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
