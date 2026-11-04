"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact decomposition for matrix multiplication 2x4 by 4x7.
    Strategy: Use numerical optimization to find a rank-r decomposition.
    We initialize with a random rank-r decomposition and use an iterative
    refinement method (alternating least squares) to minimize the error.
    If the error drops below a tolerance, we attempt to round the coefficients
    to half-integers (or integers) and verify exactness.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Leave 10s safety margin
    
    # Target: Find rank r such that there exist U, V, W with:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # U shape: (n*m, r)
    # V shape: (m*p, r)
    # W shape: (p*n, r)
    
    # We know rank is at least max(n*m, m*p, n*p) is NOT a strict lower bound for tensor rank,
    # but a simple lower bound is max(n, m, p) * something? 
    # Actually, a trivial lower bound is max(n*m, m*p, n*p) is false.
    # Rank of M_{n,m,p} >= max(n, m, p) is weak.
    # A better lower bound is max(n*m, m*p, n*p) / log2(max(n,m,p))? No.
    # Let's just try ranks starting from a reasonable guess.
    # Schoolbook rank is n*m*p = 56.
    # Strassen for 2x2x2 is 7 vs 8.
    # For 2x4x7, let's try ranks from 40 to 55.
    
    def verify_exact(U, V, W, n, m, p, tol=1e-9):
        """Verify exact decomposition."""
        rank = U.shape[1]
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Reshape for tensor contraction
        # U[i*m+j, c] -> U3[i, j, c]
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # We can do this by iterating over c or using einsum
        # T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        # This might be slow for high rank, but n,m,p are small.
        T = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            T += U3[:,:,c] * V3[:,:,c] * W3[:,:,c]
            
        # Check if T is all ones
        if not np.allclose(T, 1.0, atol=tol, rtol=0):
            return False, np.max(np.abs(T - 1.0))
        return True, 0.0

    def round_to_half_integers(arr):
        """Round array to nearest half-integer (multiple of 0.5)."""
        return np.round(arr * 2) / 2.0

    def optimize_decomposition(r, n, m, p, max_iter=1000, tol=1e-12):
        """
        Attempt to find a rank-r decomposition using alternating least squares.
        """
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Normalize to prevent scale issues
        for c in range(r):
            norm = np.linalg.norm(U[:, c])
            if norm > 0:
                U[:, c] /= norm
                V[:, c] *= norm
                W[:, c] *= norm
            else:
                U[:, c] = 1.0
                V[:, c] = 1.0
                W[:, c] = 1.0
        
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        prev_error = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U: For each (i,j), U[i,j,:] should minimize
            # || (1 - sum_c' V3[j,k,c'] W3[k,i,c'] * delta_{c,c'} ) ||_F
            # This is a linear system for each (i,j)
            # Let T_target = 1
            # Let R_jk = V3[j,k,:] (shape r)
            # Let S_ki = W3[k,i,:] (shape r)
            # We want U[i,j,:] such that sum_c U[i,j,c] * (sum_k V3[j,k,c] * W3[k,i,c]) = 1
            # Wait, the equation is:
            # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all k
            # This is not a simple linear system for U because V and W are coupled in the sum over k.
            # Actually, for fixed j, i:
            # We need vector u = U[i,j,:] such that:
            # sum_k (v_k * w_k^T) u^T = 1 (scalar? No, result is scalar 1 for each k)
            # Let M_k = outer(V3[j,k,:], W3[k,i,:])
            # We want sum_c u_c * (sum_k M_k[c,c]) = 1? No.
            
            # Let's re-evaluate the ALS step.
            # The error is E = || T - 1 ||_F^2 where T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # Fix V and W. Update U[i,j,:].
            # The term involving U[i,j,:] is:
            # T[i,j,k] = sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c])
            # Let A_k = V[j,k,:] * W[k,i,:] (element-wise product, shape r)
            # Then T[i,j,k] = dot(U[i,j,:], A_k)
            # We want dot(U[i,j,:], A_k) = 1 for all k.
            # This is an overdetermined linear system: A * u = 1
            # where A is a p x r matrix with rows A_k^T.
            # u = U[i,j,:]
            # We solve least squares: u = pinv(A) * ones(p)
            
            for i in range(n):
                for j in range(m):
                    # Build A (p x r)
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V3[j, k, :] * W3[k, i, :]
                    # Solve A @ u = 1
                    try:
                        u_new, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U3[i, j, :] = u_new
                    except:
                        pass
            
            # Update V: Fix U and W.
            # T[i,j,k] = sum_c (U[i,j,c] * W[k,i,c]) * V[j,k,c]
            # Let B_i = U[i,j,:] * W[k,i,:] (element-wise)
            # T[i,j,k] = dot(B_i, V[j,k,:])
            # We want dot(B_i, V[j,k,:]) = 1 for all i.
            # System: B @ v = 1 where B is n x r with rows B_i^T.
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, r))
                    for i in range(n):
                        B[i, :] = U3[i, j, :] * W3[k, i, :]
                    try:
                        v_new, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V3[j, k, :] = v_new
                    except:
                        pass
                        
            # Update W: Fix U and V.
            # T[i,j,k] = sum_c (U[i,j,c] * V[j,k,c]) * W[k,i,c]
            # Let C_j = U[i,j,:] * V[j,k,:]
            # T[i,j,k] = dot(C_j, W[k,i,:])
            # We want dot(C_j, W[k,i,:]) = 1 for all j.
            # System: C @ w = 1 where C is m x r with rows C_j^T.
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, r))
                    for j in range(m):
                        C[j, :] = U3[i, j, :] * V3[j, k, :]
                    try:
                        w_new, residuals, rank_c, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W3[k, i, :] = w_new
                    except:
                        pass
            
            # Calculate error
            T = np.zeros((n, m, p), dtype=np.float64)
            for c in range(r):
                T += U3[:,:,c] * V3[:,:,c] * W3[:,:,c]
            error = np.linalg.norm(T - 1.0)
            
            if error < tol:
                return U3.reshape(n*m, r), V3.reshape(m*p, r), W3.reshape(p*n, r)
            
            if error > prev_error:
                # Diverging, break
                break
            prev_error = error
            
        return None

    # Try ranks from a lower bound up to schoolbook
    # Lower bound heuristic: max(n*m, m*p, n*p) is 28.
    # Let's try from 30 to 55.
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_sb = np.zeros((n*m, n*m*p))
    V_sb = np.zeros((m*p, n*m*p))
    W_sb = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Try to find better rank
    for r in range(30, 56):
        if time.time() > deadline:
            break
        
        # Try a few random initializations for this rank
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Use a fixed seed for reproducibility within the rank loop
            np.random.seed(42 + r + attempt)
            
            U_opt, V_opt, W_opt = optimize_decomposition(r, n, m, p)
            
            if U_opt is not None:
                # Check if we can round to half-integers
                U_rounded = round_to_half_integers(U_opt)
                V_rounded = round_to_half_integers(V_opt)
                W_rounded = round_to_half_integers(W_opt)
                
                # Verify rounded version
                valid, err = verify_exact(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-9)
                if valid:
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                
                # If rounding failed, try to verify the unrounded float version?
                # The task requires integer/half-integer entries.
                # If we can't round, this rank might not have a rational solution of this form,
                # or we need a different rounding strategy.
                
                # Try rounding to integers?
                U_int = np.round(U_opt)
                V_int = np.round(V_opt)
                W_int = np.round(W_opt)
                valid_int, err_int = verify_exact(U_int, V_int, W_int, n, m, p, tol=1e-9)
                if valid_int:
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
                # If neither works, we might need to accept that this specific
                # numerical solution doesn't map to half-integers easily.
                # However, the problem states "integer/half-integer entries".
                # If the true rank is r, a rational solution exists.
                # We might need to use rational reconstruction.
                
                # For now, if we find a low error, we might return the float version
                # if the evaluator accepts floats? No, "integer/half-integer".
                # Let's assume if we find a very low error, we can try harder to round.
                
                # If error is very small, maybe just return the floats?
                # The contract says "integer/half-integer entries".
                # If I return floats, it might be invalid.
                # Let's stick to the schoolbook if we can't round.
                
                # However, if the error is extremely small, it's likely a valid decomposition
                # and rounding errors are the only issue.
                # Let's try a more aggressive rounding: nearest integer, then check.
                
                # If we fail to round, we skip this rank and try next.
                pass
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
