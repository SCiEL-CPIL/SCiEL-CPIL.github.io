import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    # Target tensor T[i, j, k] = 1
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try ranks from target_rank - 1 downwards
    # Use ALS to find a solution
    
    deadline = time.time() + 100 # 100s deadline, 20s margin
    
    # Precompute indices
    # U indices: (i, j) -> i*m + j
    # V indices: (j, k) -> j*p + k
    # W indices: (k, i) -> k*n + i
    
    # We can vectorize the equations.
    # For each r, we have a term U_r (n*m) * V_r (m*p) * W_r (p*n).
    # The product is an outer product.
    
    # Let's try a few ranks.
    # Start with a known good rank if available, or just try to reduce.
    # For 3x4x5, schoolbook is 60.
    # Let's try 50, 45, 40...
    
    best_result = None
    current_rank = target_rank
    
    # Fallback to schoolbook
    U_school = np.zeros((n*m, current_rank))
    V_school = np.zeros((m*p, current_rank))
    W_school = np.zeros((p*n, current_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find a lower rank
    # We will use a randomized ALS approach
    
    def check_solution(U, V, W):
        # Compute the tensor
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        # Sum over R: U[i, j, r] * V[j, k, r] * W[k, i, r]
        
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        # Efficient contraction:
        # First contract U and V over j and r?
        # Term[i, k, r] = sum_j U[i, j, r] * V[j, k, r]
        # Then contract with W:
        # Result[i, k] = sum_r Term[i, k, r] * W[k, i, r]
        
        # Term[i, k, r] = sum_j U[i, j, r] V[j, k, r]
        # This is a batched matrix multiplication?
        # U is (n, m, R), V is (m, p, R).
        # We want to sum over j for each r.
        # This is like einsum('ijr,jkr->ikr', U_3d, V_3d)
        
        try:
            # np.einsum is available
            term = np.einsum('ijr,jkr->ikr', U_3d, V_3d)
            # Now sum over r: term[i, k, r] * W_3d[k, i, r]
            # W_3d is (p, n, R). We need W[k, i, r].
            # So we need to transpose W_3d to (n, p, R)? No, indices are (k, i, r).
            # term is (n, p, R) if we view i as row, k as col.
            # W_3d[k, i, r] corresponds to W_3d.T? No.
            # W_3d has shape (p, n, R). Index [k, i, r].
            # term has shape (n, p, R). Index [i, k, r].
            # We need to match indices i, k, r.
            # W_3d is indexed by k, i, r.
            # So we need to permute W_3d to (n, p, R) to match i, k, r?
            # W_3d[k, i, r] -> W_perm[i, k, r] = W_3d[k, i, r].
            # So W_perm = np.transpose(W_3d, (1, 0, 2))
            
            W_perm = np.transpose(W_3d, (1, 0, 2)) # (n, p, R)
            result = np.einsum('ikr,ikr->ik', term, W_perm)
            
            # Check if result is all 1s
            err = np.linalg.norm(result - 1.0)
            return err, result
            
        except Exception:
            return np.inf, None

    # Try ranks
    # We'll try a few random restarts for each rank
    for rank in range(target_rank - 1, 20, -1):
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(5): # 5 attempts per rank
            if time.time() > deadline:
                break
                
            # Initialize
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # ALS
            for step in range(50):
                if time.time() > deadline:
                    break
                    
                # Update U
                # Fix V, W.
                # Equation: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                # For a fixed i, j, the coefficient of U[i*m+j, r] is V[j*p+k, r] * W[k*n+i, r].
                # This depends on k.
                # So for each i, j, we have a system?
                # Actually, the equation is for all i, j, k.
                # Sum_r U[i*m+j, r] * C[i, j, k, r] = 1
                # Where C[i, j, k, r] = V[j*p+k, r] * W[k*n+i, r]
                
                # This is a linear system for U.
                # We can vectorize.
                # Let's construct the matrix A_u such that A_u @ U.flatten() = b
                # This might be too large (60 equations, 60*rank variables).
                # 60 * 60 = 3600. Small enough.
                
                # However, we can solve for each column of U independently?
                # No, U[i*m+j, r] is coupled with other r's?
                # The equation is sum over r. So all r's are coupled for a fixed i,j.
                # So for each i,j, we have a system of size (m*p) equations?
                # No, for fixed i,j, the equation must hold for all k.
                # So we have p equations for the vector U[i*m+j, :].
                # Sum_r U[i*m+j, r] * C[i, j, k, r] = 1 for all k.
                # This is a linear system: C_sub @ U_vec = 1
                # C_sub is (p, rank).
                
                # So for each i, j, we solve C_sub @ U_vec = 1.
                
                U_new = np.zeros((n*m, rank))
                for i in range(n):
                    for j in range(m):
                        # Construct C_sub (p, rank)
                        # C[i, j, k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # V_j = V[j*p:(j+1)*p, :] # (p, rank)
                        # W_i = W.T[i, :, :]? No.
                        # W is (p*n, rank). W[k*n+i, r].
                        # For fixed i, k varies.
                        # W_indices = [k*n + i for k in range(p)]
                        # W_sub = W[W_indices, :] # (p, rank)
                        
                        V_sub = V[j*p:(j+1)*p, :] # (p, rank)
                        W_indices = np.array([k*n + i for k in range(p)])
                        W_sub = W[W_indices, :] # (p, rank)
                        
                        C_sub = V_sub * W_sub # (p, rank)
                        
                        # Solve C_sub @ U_vec = 1
                        # Least squares
                        U_vec, _, _, _ = np.linalg.lstsq(C_sub, np.ones(p), rcond=None)
                        U_new[i*m + j, :] = U_vec
                        
                U = U_new
                
                # Similarly for V
                V_new = np.zeros((m*p, rank))
                for j in range(m):
                    for k in range(p):
                        # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # For fixed j, k, equation holds for all i.
                        # C[i, j, k, r] = U[i*m+j, r] * W[k*n+i, r]
                        # V_vec is (rank,)
                        # System: C_sub @ V_vec = 1 (for i in 0..n-1)
                        
                        U_indices = np.array([i*m + j for i in range(n)])
                        U_sub = U[U_indices, :] # (n, rank)
                        W_indices = np.array([k*n + i for i in range(n)])
                        W_sub = W[W_indices, :] # (n, rank)
                        
                        C_sub = U_sub * W_sub # (n, rank)
                        
                        V_vec, _, _, _ = np.linalg.lstsq(C_sub, np.ones(n), rcond=None)
                        V_new[j*p + k, :] = V_vec
                        
                V = V_new
                
                # Similarly for W
                W_new = np.zeros((p*n, rank))
                for k in range(p):
                    for i in range(n):
                        # Equation holds for all j.
                        # C[i, j, k, r] = U[i*m+j, r] * V[j*p+k, r]
                        # W_vec is (rank,)
                        # System: C_sub @ W_vec = 1 (for j in 0..m-1)
                        
                        U_indices = np.array([i*m + j for j in range(m)])
                        U_sub = U[U_indices, :] # (m, rank)
                        V_indices = np.array([j*p + k for j in range(m)])
                        V_sub = V[V_indices, :] # (m, rank)
                        
                        C_sub = U_sub * V_sub # (m, rank)
                        
                        W_vec, _, _, _ = np.linalg.lstsq(C_sub, np.ones(m), rcond=None)
                        W_new[k*n + i, :] = W_vec
                        
                W = W_new
                
            # Check solution
            err, _ = check_solution(U, V, W)
            if err < 1e-5:
                # Found a solution
                best_result = {"U": U, "V": V, "W": W}
                found = True
                break
        
        if found:
            break

    return best_result
