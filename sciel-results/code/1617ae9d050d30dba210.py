"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Schoolbook rank
    schoolbook_rank = n * m * p  # 42
    
    # Build the matmul tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a lower rank decomposition
    # Known lower bounds: max(n*p, m*p, n*m) = max(14, 21, 6) = 21
    # Also, the Strassen-type lower bounds may apply
    # For 2x3x7, let's try ranks from 21 upward
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Generate schoolbook solution as fallback
    U_school = np.zeros((n*m, schoolbook_rank), dtype=np.float64)
    V_school = np.zeros((m*p, schoolbook_rank), dtype=np.float64)
    W_school = np.zeros((p*n, schoolbook_rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try to find lower rank via ALS (Alternating Least Squares)
    # We'll try ranks from 21 to 41
    min_possible_rank = max(n*p, m*p, n*m)  # 21
    
    for target_rank in range(min_possible_rank, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts for each rank
        num_restarts = 3
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # ALS iterations
            converged = False
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each row i*m+j, solve least squares
                # T[i*m+j, :, :] = U[i*m+j, :] @ (V.T @ W)
                # Actually, the tensor decomposition is:
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed b,c, T[a,b,c] as a function of a is:
                # T[:,b,c] = U @ diag(V[b,:]*W[c,:])
                # So U[:,r] should be proportional to T[:,b,c] weighted by V[b,r]*W[c,r]
                
                # Let's use a different ALS formulation
                # Reshape T to a matrix for each mode
                # Mode-1 unfolding: T1[a, (b,c)] = T[a,b,c]
                # T1 = U @ (V * W) where (V*W) has entries V[b,r]*W[c,r]
                
                # Update U: T1 @ (V*W)^T = U @ I, so U = T1 @ (V*W)^T
                # But this is not quite right for ALS. Let me think more carefully.
                
                # The tensor is T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed a, T[a,b,c] = sum_r U[a,r] (V[b,r] W[c,r])
                # Let M_a be the matrix with M_a[b,c] = T[a,b,c]
                # Then M_a = sum_r U[a,r] (v_r ⊗ w_r) where v_r = V[:,r], w_r = W[:,r]
                # So [M_0; M_1; ...] = U @ K where K[r, (b,c)] = V[b,r] * W[c,r]
                
                # For ALS, we can update each factor by solving a least squares problem.
                # Update U: fix V, W. For each a, M_a = sum_r U[a,r] (v_r ⊗ w_r)
                # This is a linear system: [v_0⊗w_0, v_1⊗w_1, ..., v_{R-1}⊗w_{R-1}] @ U[a,:] = vec(M_a)
                
                # Build the design matrix for mode 1
                # K1: shape (m*p, target_rank), K1[(b,c), r] = V[b,r] * W[c,r]
                K1 = np.zeros((m*p, p*n, target_rank), dtype=np.float64)
                for r in range(target_rank):
                    K1[:, :, r] = np.outer(V[:, r], W[:, r])
                
                # Reshape K1 to (m*p*p*n, target_rank) and T to (n, m*p*p*n)
                K1_flat = K1.reshape(m*p*p*n, target_rank)
                T_flat = T.reshape(n, m*p*p*n)
                
                # Solve: T_flat @ K1_flat^T = U @ I
                # U = T_flat @ K1_flat^T
                U_new = T_flat @ K1_flat.T
                
                # Update V: fix U, W
                K2 = np.zeros((n*m, p*n, target_rank), dtype=np.float64)
                for r in range(target_rank):
                    K2[:, :, r] = np.outer(U[:, r], W[:, r])
                K2_flat = K2.reshape(n*m*p*n, target_rank)
                T_flat2 = T.transpose(1, 0, 2).reshape(m, n*m*p*n)
                V_new = T_flat2 @ K2_flat.T
                
                # Update W: fix U, V
                K3 = np.zeros((n*m, m*p, target_rank), dtype=np.float64)
                for r in range(target_rank):
                    K3[:, :, r] = np.outer(U[:, r], V[:, r])
                K3_flat = K3.reshape(n*m*m*p, target_rank)
                T_flat3 = T.transpose(2, 0, 1).reshape(p, n*m*m*p)
                W_new = T_flat3 @ K3_flat.T
                
                # Check convergence
                diff = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
                U, V, W = U_new, V_new, W_new
                
                if diff < 1e-6:
                    converged = True
                    break
            
            # Verify the decomposition
            # T_reconstructed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
            
            if error < 1e-6:
                best_rank = target_rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                break
        
        if best_rank == target_rank:
            break
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
