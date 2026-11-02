"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s margin before 150s kill
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, d] = 1 if a = i*m+j, b = j*p+k, d = k*n+i for some i,j,k
    # Actually, the tensor is: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n*m*p
    
    # Try to find lower rank decompositions using ALS
    # Start from best_rank - 1 and go down
    for target_rank in range(best_rank - 1, max(20, best_rank - 20), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(3):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # ALS iterations
            converged = False
            for it in range(50):
                if time.time() > deadline:
                    break
                
                # Update U: for each a, solve least squares
                # T[a, :, :] = sum_c U[a,c] * V[:,c] outer W[:,c]
                # Reshape: T[a] is (m*p, p*n), and we want U[a,c] such that
                # T[a] ≈ sum_c U[a,c] * (V[:,c] ⊗ W[:,c])
                # This is: T[a].reshape(m*p, p*n) ≈ sum_c U[a,c] * np.outer(V[:,c], W[:,c])
                # Let M = np.einsum('ac,bc->abc', V, W) shape (m*p, p*n, target_rank)
                # Then T[a] ≈ U[a, :] @ M.reshape(m*p, p*n, target_rank).transpose(1,2,0).reshape(p*n, m*p, target_rank)
                # Simpler: for each c, the contribution is U[a,c] * outer(V[:,c], W[:,c])
                # So T[a] = sum_c U[a,c] * outer(V[:,c], W[:,c])
                # This is a linear system in U[a, :]
                
                # Build the design matrix for U
                # For fixed a, T[a, b, d] = sum_c U[a,c] * V[b,c] * W[d,c]
                # Let's vectorize: for each a, we have a system:
                # sum_c U[a,c] * (V[:,c] ⊗ W[:,c]) = T[a, :, :].ravel()
                # Design matrix D_U has shape (m*p*p*n, target_rank) where D_U[idx, c] = V[b,c]*W[d,c] for idx = b*p*n+d
                
                # Precompute design matrices
                # D_U: for each (b,d), column c is V[b,c]*W[d,c]
                VW = np.einsum('bc,dc->bd c', V, W)  # (m*p, p*n, target_rank)
                D_U = VW.reshape(m*p * p*n, target_rank)
                
                # Solve for each a
                Ta = T.reshape(n*m, m*p*p*n)
                # U[a, :] = argmin ||D_U^T U[a,:] - Ta[a,:]||  (since T[a,b,d] = sum_c U[a,c] D_U[b*p*n+d, c])
                # Actually: Ta[a, idx] = sum_c U[a,c] * D_U[idx, c]
                # So U[a, :] = (D_U^T D_U)^{-1} D_U^T Ta[a, :]
                try:
                    U_new, _, _, _ = np.linalg.lstsq(D_U.T, Ta.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                # Similarly for V
                UW = np.einsum('ac,dc->ad c', U, W)  # (n*m, p*n, target_rank)
                D_V = UW.reshape(n*m * p*n, target_rank)
                Tv = T.reshape(n*m, m*p, p*n).transpose(1, 0, 2).reshape(m*p, n*m*p*n)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(D_V.T, Tv.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                # Similarly for W
                UV = np.einsum('ac,bc->ab c', U, V)  # (n*m, m*p, target_rank)
                D_W = UV.reshape(n*m * m*p, target_rank)
                Tw = T.reshape(n*m, m*p, p*n).transpose(2, 0, 1).reshape(p*n, n*m*m*p)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(D_W.T, Tw.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                
                # Check convergence
                T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
                if err < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Verify exactness with rational check
                T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
                if np.allclose(T_recon, T, atol=1e-10):
                    # Convert to rational if possible
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    break
        
        if best_rank <= target_rank:
            break
    
    # Convert to list format
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
