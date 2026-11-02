"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = delta (we need C[i,k] = sum_j A[i,j]*B[j,k])
    # A rank-r decomposition: T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # where (U⊗V⊗W)[i,j,k] = U[i,c]*V[j,c]*W[k,c]
    # We need: for all i,k, sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta_{i,k is in range}
    # Actually the tensor is: T[i,j,k] = 1 for the multiplication, and we decompose it.
    
    # Let me reformulate. The matrix multiplication tensor M has:
    # M[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The rank is the minimum r such that M = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where (u⊗v◊w)[i,j,k] = u[i]*v[j]*w[k]
    # And we need: for each (i,k), sum_j M[i,j,k] * A[i,j]*B[j,k] = C[i,k]
    # This means the decomposition must satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k where the product contributes
    # Wait, I need to be more careful.
    
    # The correct condition: The tensor T of shape (n*m, m*p, p*n) where
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0
    # But actually for matrix multiplication, each (i,j,k) triple gives one term.
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    
    # A rank-r decomposition: T = sum_{c=0}^{r-1} U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # means T[a,b,c] = sum_{d=0}^{r-1} U[a,d]*V[b,d]*W[c,d]
    
    # We need this to equal the indicator that (a,b,c) corresponds to a valid (i,j,k) triple.
    
    # Let me build the target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Now search for low-rank decompositions
    # Try ranks from a lower bound up to schoolbook
    
    # Lower bound: max(n*m, m*p, p*n) = max(8, 32, 16) = 32
    # But that's not tight. The actual lower bound is at least the max of the matrix ranks.
    # For matrix multiplication, a known lower bound is max(n*m, m*p, p*n) but that's not always tight.
    
    # Let's try to find good decompositions using random initialization and alternating least squares
    
    np.random.seed(42)
    
    for target_rank in range(30, min(rank_schoolbook + 1, 45)):
        if time.time() > deadline:
            break
            
        found = False
        for trial in range(50):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Alternating minimization: fix two, solve for the third
            converged = False
            for it in range(200):
                if time.time() > deadline:
                    break
                    
                # Fix V, W, solve for U
                # T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
                # For each a: T[a,:,:] = sum_d U[a,d] * (V[:,d]*W[:,d])
                # This is: T[a] = U[a,:] @ (V*W) where (V*W)[b,c] = sum_d V[b,d]*W[c,d]
                # So U[a,:] = T[a,:] / (V*W) in a least squares sense
                # Actually: T[a] = sum_d U[a,d] * outer(V[:,d], W[:,d]).reshape(-1)
                # Let's define: for each d, let M_d = outer(V[:,d], W[:,d])
                # Then T[a] = sum_d U[a,d] * M_d
                # So [T[0],...,T[nm-1]] = U @ [M_0, M_1, ..., M_{r-1}]
                # Let A_mat = np.stack([M_d.flatten() for d in range(r)], axis=1)  # shape (nm*mp*pn, r)
                # Then T_flat = U @ A_mat
                # U = T_flat @ A_mat.T @ inv(A_mat @ A_mat.T) if A_mat @ A_mat.T is invertible
                
                # Build the design matrix
                # A_mat[a, d] = V[b,a]*W[c,a] where a = i*m+j, b = j*p+k, c = k*n+i
                # But a, b, c are coupled. Let me think differently.
                
                # Actually, for fixed V and W, the equation is:
                # For each (a,b,c): T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
                # This is a linear system in U.
                # Let's vectorize: for each a, we have a linear system.
                
                # Simpler: use numpy to do the ALS
                # Fix V, W: solve for U
                # T.reshape(n*m, -1) = U @ (V*W).reshape(r, -1) ??? No.
                
                # Let me define: B_mat[b, d] = V[b, d], C_mat[c, d] = W[c, d]
                # Then T[a, b, c] = sum_d U[a, d] * B_mat[b, d] * C_mat[c, d]
                # For fixed a: T[a, :, :] = sum_d U[a, d] * outer(B_mat[:, d], C_mat[:, d])
                # Let E_d = outer(B_mat[:, d], C_mat[:, d])  # shape (m*p, p*n)
                # Then T[a].reshape(-1) = sum_d U[a, d] * E_d.reshape(-1)
                # So T_flat = U @ E_mat where E_mat[d] = E_d.reshape(-1)
                # E_mat has shape (r, m*p*p*n)
                # T_flat has shape (n*m, m*p*p*n)
                # U = T_flat @ E_mat.T @ pinv(E_mat @ E_mat.T)
                
                E_mat = np.zeros((target_rank, m*p * p*n))
                for d in range(target_rank):
                    E_mat[d] = np.outer(V[:, d], W[:, d]).reshape(-1)
                
                T_flat = target.reshape(n*m, -1)
                # Solve U @ E_mat = T_flat
                # U = T_flat @ E_mat.T @ inv(E_mat @ E_mat.T)
                Gram = E_mat @ E_mat.T
                try:
                    U_new = T_flat @ E_mat.T @ np.linalg.inv(Gram)
                except np.linalg.LinAlgError:
                    U_new = T_flat @ np.linalg.pinv(E_mat)
                
                # Fix U, W: solve for V
                # T[a, b, c] = sum_d U[a, d] * V[b, d] * W[c, d]
                # For fixed b: T[:, b, :] = sum_d V[b, d] * outer(U[:, d], W[:, d])
                # Let F_d = outer(U[:, d], W[:, d])  # shape (n*m, p*n)
                # T[:, b].reshape(-1) = sum_d V[b, d] * F_d.reshape(-1)
                # T_col[b] = sum_d V[b, d] * F_d_flat
                # T_col = V @ F_mat where F_mat[d] = F_d.reshape(-1)
                F_mat = np.zeros((target_rank, n*m * p*n))
                for d in range(target_rank):
                    F_mat[d] = np.outer(U_new[:, d], W[:, d]).reshape(-1)
                
                T_col = target.reshape(m*p, -1)  # T[b, a, c] = T[a, b, c]
                # Actually T is indexed [a, b, c], so T.reshape(m*p, -1) gives T[b, (a,c)]
                # But we want for each b: T[:, b, :] which is T.transpose(1,0,2).reshape(m*p, -1)
                T_b = target.transpose(1, 0, 2).reshape(m*p, -1)
                try:
                    V_new = T_b @ F_mat.T @ np.linalg.inv(F_mat @ F_mat.T)
                except np.linalg.LinAlgError:
                    V_new = T_b @ np.linalg.pinv(F_mat)
                
                # Fix U, V: solve for W
                # T[a, b, c] = sum_d U[a, d] * V[b, d] * W[c, d]
                # For fixed c: T[:, :, c] = sum_d W[c, d] * outer(U[:, d], V[:, d])
                # Let G_d = outer(U[:, d], V[:, d])  # shape (n*m, m*p)
                # T[:,:,c].reshape(-1) = sum_d W[c, d] * G_d.reshape(-1)
                # T_c[c] = sum_d W[c, d] * G_d_flat
                G_mat = np.zeros((target_rank, n*m * m*p))
                for d in range(target_rank):
                    G_mat[d] = np.outer(U_new[:, d], V_new[:, d]).reshape(-1)
                
                T_c = target.transpose(2, 0, 1).reshape(p*n, -1)
                try:
                    W_new = T_c @ G_mat.T @ np.linalg.inv(G_mat @ G_mat.T)
                except np.linalg.LinAlgError:
                    W_new = T_c @ np.linalg.pinv(G_mat)
                
                # Check convergence
                if np.allclose(U_new, U, atol=1e-8) and np.allclose(V_new, V, atol=1e-8) and np.allclose(W_new, W, atol=1e-8):
                    converged = True
                    break
                
                U, V, W = U_new, V_new, W_new
            
            # Check if we found a valid decomposition
            if time.time() > deadline:
                break
                
            # Reconstruct and check
            T_recon = np.zeros_like(target)
            for d in range(target_rank):
                T_recon += np.outer(np.outer(U[:, d], V[:, d]), W[:, d]).reshape(n*m, m*p, p*n)
            
            if np.allclose(T_recon, target, atol=1e-6):
                # Found a valid decomposition!
                # Round to nearest half-integer or integer if possible
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify the rounded version
                T_rounded = np.zeros_like(target)
                for d in range(target_rank):
                    T_rounded += np.outer(np.outer(U_rounded[:, d], V_rounded[:, d]), W_rounded[:, d]).reshape(n*m, m*p, p*n)
                
                if np.allclose(T_rounded, target, atol=1e-4):
                    # Convert to lists
                    U_list = [[float(x) for x in row] for row in U_rounded.tolist()]
                    V_list = [[float(x) for x in row] for row in V_rounded.tolist()]
                    W_list = [[float(x) for x in row] for row in W_rounded.tolist()]
                    
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END

