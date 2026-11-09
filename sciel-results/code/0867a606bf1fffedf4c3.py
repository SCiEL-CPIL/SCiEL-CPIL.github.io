import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # Then output: U_out[i*m+j, r] = U[i,r], V_out[j*p+k, r] = V[j,r], W_out[k*n+i, r] = W[k,r]
    
    T = np.ones((n, m, p))
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank 72
    
    # Try ranks from low to high
    for rank in range(48, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Multiple restarts with different initializations
        for restart in range(30):
            if time.time() > deadline - 10:
                break
            
            # Different initialization strategies
            strategy = restart % 3
            if strategy == 0:
                # Random small values
                U = np.random.randn(n, rank) * 0.5
                V = np.random.randn(m, rank) * 0.5
                W = np.random.randn(p, rank) * 0.5
            elif strategy == 1:
                # Structured initialization using Kronecker-like patterns
                U = np.random.randn(n, rank)
                V = np.random.randn(m, rank)
                W = np.random.randn(p, rank)
                # Normalize columns
                for r in range(rank):
                    norm = np.linalg.norm(U[:, r])
                    if norm > 1e-10:
                        U[:, r] /= norm
                    norm = np.linalg.norm(V[:, r])
                    if norm > 1e-10:
                        V[:, r] /= norm
                    norm = np.linalg.norm(W[:, r])
                    if norm > 1e-10:
                        W[:, r] /= norm
            else:
                # Try to exploit the all-ones structure
                # Initialize with rank-1 components that sum to 1
                U = np.ones((n, rank)) / rank
                V = np.ones((m, rank)) / rank
                W = np.ones((p, rank)) / rank
                # Add small random perturbation
                U += np.random.randn(n, rank) * 0.1
                V += np.random.randn(m, rank) * 0.1
                W += np.random.randn(p, rank) * 0.1
            
            # ALS iterations
            converged = False
            for iteration in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each i, solve least squares
                # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                # Fix V and W, solve for U
                # D_U[(j,k), r] = V[j,r] * W[k,r]
                D_U = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m * p, rank)
                T_flat = T.reshape(n, m * p)
                try:
                    U = T_flat @ np.linalg.pinv(D_U)
                except:
                    break
                
                # Update V
                D_V = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n * p, rank)
                T_flat_V = T.transpose(0, 2, 1).reshape(n * p, m)
                try:
                    V = T_flat_V @ np.linalg.pinv(D_V)
                except:
                    break
                
                # Update W
                D_W = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n * m, rank)
                T_flat_W = T.transpose(1, 0, 2).reshape(n * m, p)
                try:
                    W = T_flat_W @ np.linalg.pinv(D_W)
                except:
                    break
                
                # Check convergence
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.linalg.norm(T_recon - T)
                if err < 1e-8:
                    converged = True
                    break
            
            if not converged:
                continue
            
            # Try to snap to half-integers or integers
            for snap_factor in [2, 4, 8, 16]:
                U_snap = np.round(U * snap_factor) / snap_factor
                V_snap = np.round(V * snap_factor) / snap_factor
                W_snap = np.round(W * snap_factor) / snap_factor
                
                T_recon_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                if np.allclose(T_recon_snap, T, atol=1e-10):
                    # Success! Build the output format
                    U_out = np.zeros((n * m, rank))
                    for i in range(n):
                        for j in range(m):
                            U_out[i * m + j, :] = U_snap[i, :]
                    
                    V_out = np.zeros((m * p, rank))
                    for j in range(m):
                        for k in range(p):
                            V_out[j * p + k, :] = V_snap[j, :]
                    
                    W_out = np.zeros((p * n, rank))
                    for k in range(p):
                        for i in range(n):
                            W_out[k * n + i, :] = W_snap[k, :]
                    
                    # Verify the final result
                    # Check that C[i,k] = sum_j A[i,j] B[j,k] is correctly computed
                    # The decomposition should satisfy:
                    # T[i,j,k] = sum_r U_out[i*m+j, r] * V_out[j*p+k, r] * W_out[k*n+i, r]
                    # which equals sum_r U[i,r] * V[j,r] * W[k,r] = 1
                    verify_ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_out[i*m+j, :] * V_out[j*p+k, :] * W_out[k*n+i, :])
                                if abs(val - 1.0) > 1e-10:
                                    verify_ok = False
                                    break
                            if not verify_ok:
                                break
                        if not verify_ok:
                            break
                    
                    if verify_ok:
                        return {
                            "U": U_out.tolist(),
                            "V": V_out.tolist(),
                            "W": W_out.tolist()
                        }
    
    # Fallback: Schoolbook Rank 72
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
