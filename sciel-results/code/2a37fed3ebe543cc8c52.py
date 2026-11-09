import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Start with schoolbook as fallback
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    start_time = time.time()
    time_limit = 140  # Leave margin before 150s kill
    
    # Try ranks from 74 down to some minimum
    # But first, let's try a smarter approach: alternating optimization
    # For each rank, try to find a decomposition using alternating minimization
    
    def try_rank(target_rank, seed_offset=0):
        if time.time() - start_time > time_limit:
            return None
            
        np.random.seed(42 + seed_offset)
        
        # Initialize with random values
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Target tensor: T[i,j,k] = 1 for all i,j,k
        target = np.ones((n, m, p))
        
        def reconstruct(u, v, w):
            u_3d = u.reshape(n, m, target_rank)
            v_3d = v.reshape(m, p, target_rank)
            w_3d = w.reshape(p, n, target_rank)
            return np.einsum('ijr,jkr,kir->ijk', u_3d, v_3d, w_3d)
        
        def objective_from_matrices(u, v, w):
            recon = reconstruct(u, v, w)
            diff = recon - target
            return np.sum(diff * diff)
        
        # Alternating optimization: optimize one matrix at a time
        # When optimizing U (fixing V, W), the problem is linear in U
        # For each (i,j), we need: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
        # This is a linear system for U[i*m+j, :]
        
        max_iters = 50
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                return None
            
            # Optimize U given V, W
            # For each (i,j): find u_vec such that for all k: sum_r u_vec[r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Let A_u[i,j,k,r] = V[j*p+k, r] * W[k*n+i, r]
            # Then we need: sum_r u_vec[r] * A_u[i,j,k,r] = 1 for all k
            # This is: A_u[i,j] @ u_vec = ones(p)
            # Where A_u[i,j] is a p x target_rank matrix with A_u[i,j,k,r] = V[j*p+k,r] * W[k*n+i,r]
            
            v_3d = V.reshape(m, p, target_rank)
            w_3d = W.reshape(p, n, target_rank)
            
            for i in range(n):
                for j in range(m):
                    # Build A: p x target_rank
                    # A[k, r] = v_3d[j, k, r] * w_3d[k, i, r]
                    A = v_3d[j, :, :] * w_3d[:, i, :]  # (p, target_rank)
                    # Solve A @ u_vec = ones(p)
                    # Use least squares
                    try:
                        u_vec, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j, :] = u_vec
                    except:
                        pass
            
            # Optimize V given U, W
            u_3d = U.reshape(n, m, target_rank)
            w_3d = W.reshape(p, n, target_rank)
            
            for j in range(m):
                for k in range(p):
                    # For each (j,k): find v_vec such that for all i: sum_r v_vec[r] * U[i*m+j, r] * W[k*n+i, r] = 1
                    # A[i, r] = u_3d[i, j, r] * w_3d[k, i, r]
                    A = u_3d[:, j, :] * w_3d[k, :, :]  # (n, target_rank)
                    try:
                        v_vec, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j * p + k, :] = v_vec
                    except:
                        pass
            
            # Optimize W given U, V
            u_3d = U.reshape(n, m, target_rank)
            v_3d = V.reshape(m, p, target_rank)
            
            for k in range(p):
                for i in range(n):
                    # For each (k,i): find w_vec such that for all j: sum_r w_vec[r] * U[i*m+j, r] * V[j*p+k, r] = 1
                    # A[j, r] = u_3d[i, j, r] * v_3d[j, k, r]
                    A = u_3d[i, :, :] * v_3d[:, k, :]  # (m, target_rank)
                    try:
                        w_vec, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k * n + i, :] = w_vec
                    except:
                        pass
            
            # Check convergence
            recon = reconstruct(U, V, W)
            error = np.max(np.abs(recon - target))
            if error < 1e-8:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify
                recon_snap = reconstruct(U_snap, V_snap, W_snap)
                if np.allclose(recon_snap, target, atol=1e-9):
                    return {"U": U_snap, "V": V_snap, "W": W_snap}
        
        # If not converged, return None
        return None
    
    # Try rank 74 first with multiple random restarts
    best_result = None
    for restart in range(10):
        if time.time() - start_time > time_limit:
            break
        result = try_rank(74, seed_offset=restart)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return {
            "U": best_result["U"].tolist(),
            "V": best_result["V"].tolist(),
            "W": best_result["W"].tolist()
        }
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
