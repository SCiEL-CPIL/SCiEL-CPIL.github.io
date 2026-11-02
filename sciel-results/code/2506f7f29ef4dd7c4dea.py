import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    # T[i, j, j2, k, i2, k2] = 1 if j==j2 and i==i2 and k==k2 else 0
    # Shape: (n, m, m, p, n, p)
    T = np.zeros((n, m, m, p, n, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, i, k] = 1.0
                
    T_flat = T.flatten()
    
    def residual_fn(params, R):
        """Compute residual of tensor decomposition."""
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        # Reshape to (R, n, m), (R, m, p), (R, p, n)
        U_3d = U_flat.reshape((R, n, m))
        V_3d = V_flat.reshape((R, m, p))
        W_3d = W_flat.reshape((R, p, n))
        
        # Reconstruct: T_recon[i,j,j2,k,i2,k2] = sum_r U[r,i,j]*V[r,j2,k]*W[r,k2,i2]
        # Using einsum: 'rij,rab,rcd->ijabdc'
        T_recon = np.einsum('rij,rab,rcd->ijabdc', U_3d, V_3d, W_3d)
        
        return (T_recon.flatten() - T_flat)
    
    def verify_solution(U, V, W, R):
        """Verify that the decomposition is exact."""
        U_3d = U.reshape((R, n, m))
        V_3d = V.reshape((R, m, p))
        W_3d = W.reshape((R, p, n))
        T_recon = np.einsum('rij,rab,rcd->ijabdc', U_3d, V_3d, W_3d)
        return np.max(np.abs(T_recon.flatten() - T_flat)) < 1e-9
    
    def snap_to_half_integers(arr):
        """Snap values to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    # Schoolbook fallback
    def get_schoolbook_solution():
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
    
    best_solution = None
    best_rank = n * m * p
    
    # Try ranks from 70 to 83 (below schoolbook)
    np.random.seed(42)
    
    for R in range(70, 84):
        if time.time() > deadline:
            break
            
        # Number of restarts: more for lower ranks (harder)
        # Start with fewer restarts for higher ranks (closer to schoolbook, easier)
        n_restarts = 20 if R > 80 else 50
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Initialize with small random values
            # Scale 0.5 is good for integer-like solutions
            scale = 0.5
            x0 = np.random.randn(n*m*R + m*p*R + p*n*R) * scale
            
            try:
                res = least_squares(
                    residual_fn, x0, args=(R,), 
                    method='lm', max_nfev=1000,
                    ftol=1e-12, xtol=1e-12, gtol=1e-12
                )
            except Exception:
                continue
            
            # Check if the solution is close to exact
            if res.cost < 1e-8:
                # Extract and snap
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R
                
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                U_snap = snap_to_half_integers(U_flat)
                V_snap = snap_to_half_integers(V_flat)
                W_snap = snap_to_half_integers(W_flat)
                
                # Verify snapped solution
                if verify_solution(U_snap, V_snap, W_snap, R):
                    best_rank = R
                    best_solution = {
                        "U": U_snap.reshape((n*m, R)).tolist(),
                        "V": V_snap.reshape((m*p, R)).tolist(),
                        "W": W_snap.reshape((p*n, R)).tolist()
                    }
                    break
        
        if best_solution:
            break
    
    if best_solution:
        return best_solution
    else:
        # Fallback to schoolbook
        U, V, W = get_schoolbook_solution()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
