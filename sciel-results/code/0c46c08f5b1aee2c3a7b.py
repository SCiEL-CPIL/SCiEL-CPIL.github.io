"""Matrix multiplication tensor decomposition for 3x4x4."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _verify_exact(U, V, W, tol=1e-10):
    """Verify U @ diag(W) @ V^T == T"""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    U2 = U.reshape(n * m, -1)
    V2 = V.reshape(m * p, -1)
    # Compute U * W * V^T in tensor form
    prod = U2 @ (W.reshape(-1, 1) * V2.T)  # (n*m, m*p)
    prod_3d = prod.reshape(n, m, m, p)
    # We need sum over j of prod_3d[i,j,j,k] = T[i,j,k]
    # Actually, the matrix product (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = A[i,j] * B[j,k] = 1
    # So we need U @ diag(W) @ V^T to have entries (i*m+j) x (j*p+k) that give the right structure
    # Let's just check the matrix product directly
    # Actually, let's just return the max difference
    
    # Standard check: the bilinear form
    # For each (i,j,k), we need sum_r U[i*m+j, r] * W[r] * V[j*p+k, r] = 1
    # Wait, that's not right either. Let me think again.
    # T[i,j,k] = A[i,j] * B[j,k]
    # So the tensor T is just a 3-tensor where T[i,j,k] = 1 for all valid (i,j,k)
    # The decomposition is: T = sum_r U_r * W_r * V_r^T (tensor product)
    # Where (U_r * W_r * V_r^T)[i,j,k] = U_r[i,j] * W_r * V_r[j,k]... no
    # Actually: T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r
    # where U_r is (n,m), V_r is (m,p), W_r is scalar
    # So: sum_r U_r[i,j] * W_r * V_r[j,k] = T[i,j,k] = 1
    
    U_flat = U2.reshape(n, m, -1)  # (n, m, rank)
    V_flat = V2.reshape(m, p, -1)  # (m, p, rank)
    W_vec = W  # (rank,)
    
    # T[i,j,k] = sum_r U_flat[i,j,r] * W_vec[r] * V_flat[j,k,r]
    # = (U_flat @ diag(W_vec) @ V_flat^T)[i,j,k]
    # Let's compute this efficiently
    UW = U_flat * W_vec[np.newaxis, np.newaxis, :]  # (n, m, rank)
    result = np.einsum('ijr,jkr->ijk', UW, V_flat)
    return np.max(np.abs(result - 1.0))

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s limit, 10s margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    # Schoolbook
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros(rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[r] = 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try various starting ranks and run optimization
    # For each starting rank r0 from rank down to some lower bound
    for r0 in range(rank - 1, max(rank - 6, 0), -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple initializations for each rank
        for init in range(3):
            if time.time() > deadline - 30:
                break
            
            # Random sparse initialization
            np.random.seed(42 + r0 * 100 + init)
            U = np.zeros((n * m, r0))
            V = np.zeros((m * p, r0))
            W = np.ones(r0)
            
            # Initialize sparsely
            for r in range(r0):
                # Pick random positions
                idx = np.random.randint(0, n * m)
                jdk = np.random.randint(0, m * p)
                U[idx, r] = 1
                V[jdk, r] = 1
                W[r] = 1.0
            
            # Flatten to vector
            x = np.concatenate([U.flatten(), V.flatten(), W])
            orig_shape_U = (n * m, r0)
            orig_shape_V = (m * p, r0)
            
            def loss_func(x):
                Uf = x[:n*m*r0].reshape(orig_shape_U)
                Vf = x[n*m*r0:n*m*r0 + m*p*r0].reshape(orig_shape_V)
                Wf = x[n*m*r0 + m*p*r0:]
                
                U3 = Uf.reshape(n, m, -1)
                V3 = Vf.reshape(m, p, -1)
                UW = U3 * Wf[np.newaxis, np.newaxis, :]
                result = np.einsum('ijr,jkr->ijk', UW, V3)
                return np.sum((result - 1.0) ** 2)
            
            initial_loss = loss_func(x)
            if initial_loss > 100000:
                continue
            
            # Optimize
            try:
                res = minimize(loss_func, x, method='L-BFGS-B', 
                               options={'maxiter': 50, 'ftol': 1e-12})
                final_loss = res.fun
                
                # Convert back
                Uf = res.x[:n*m*r0].reshape(orig_shape_U)
                Vf = res.x[n*m*r0:n*m*r0 + m*p*r0].reshape(orig_shape_V)
                Wf = res.x[n*m*r0 + m*p*r0:]
                
                # Try to snap to nice values
                # Round small values to 0
                Uf[np.abs(Uf) < 1e-8] = 0
                Vf[np.abs(Vf) < 1e-8] = 0
                Wf[np.abs(Wf) < 1e-8] = 0
                
                # Try to snap to half-integers
                Uf_snapped = np.round(Uf * 2) / 2
                Vf_snapped = np.round(Vf * 2) / 2
                Wf_snapped = np.round(Wf * 2) / 2
                
                # Check exactness of snapped version
                err_snapped = _verify_exact(Uf_snapped, Vf_snapped, Wf_snapped)
                if err_snapped < 1e-9:
                    if r0 < best_rank:
                        best_rank = r0
                        best_U, best_V, best_W = Uf_snapped, Vf_snapped, Wf_snapped
                    break
                else:
                    # Try unsnapped
                    err_unsnapped = _verify_exact(Uf, Vf, Wf)
                    if err_unsnapped < 1e-9:
                        if r0 < best_rank:
                            best_rank = r0
                            best_U, best_V, best_W = Uf, Vf, Wf
                        break
            except Exception:
                continue
    
    # Also try a more systematic approach: start from schoolbook and try to zero out entries
    if time.time() < deadline - 30:
        # Start from a dense rank-40 initialization
        for r0 in [rank - 2, rank - 3, rank - 4]:
            if time.time() > deadline - 30:
                break
            
            for init in range(5):
                if time.time() > deadline - 30:
                    break
                    
                np.random.seed(1234 + r0 * 10 + init)
                U = np.zeros((n * m, r0))
                V = np.zeros((m * p, r0))
                W = np.ones(r0)
                
                # Initialize from schoolbook pattern (first r0 terms)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r = i * m * p + j * p + k
                            if r < r0:
                                U[i * m + j, r] = 1
                                V[j * p + k, r] = 1
                                W[r] = 1.0
                
                # Add some random perturbation
                U += np.random.randn(n * m, r0) * 0.1
                V += np.random.randn(m * p, r0) * 0.1
                
                x = np.concatenate([U.flatten(), V.flatten(), W])
                orig_shape_U = (n * m, r0)
                orig_shape_V = (m * p, r0)
                
                def loss_func(x):
                    Uf = x[:n*m*r0].reshape(orig_shape_U)
                    Vf = x[n*m*r0:n*m*r0 + m*p*r0].reshape(orig_shape_V)
                    Wf = x[n*m*r0 + m*p*r0:]
                    U3 = Uf.reshape(n, m, -1)
                    V3 = Vf.reshape(m, p, -1)
                    UW = U3 * Wf[np.newaxis, np.newaxis, :]
                    result = np.einsum('ijr,jkr->ijk', UW, V3)
                    return np.sum((result - 1.0) ** 2)
                
                initial_loss = loss_func(x)
                if initial_loss > 1e6:
                    continue
                    
                try:
                    res = minimize(loss_func, x, method='L-BFGS-B',
                                   options={'maxiter': 100, 'ftol': 1e-14})
                    final_loss = res.fun
                    
                    Uf = res.x[:n*m*r0].reshape(orig_shape_U)
                    Vf = res.x[n*m*r0:n*m*r0 + m*p*r0].reshape(orig_shape_V)
                    Wf = res.x[n*m*r0 + m*p*r0:]
                    
                    # Try snapping to various grids
                    for snap_denom in [1, 2, 4, 10]:
                        Uf_snapped = np.round(Uf * snap_denom) / snap_denom
                        Vf_snapped = np.round(Vf * snap_denom) / snap_denom
                        Wf_snapped = np.round(Wf * snap_denom) / snap_denom
                        
                        err = _verify_exact(Uf_snapped, Vf_snapped, Wf_snapped)
                        if err < 1e-9:
                            if r0 < best_rank:
                                best_rank = r0
                                best_U, best_V, best_W = Uf_snapped, Vf_snapped, Wf_snapped
                            break
                except Exception:
                    continue
    
    # Convert to lists
    result_U = [[float(best_U[i, j]) for j in range(best_rank)] for i in range(n * m)]
    result_V = [[float(best_V[i, j]) for j in range(best_rank)] for i in range(m * p)]
    result_W = [float(best_W[i]) for i in range(best_rank)]
    
    return {"U": result_U, "V": result_V, "W": result_W}
# EVOLVE-BLOCK-END
