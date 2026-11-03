"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18
    
    # Build the tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1 for i in [0,n), j in [0,m), k in [0,p)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Schoolbook decomposition: rank = n*m*p = 54
    def generate_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_school, V_school, W_school, schoolbook_rank = generate_schoolbook()
    best_U, best_V, best_W, best_rank = U_school.copy(), V_school.copy(), W_school.copy(), schoolbook_rank
    
    # Precompute useful matrices for ALS
    # For ALS on U: fix V,W. T[i,:,:] = sum_s U[i,s] * (V[:,s] ⊗ W[:,s])
    # So T_flat[i, :] = U[i, :] @ (V ⊗ W) where V⊗W is (mp*pn, rank)
    
    def check_decomposition(U, V, W, tol=1e-8):
        T_approx = np.einsum('is,js,ks->ijk', U, V, W)
        return np.max(np.abs(T_approx - T)) < tol
    
    def als_optimize(rank, max_iter=500, tol=1e-10, seed=42):
        """ALS optimization with multiple restarts"""
        if time.time() > deadline - 15:
            return None
        
        best_err = float('inf')
        best_result = None
        
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(seed + restart)
            # Initialize with small random values
            scale = 0.5
            U = np.random.randn(nm, rank) * scale
            V = np.random.randn(mp, rank) * scale
            W = np.random.randn(pn, rank) * scale
            
            # Precompute for ALS
            # For U update: T[i, :, :] = sum_s U[i,s] * outer(V[:,s], W[:,s])
            # Flatten: T_flat[i, j*pn+k] = sum_s U[i,s] * V[j,s] * W[k,s]
            # So T_flat = U @ (V ⊙ W) where (V ⊙ W)[j*pn+k, s] = V[j,s]*W[k,s]
            
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    break
                
                # Update U: for each i, solve U[i,:] from T[i,:,:]
                # T[i, j, k] = sum_s U[i,s] * V[j,s] * W[k,s]
                # Let M_U[j*pn+k, s] = V[j,s] * W[k,s]
                # Then T_flat[i, :] = U[i, :] @ M_U
                # U[i, :] = T_flat[i, :] @ pinv(M_U)
                
                M_U = np.einsum('js,ks->jks', V, W).reshape(mp * pn, rank)
                # U = T_flat @ pinv(M_U)
                try:
                    U_new = np.linalg.lstsq(M_U.T, T.reshape(nm, mp*pn).T, rcond=None)[0].T
                except:
                    U_new = U
                
                # Update V: for each j, solve V[j,:] from T[:, j, :]
                # T[i, j, k] = sum_s U[i,s] * V[j,s] * W[k,s]
                # M_V[i*pn+k, s] = U[i,s] * W[k,s]
                M_V = np.einsum('is,ks->iks', U, W).reshape(nm * pn, rank)
                try:
                    V_new = np.linalg.lstsq(M_V.T, T.reshape(nm, mp, pn).transpose(0, 2, 1).reshape(mp, nm*pn).T, rcond=None)[0].T
                except:
                    V_new = V
                
                # Update W: for each k, solve W[k,:] from T[:, :, k]
                # T[i, j, k] = sum_s U[i,s] * V[j,s] * W[k,s]
                # M_W[i*mp+j, s] = U[i,s] * V[j,s]
                M_W = np.einsum('is,js->ijs', U, V).reshape(pn, nm*mp)  # Wait, this is wrong
                # Actually: for fixed k, T[:, :, k] (shape nm x mp) = sum_s (U[:,s] ⊗ V[:,s]) * W[k,s]
                # So T[:,:,k].flatten() = W[k,:] @ (U ⊗ V) where (U⊗V)[i*mp+j, s] = U[i,s]*V[j,s]
                M_W_full = np.einsum('is,js->ijs', U, V).reshape(nm*mp, rank)
                try:
                    W_new = np.linalg.lstsq(M_W_full.T, T.reshape(nm, mp, pn).transpose(2, 0, 1).reshape(pn, nm*mp).T, rcond=None)[0].T
                except:
                    W_new = W
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                if it % 50 == 0:
                    T_approx = np.einsum('is,js,ks->ijk', U, V, W)
                    err = np.max(np.abs(T_approx - T))
                    if err < tol:
                        break
            
            # Final check
            T_approx = np.einsum('is,js,ks->ijk', U, V, W)
            err = np.max(np.abs(T_approx - T))
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
        
        if best_result is not None and best_err < 1e-6:
            return best_result[0], best_result[1], best_result[2], best_err
        return None
    
    # Try ranks from 30 to 53 (below schoolbook)
    for rank in range(30, schoolbook_rank):
        if time.time() > deadline - 30:
            break
        
        result = als_optimize(rank)
        if result is not None:
            U_new, V_new, W_new, err = result
            if err < 1e-6:
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, rank
                break
    
    # Clean up: round to nearest representable value, zero out tiny entries
    def clean_matrix(M):
        M_clean = M.copy()
        # Zero out very small values
        M_clean[np.abs(M_clean) < 1e-8] = 0.0
        # Try to round to simple fractions: 0, ±0.5, ±1, ±1.5, ±2
        for i in range(M_clean.shape[0]):
            for j in range(M_clean.shape[1]):
                val = M_clean[i, j]
                if val == 0:
                    continue
                # Check if close to a simple fraction
                for denom in [1, 2]:
                    for num in range(-10, 11):
                        target = num / denom
                        if abs(val - target) < 1e-6:
                            M_clean[i, j] = target
                            break
                    else:
                        continue
                    break
        return M_clean
    
    best_U = clean_matrix(best_U)
    best_V = clean_matrix(best_V)
    best_W = clean_matrix(best_W)
    
    # Final verification
    T_approx = np.einsum('is,js,ks->ijk', best_U, best_V, best_W)
    final_err = np.max(np.abs(T_approx - T))
    
    if final_err > 1e-4:
        # Fall back to schoolbook
        best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
        best_rank = schoolbook_rank
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
