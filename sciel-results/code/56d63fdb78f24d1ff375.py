import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    
    n, m, p = 2, 3, 6
    
    # Build the target tensor T[i, j, k] = 1 for all (i,j,k) in [0,n) x [0,m) x [0,p)
    T = np.ones((n, m, p))
    
    # We want to find U, V, W such that:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # Let's reshape for clarity:
    # U_3d[i, j, r] has shape (n, m, rank)
    # V_3d[j, k, r] has shape (m, p, rank) 
    # W_3d[k, i, r] has shape (p, n, rank)
    # 
    # The constraint is: sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r] = 1
    
    # Try a structured search using known matrix multiplication algorithms
    # First, let's try to find a low-rank decomposition using a combination of
    # known algorithms and optimization.
    
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # But structural constraints typically require higher rank.
    
    # Let's use a more systematic approach: try to find decompositions using
    # bilinear forms and exploit the specific structure.
    
    # For 2x3 * 3x6, we can think of this as computing 12 output entries.
    # Each output C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Strategy: Use a combination of:
    # 1. Direct enumeration of small candidate ranks
    # 2. ALS (Alternating Least Squares) with multiple restarts
    # 3. Rounding to integer/half-integer values
    
    best_result = None
    best_rank = n * m * p  # 36 (schoolbook)
    
    # First, construct the schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W decompose the tensor exactly."""
        rank = len(U[0])
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Reshape
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        # Compute T_calc
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        return np.allclose(T, T_calc, atol=tol)
    
    def try_als(rank, max_iters=300, tol=1e-10, seed=0):
        """Try ALS to find a decomposition of given rank."""
        if time.time() - start_time > 140:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n, m, rank) * 0.5
        V = rng.randn(m, p, rank) * 0.5
        W = rng.randn(p, n, rank) * 0.5
        
        prev_err = np.inf
        
        for it in range(max_iters):
            if time.time() - start_time > 145:
                break
            
            # Update U: for each (i,j), solve for U[i,j,:]
            for i in range(n):
                for j in range(m):
                    # M_u[k, r] = V[j,k,r] * W[k,i,r]
                    M_u = V[j, :, :] * W[:, i, :]  # (p, rank)
                    A = M_u.T @ M_u
                    b = M_u.T @ T[i, j, :]
                    A += 1e-8 * np.eye(rank)
                    try:
                        U[i, j, :] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        U[i, j, :] = np.linalg.lstsq(M_u, T[i, j, :], rcond=None)[0]
            
            # Update V: for each (j,k), solve for V[j,k,:]
            for j in range(m):
                for k in range(p):
                    # M_v[i, r] = U[i,j,r] * W[k,i,r]
                    M_v = U[:, j, :] * W[k, :, :]  # (n, rank)
                    A = M_v.T @ M_v
                    b = M_v.T @ T[:, j, k]
                    A += 1e-8 * np.eye(rank)
                    try:
                        V[j, k, :] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        V[j, k, :] = np.linalg.lstsq(M_v, T[:, j, k], rcond=None)[0]
            
            # Update W: for each (k,i), solve for W[k,i,:]
            for k in range(p):
                for i in range(n):
                    # M_w[j, r] = U[i,j,r] * V[j,k,r]
                    M_w = U[i, :, :] * V[:, k, :]  # (m, rank)
                    A = M_w.T @ M_w
                    b = M_w.T @ T[i, :, k]
                    A += 1e-8 * np.eye(rank)
                    try:
                        W[k, i, :] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        W[k, i, :] = np.linalg.lstsq(M_w, T[i, :, k], rcond=None)[0]
            
            # Compute error
            T_calc = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            err = np.linalg.norm(T - T_calc)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        # Check if we have an exact solution
        T_calc = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        if not np.allclose(T, T_calc, atol=1e-8):
            return None
        
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        T_calc_r = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        if not np.allclose(T, T_calc_r, atol=1e-9):
            # Try rounding to integers
            U_r = np.round(U)
            V_r = np.round(V)
            W_r = np.round(W)
            T_calc_r = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            if not np.allclose(T, T_calc_r, atol=1e-9):
                return None
        
        U_flat = U_r.reshape(-1, rank)
        V_flat = V_r.reshape(-1, rank)
        W_flat = W_r.reshape(-1, rank)
        
        # Final verification
        if verify(U_flat.tolist(), V_flat.tolist(), W_flat.tolist()):
            return {"U": U_flat.tolist(), "V": V_flat.tolist(), "W": W_flat.tolist()}
        
        return None
    
    # Try ranks from lower bound upward
    # Start from a reasonable lower bound
    lower_bound = max(n*m, m*p, p*n)  # 18
    
    for rank in range(lower_bound, best_rank + 1):
        if time.time() - start_time > 140:
            break
        
        # Try multiple seeds
        max_seeds = 15 if rank < 25 else 8
        
        for seed in range(max_seeds):
            if time.time() - start_time > 145:
                break
            
            try:
                result = try_als(rank, max_iters=200, seed=seed)
                if result is not None:
                    if rank < best_rank:
                        best_rank = rank
                        best_result = result
                    # If we found a decomposition at the lower bound, we're done
                    if rank == lower_bound:
                        return result
            except Exception:
                pass
        
        # If we found a good result, try a few more ranks to see if we can do better
        if best_result is not None and best_rank <= lower_bound + 3:
            # Continue searching for even lower ranks isn't possible (we're at lower bound)
            # But let's try a few more seeds at this rank
            for seed in range(20):
                if time.time() - start_time > 145:
                    break
                try:
                    result = try_als(rank, max_iters=200, seed=seed + 100)
                    if result is not None and rank < best_rank:
                        best_rank = rank
                        best_result = result
                except Exception:
                    pass
    
    # Fallback to schoolbook
    if best_result is None:
        best_result = schoolbook()
    
    return best_result

# Quick self-test
if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print(f"Found decomposition with rank {rank}")
    
    n, m, p = 2, 3, 6
    T = np.ones((n, m, p))
    
    U_arr = np.array(result['U'])
    V_arr = np.array(result['V'])
    W_arr = np.array(result['W'])
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    T_calc = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    print(f"Max error: {np.max(np.abs(T - T_calc))}")
