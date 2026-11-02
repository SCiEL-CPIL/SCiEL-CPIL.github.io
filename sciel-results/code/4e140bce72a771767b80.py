import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Target tensor T represented as a list of (i, j, k) indices
    # We want sum_k U[:, k] @ V[:, k] @ W[:, k] == T
    # U is (n*m, R), V is (m*p, R), W is (p*n, R)
    # The product is sum_k (U[:, k] outer V[:, k]) outer W[:, k] ? No.
    # The tensor is sum_k u_k v_k^T w_k^T ? 
    # The multiplication tensor is T = sum_{i,j,k} e_i (A) e_j (B) e_k (C)
    # Where A is n x m, B is m x p, C is p x n (flattened)
    # U maps A to intermediate, V maps B to intermediate, W maps intermediate to C?
    # Standard form: C_{ik} = sum_j A_{ij} B_{jk}
    # Tensor: sum_{i,j,k} C_{ik} A_{ij} B_{jk}
    # Decomposition: sum_{r=1}^R u_r A v_r B w_r^T C ?
    # The interface says:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # And the decomposition is exact.
    # This implies: sum_{r} U[:, r] (as A) * V[:, r] (as B) * W[:, r] (as C) ?
    # No, the tensor is linear in A, B, and C.
    # T = sum_r (U[:, r] \otimes V[:, r] \otimes W[:, r])
    # Where U[:, r] is a vector of length n*m (flattened A)
    # V[:, r] is a vector of length m*p (flattened B)
    # W[:, r] is a vector of length p*n (flattened C)
    
    # The schoolbook decomposition:
    # For each i, j, k:
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    
    # We will search for a decomposition with rank R < 105.
    
    # Time limit guard
    start_time = time.time()
    timeout = 110  # Leave 10s margin
    
    # Function to verify and snap
    def verify_and_snap(U, V, W, tol=1e-8):
        # Check if entries are close to half-integers
        # And if the decomposition is exact
        n, m, p = DIMENSIONS
        # Reconstruct tensor
        # T_idx[i, j, k] should be 1 if i,j,k are valid indices in the schoolbook
        # Actually, we just need to check if sum_r U[:,r] V[:,r] W[:,r] matches the target tensor
        
        # Target tensor is sparse: 1 at (i*m+j, j*p+k, k*n+i) for all i,j,k
        # We can check the residual of the decomposition against the target.
        
        # Efficient check:
        # Residual = sum_r U[:,r] outer V[:,r] outer W[:,r] - Target
        # This is expensive.
        # Instead, we can check a few random linear combinations?
        # Or just compute the full tensor if R is small?
        # R ~ 100, n*m=15, m*p=35, p*n=21.
        # Tensor size 15*35*21 = 11025.
        # R * 11025 operations. 100 * 11025 = 1.1M. Very fast.
        
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Compute decomposition
        # D = sum_r np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        D = np.zeros_like(T)
        for r in range(U.shape[1]):
            D += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        residual = np.linalg.norm(D - T)
        if residual > tol:
            return False
        
        # Check half-integers
        for mat in [U, V, W]:
            # Multiply by 2 and check if close to integers
            scaled = 2.0 * mat
            # Round to nearest integer
            rounded = np.round(scaled)
            if np.linalg.norm(scaled - rounded) > 1e-6:
                return False
        
        # Snap to half-integers
        U_snap = np.round(2.0 * U) / 2.0
        V_snap = np.round(2.0 * V) / 2.0
        W_snap = np.round(2.0 * W) / 2.0
        
        return U_snap, V_snap, W_snap

    def solve_for_rank(R, num_restarts=10):
        n, m, p = DIMENSIONS
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        # Target tensor
        T = np.zeros((size_u, size_v, size_w))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        for restart in range(num_restarts):
            # Random initialization
            # Scale matters. Schoolbook entries are 1.
            U = np.random.randn(size_u, R) * 0.5
            V = np.random.randn(size_v, R) * 0.5
            W = np.random.randn(size_w, R) * 0.5
            
            # Objective function
            def objective(params):
                U_flat = params[:size_u*R].reshape(size_u, R)
                V_flat = params[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
                W_flat = params[size_u*R + size_v*R :].reshape(size_w, R)
                
                D = np.zeros_like(T)
                for r in range(R):
                    D += np.einsum('a,b,c->abc', U_flat[:,r], V_flat[:,r], W_flat[:,r])
                
                return np.linalg.norm(D - T)
            
            # Use scipy.optimize
            from scipy.optimize import least_squares
            initial_params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(objective, initial_params, method='lm', max_nfev=1000)
                if result.success or result.cost < 1e-8:
                    U_sol = result.x[:size_u*R].reshape(size_u, R)
                    V_sol = result.x[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
                    W_sol = result.x[size_u*R + size_v*R :].reshape(size_w, R)
                    
                    verified = verify_and_snap(U_sol, V_sol, W_sol)
                    if verified:
                        return verified
            except Exception:
                pass
                
            if time.time() - start_time > timeout:
                return None
        
        return None

    # Search for lower ranks
    # Start from 104 down to 90
    # We will try a few restarts for each rank
    for R in range(104, 89, -1):
        if time.time() - start_time > timeout:
            break
        result = solve_for_rank(R, num_restarts=5)
        if result:
            U_snap, V_snap, W_snap = result
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
    # Fallback to schoolbook
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
