import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    rank_sb = n * m * p  # 54
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition as fallback
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose T."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=tol)
    
    def round_to_half_int(A, tol=1e-6):
        """Round entries to nearest half-integer if close."""
        A_rounded = np.round(2 * A) / 2.0
        if np.allclose(A, A_rounded, atol=tol):
            return A_rounded
        return A
    
    # Try to find a lower rank decomposition
    # Start from rank 50 and go down
    deadline = time.time() + 120  # 120 second deadline
    best_result = (U_sb.tolist(), V_sb.tolist(), W_sb.tolist(), rank_sb)
    
    for rank in range(rank_sb - 4, rank_sb - 1):  # Try 50, 51, 52, 53
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try multiple random restarts
        n_restarts = max(1, int((deadline - time.time()) / 5))
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # Use gradient descent to minimize ||T_recon - T||^2
            # T_recon = sum_r U[:,r] * V[:,r] * W[:,r]
            # Gradient with respect to U: for each r, grad_U = V[:,r] * W[:,r] * (T_recon[:, :, :] - T) contracted
            # This is complex. Let's use a simpler approach.
            
            # Alternative: use ALS (Alternating Least Squares)
            # Fix V and W, solve for U, then fix U and W, solve for V, etc.
            
            for iteration in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Solve for U: for each r, U[:,r] = (V[:,r] * W[:,r])^T * T_vec / ||V[:,r] * W[:,r]||^2
                # where T_vec is T reshaped
                T_flat = T.reshape(-1)
                
                for r in range(rank):
                    # The r-th term contributes U[:,r] * V[:,r] * W[:,r]
                    # We want to minimize ||sum_s U[:,s] * V[:,s] * W[:,s] - T||^2
                    # For fixed r, the optimal U[:,r] given other terms is:
                    # residual = T - sum_{s!=r} U[:,s] * V[:,s] * W[:,s]
                    # U[:,r] = (V[:,r] * W[:,r])^T * residual / ||V[:,r] * W[:,r]||^2
                    
                    # Compute current reconstruction
                    T_current = np.einsum('ar,br,cr->abc', U, V, W)
                    residual = T - T_current
                    
                    # The contribution of term r is U[:,r] * V[:,r] * W[:,r]
                    # We want to find U[:,r] that best fits residual
                    # This is a least squares problem:
                    # minimize ||U[:,r] * V[:,r] * W[:,r] - residual||^2
                    # Let w = V[:,r] * W[:,r] (element-wise product, shape (nm, mp, pn))
                    # Then we want U[:,r] such that sum_a U[a,r] * w[a,b,c] ≈ residual[b,c] for all b,c
                    # This is: for each (b,c), sum_a U[a,r] * w[a,b,c] = residual[b,c]
                    # Let W_bc = w[:,b,c] (shape (nm,)), then U[:,r] * W_bc = residual[b,c]
                    # This is a system of equations for U[:,r]
                    
                    # Actually, let's think of it differently.
                    # The tensor T can be written as sum_r U_r * V_r * W_r where U_r is a vector
                    # The gradient descent approach is simpler to implement correctly.
                    
                    pass
                
                # Let's use a simpler gradient descent
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                error = T_recon - T
                err_norm = np.linalg.norm(error)
                
                if err_norm < 1e-10:
                    break
                
                # Gradient with respect to U
                # d/dU[a,r] sum_{b,c} (T_recon[a,b,c] - T[a,b,c])^2
                # = 2 * sum_{b,c} (T_recon[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
                grad_U = np.zeros_like(U)
                for a in range(nm):
                    for r in range(rank):
                        grad_U[a, r] = 2 * np.sum(error[a, :, :] * V[:, r] * W[:, r])
                
                # Gradient with respect to V
                grad_V = np.zeros_like(V)
                for b in range(mp):
                    for r in range(rank):
                        grad_V[b, r] = 2 * np.sum(error[:, b, :] * U[:, r] * W[:, r])
                
                # Gradient with respect to W
                grad_W = np.zeros_like(W)
                for c in range(pn):
                    for r in range(rank):
                        grad_W[c, r] = 2 * np.sum(error[:, :, c] * U[:, r] * V[:, r])
                
                # Simple gradient descent with adaptive step size
                lr = 0.01
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            # Check if we found a solution
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_recon, T, atol=1e-8):
                # Try to round to half-integers
                U_r = round_to_half_int(U)
                V_r = round_to_half_int(V)
                W_r = round_to_half_int(W)
                if verify(U_r, V_r, W_r):
                    found = True
                    best_result = (U_r.tolist(), V_r.tolist(), W_r.tolist(), rank)
                    break
                elif verify(U, V, W):
                    # Keep as float if half-integer rounding doesn't work
                    found = True
                    best_result = (U.tolist(), V.tolist(), W.tolist(), rank)
                    break
        
        if found:
            break
    
    # Return the best result found
    U_best, V_best, W_best, rank_best = best_result
    return {"U": U_best, "V": V_best, "W": W_best}
