import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the target tensor
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
    
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the target tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, target, atol=1e-9)
    
    def snap_to_half_int(arr):
        """Snap array to nearest half-integer."""
        return np.round(arr * 2.0) / 2.0
    
    def get_schoolbook():
        """Return schoolbook decomposition (rank = n*m*p = 50)."""
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Try to find a decomposition with rank < 50
    best_result = None
    
    # Strategy 1: Try ranks from 49 down to 40 with structured random search
    for target_rank in range(49, 35, -1):
        if time.time() > deadline:
            break
        
        found = False
        n_restarts = 50
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(1000 + restart)
            
            # Initialize with small random half-integers
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_u, target_rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_v, target_rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_w, target_rank))
            
            # Alternating minimization with half-integer projection
            for iteration in range(30):
                if time.time() > deadline:
                    break
                
                # Update U: for each column r, U[:,r] = argmin_U sum_{a,b,c} (target[a,b,c] - sum_r' U[a,r']V[b,r']W[c,r'])^2 * U[a,r]^2
                # This is a quadratic in U[:,r]
                # Fix V and W, solve for U
                # T_target = sum_r U[:,r] outer(V[:,r], W[:,r])
                # For fixed r, the contribution is U[:,r] * (V[:,r] outer W[:,r])
                # We want to minimize ||target - sum_r' U[:,r'] (V[:,r'] outer W[:,r'])||^2
                # Partial derivative w.r.t. U[:,r]:
                # dL/dU[:,r] = -2 * sum_{a,b,c} diff[a,b,c] * V[b,r] * W[c,r] * (1 if we're updating U[:,r])
                # Actually, let's use a different approach: solve the linear system
                
                # Compute residual
                T_current = np.einsum('ar,br,cr->abc', U, V, W)
                diff = target - T_current
                
                # Update U: for each r, we want to find u_r that minimizes
                # ||diff + u_r * (V[:,r] outer W[:,r])||^2
                # This gives: u_r = <diff, V[:,r] outer W[:,r]> / ||V[:,r] outer W[:,r]||^2
                # But this is for a single term. For the full sum, we need to solve a linear system.
                
                # Better approach: use Gauss-Seidel style updates
                # For each r, hold others fixed, solve for U[:,r]
                for r in range(target_rank):
                    # Compute the contribution of all other terms
                    if target_rank > 1:
                        T_others = np.einsum('ar,br,cr->abc', 
                                             np.delete(U, r, axis=1), 
                                             np.delete(V, r, axis=1), 
                                             np.delete(W, r, axis=1))
                    else:
                        T_others = np.zeros((dim_u, dim_v, dim_w))
                    
                    # We need to find u_r such that:
                    # T_others + u_r * (V[:,r] outer W[:,r]) ≈ target
                    # Let A = V[:,r] outer W[:,r]
                    # We want: <u_r, A> = <target - T_others, A> (in the tensor inner product)
                    # u_r * ||A||^2 = <target - T_others, A>
                    # But u_r is a vector, and A is a tensor. This doesn't work directly.
                    
                    # Alternative: use gradient descent step
                    # grad_U[:,r] = -2 * einsum('abc,br,cr->ar', diff, V, W)
                    grad_u = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
                    grad_v = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
                    grad_w = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
                    
                    # Take a step
                    lr = 0.1
                    U -= lr * grad_u
                    V -= lr * grad_v
                    W -= lr * grad_w
                    
                    # Snap to half-integers
                    U = snap_to_half_int(U)
                    V = snap_to_half_int(V)
                    W = snap_to_half_int(W)
                
                # Check convergence
                T_check = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(T_check, target, atol=1e-9):
                    found = True
                    best_result = (U.copy(), V.copy(), W.copy(), target_rank)
                    break
            
            if found:
                break
        
        if found:
            break
    
    # Strategy 2: If not found, try a different initialization based on schoolbook
    if best_result is None and time.time() < deadline:
        U_sb, V_sb, W_sb, rank_sb = get_schoolbook()
        
        # Try to merge pairs of schoolbook terms
        # If two terms have the same W vector, we can merge their U and V
        for target_rank in range(49, 40, -1):
            if time.time() > deadline:
                break
            
            found = False
            for restart in range(20):
                if time.time() > deadline:
                    break
                
                np.random.seed(2000 + restart)
                
                # Start from schoolbook and randomly perturb
                U = U_sb[:, :target_rank].copy() + np.random.randn(dim_u, target_rank) * 0.1
                V = V_sb[:, :target_rank].copy() + np.random.randn(dim_v, target_rank) * 0.1
                W = W_sb[:, :target_rank].copy() + np.random.randn(dim_w, target_rank) * 0.1
                
                # Snap to half-integers
                U = snap_to_half_int(U)
                V = snap_to_half_int(V)
                W = snap_to_half_int(W)
                
                # Optimize
                for iteration in range(20):
                    if time.time() > deadline:
                        break
                    
                    T_current = np.einsum('ar,br,cr->abc', U, V, W)
                    diff = target - T_current
                    
                    grad_u = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
                    grad_v = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
                    grad_w = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
                    
                    lr = 0.05
                    U -= lr * grad_u
                    V -= lr * grad_v
                    W -= lr * grad_w
                    
                    U = snap_to_half_int(U)
                    V = snap_to_half_int(V)
                    W = snap_to_half_int(W)
                
                T_check = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(T_check, target, atol=1e-9):
                    found = True
                    best_result = (U.copy(), V.copy(), W.copy(), target_rank)
                    break
            
            if found:
                break
    
    # Return the best result found, or schoolbook as fallback
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W, rank = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
