import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    # Allow 140s out of 150s
    deadline = start_time + 140.0
    
    # Target tensor T[a, b, c]
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # We represent T as a flat array of length (n*m)*(m*p)*(p*n) = 9*9*9 = 729
    # But iterating over 729 entries in Python loops is slow.
    # We can precompute the indices.
    
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_indices.append((a, b, c))
                
    nz_set = set(nz_indices)
    all_indices = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
    
    def compute_error_and_grad(U, V, W):
        """
        Computes error and gradients for the tensor decomposition.
        U: (nm, R), V: (mp, R), W: (pn, R)
        """
        R = U.shape[1]
        err = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Pre-compute outer products to speed up? 
        # val = sum(U[a] * V[b] * W[c])
        
        for a, b, c in all_indices:
            # Compute value
            val = np.dot(np.dot(U[a], V[b]), W[c]) # This is not right. U[a] is vector, V[b] is vector.
            # val = sum_k U[a,k] * V[b,k] * W[c,k]
            val = np.sum(U[a] * V[b] * W[c])
            
            if (a, b, c) in nz_set:
                target = 1.0
            else:
                target = 0.0
                
            diff = val - target
            err += diff * diff
            
            # Gradients
            # d/dU[a,k] (val-target)^2 = 2 * diff * V[b,k] * W[c,k]
            grad_U[a] += 2 * diff * V[b] * W[c]
            grad_V[b] += 2 * diff * U[a] * W[c]
            grad_W[c] += 2 * diff * U[a] * V[b]
            
        return err, (grad_U, grad_V, grad_W)

    def snap_and_verify(U, V, W, rank):
        # Snap to nearest half-integer
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Verify exactness
        for a, b, c in all_indices:
            val = np.sum(U_snap[a] * V_snap[b] * W_snap[c])
            if (a, b, c) in nz_set:
                if abs(val - 1.0) > 1e-9:
                    return None
            else:
                if abs(val) > 1e-9:
                    return None
        
        # Prune zero columns
        mask = np.any(U_snap != 0, axis=0) | np.any(V_snap != 0, axis=0) | np.any(W_snap != 0, axis=0)
        U_final = U_snap[:, mask]
        V_final = V_snap[:, mask]
        W_final = W_snap[:, mask]
        
        return {
            "U": U_final.tolist(),
            "V": V_final.tolist(),
            "W": W_final.tolist()
        }

    def get_fallback(rank):
        # Return schoolbook decomposition for given rank (only valid for rank=27)
        if rank != 27:
            return None
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for idx, (a, b, c) in enumerate(nz_indices):
            U[a, idx] = 1.0
            V[b, idx] = 1.0
            W[c, idx] = 1.0
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    best_result = get_fallback(27)
    
    # Try to find lower rank
    for rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
            
        # Multiple restarts
        num_restarts = 5 if rank == 23 else 3
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Optimize
            # We use a custom gradient descent with momentum to handle the non-convexity
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            # We run for a fixed number of steps or until time limit
            steps = 200
            for step in range(steps):
                if time.time() > deadline:
                    break
                    
                err, (gU, gV, gW) = compute_error_and_grad(U, V, W)
                
                # Add L1/L2 regularization towards half-integers?
                # Just use the gradient for now.
                
                vU = momentum * vU - lr * gU
                vV = momentum * vV - lr * gV
                vW = momentum * vW - lr * gW
                
                U += vU
                V += vV
                W += vW
                
                # Optional: Snap to grid periodically? No, it breaks gradients.
                
            # Check if we converged
            err, _ = compute_error_and_grad(U, V, W)
            if err < 1e-3:
                result = snap_and_verify(U, V, W, rank)
                if result:
                    return result

    return best_result
