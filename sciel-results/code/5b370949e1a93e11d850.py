"""Rewritten program: rank-minimal exact tensor decomposition for 3x4 x 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 140  # 140s deadline with 10s margin
    
    # Start with schoolbook rank 72
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        result = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            result += np.einsum('i,j,k->ijk', U_arr[:,r], V_arr[:,r], W_arr[:,r])
        
        if not np.allclose(result, T, atol=1e-9):
            return False
        
        # Check integer/half-integer constraint
        # The problem statement says "integer/half-integer entries".
        # We will check if 2*val is close to an integer.
        for arr in [U_arr, V_arr, W_arr]:
            doubled = arr * 2.0
            if not np.allclose(doubled, np.round(doubled), atol=1e-9):
                return False
        
        return True

    def optimize_rank(R, seed):
        """Try to find a decomposition of rank R using gradient descent."""
        np.random.seed(seed)
        
        # Flatten parameters
        # U: n*m x R, V: m*p x R, W: p*n x R
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W
        
        # Initialize with small random values
        x0 = np.random.randn(total_size) * 0.1
        
        def objective(x):
            U_flat = x[:size_U]
            V_flat = x[size_U:size_U+size_V]
            W_flat = x[size_U+size_V:]
            
            U = U_flat.reshape((n*m, R))
            V = V_flat.reshape((m*p, R))
            W = W_flat.reshape((p*n, R))
            
            # Compute tensor contraction
            # result[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Using einsum for speed
            res = np.einsum('ar,br,cr->abc', U, V, W)
            
            # Error is sum of squares of (res - T)
            err = res - T
            return np.sum(err * err)
            
        # We don't strictly need gradient for scipy.optimize if we use a derivative-free method,
        # but BFGS is faster if we provide gradient. 
        # However, computing the gradient of a rank-R tensor contraction is straightforward.
        # grad_U[a,r] = 2 * sum_{b,c} (res[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
        # Let's implement gradient for better convergence.
        
        def objective_and_gradient(x):
            U_flat = x[:size_U]
            V_flat = x[size_U:size_U+size_V]
            W_flat = x[size_U+size_V:]
            
            U = U_flat.reshape((n*m, R))
            V = V_flat.reshape((m*p, R))
            W = W_flat.reshape((p*n, R))
            
            res = np.einsum('ar,br,cr->abc', U, V, W)
            diff = res - T
            
            obj = np.sum(diff * diff)
            
            # Gradients
            # d/dU[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
            # d/dV[b,r] = 2 * sum_{a,c} diff[a,b,c] * U[a,r] * W[c,r]
            # d/dW[c,r] = 2 * sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
            
            grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
            grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            return obj, grad

        try:
            res_opt = minimize(objective_and_gradient, x0, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res_opt.success or res_opt.fun < 1e-6:
                x_opt = res_opt.x
                U_opt = x_opt[:size_U].reshape((n*m, R))
                V_opt = x_opt[size_U:size_U+size_V].reshape((m*p, R))
                W_opt = x_opt[size_U+size_V:].reshape((p*n, R))
                
                # Round to nearest 0.5
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                # Verify exactness
                if verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), R):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                    
        except Exception:
            pass
            
        return None

    # Try ranks from 60 to 71
    # We use multiple seeds for each rank to handle local minima
    for target_rank in range(60, best_rank):
        if time.time() > deadline:
            break
            
        found = False
        # Try a few random initializations
        for seed in range(10):
            if time.time() > deadline:
                break
                
            res = optimize_rank(target_rank, seed)
            if res is not None:
                U_r, V_r, W_r = res
                found = True
                best_rank = target_rank
                best_U = U_r
                best_V = V_r
                best_W = W_r
                break
                
        if found:
            continue
            
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
