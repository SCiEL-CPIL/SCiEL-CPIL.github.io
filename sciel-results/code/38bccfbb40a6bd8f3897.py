import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build the tensor T[a,b,c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i with j=j' and i=i'
    # Actually: T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find rank-r decomposition
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # such that T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Start with schoolbook
    U = np.zeros((n*m, schoolbook_rank))
    V = np.zeros((m*p, schoolbook_rank))
    W = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    best_result = (U, V, W)
    
    # Try to reduce rank
    for target_rank in range(schoolbook_rank - 1, 30, -1):
        if time.time() > deadline:
            break
        
        # Try to find a rank-target_rank decomposition
        # Use random initialization and optimization
        np.random.seed(42)
        
        U_init = np.random.randn(n*m, target_rank) * 0.1
        V_init = np.random.randn(m*p, target_rank) * 0.1
        W_init = np.random.randn(p*n, target_rank) * 0.1
        
        # Define the objective: sum of squares of (T - U*V*W)
        def objective(params):
            U_flat, V_flat, W_flat = params
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            # Compute U*V*W tensor
            # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # This is expensive, so use a trick
            # For each r, outer product of U[:,r], V[:,r], W[:,r]
            # But that's still expensive. Let's use einsum
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T - T_approx
            return np.sum(diff**2)
        
        # This is too slow for large tensors. Let's use a different approach.
        # Instead, fix W and solve for U,V, or use alternating least squares.
        
        # Alternating minimization
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            # Update U: for each a, U[a,:] should minimize sum_{b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
            # This is a linear system in U[a,:]
            # Let M[a] be a matrix where M[a][b*c + r, r'] = ... this is getting complex.
            
            # Simpler: use gradient descent with small steps
            lr = 0.01
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_approx - T
            
            # Gradient w.r.t. U: grad_U[a,r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
            grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
            grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if np.sum(diff**2) < 1e-10:
                # Found a valid decomposition
                best_result = (U, V, W)
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
        if time.time() > deadline:
            break
    
    U, V, W = best_result
    
    # Convert to list of lists with integer/half-integer entries if possible
    # Round to nearest 0.5
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Check if rounded version is still valid
    T_approx = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
    if np.max(np.abs(T_approx - T)) < 1e-9:
        U, V, W = U_rounded, V_rounded, W_rounded
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
