"""Find rank-minimal exact tensor decomposition for 3x4x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Build the target tensor T with shape (n*m, m*p, p*n)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for valid i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = rank_sb
    
    def verify(U, V, W, tol=1e-8):
        """Check if decomposition is exact."""
        r = U.shape[1]
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_approx - T)) < tol
    
    def try_rank(rank, max_trials=50, seed=0):
        """Try to find a rank-r decomposition using optimization."""
        nonlocal best_U, best_V, best_W, best_rank
        
        if rank >= best_rank:
            return False
        
        np.random.seed(seed)
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return False
            
            # Random initialization
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Flatten parameters
            def flatten(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten(x):
                U = x[:n*m*rank].reshape(n*m, rank)
                V = x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W = x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                return U, V, W
            
            def objective(x):
                U, V, W = unflatten(x)
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                return np.sum((T_approx - T)**2)
            
            x0 = flatten(U, V, W)
            
            # Use L-BFGS-B for optimization
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                 options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if result.fun < 1e-6:
                    U_opt, V_opt, W_opt = unflatten(result.x)
                    
                    # Round to nearest half-integer
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    if verify(U_rounded, V_rounded, W_rounded, tol=1e-6):
                        best_U, best_V, best_W = U_rounded.copy(), V_rounded.copy(), W_rounded.copy()
                        best_rank = rank
                        return True
            except Exception:
                pass
        
        return False
    
    # Try ranks from lower bound upward
    lower_bound = max(n*p, m*p, n*m)  # = 20
    for rank in range(lower_bound, min(best_rank, lower_bound + 15)):
        if time.time() > deadline:
            break
        if try_rank(rank, max_trials=30, seed=rank*100):
            break
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
