"""Rewritten program for matrix multiplication 2x3 by 3x4 using numerical search for rank minimization."""
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Target tensor T
    # T[i, j, k, l] = 1 if i == l, else 0
    T = np.zeros((n, m, p, n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k, i] = 1.0

    # Try to find a decomposition with rank 12
    rank = 12
    
    # Use multiple random restarts to find a solution
    for restart in range(20):
        try:
            result = _solve_rank(n, m, p, T, rank, restart)
            if result is not None:
                if _verify_decomposition(result, n, m, p, T):
                    return result
        except Exception:
            continue
            
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

def _solve_rank(n, m, p, T, rank, seed):
    import numpy as np
    import scipy.optimize as opt
    
    np.random.seed(seed)
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(x):
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # T_comp[i, j, k, l] = sum_c U[i,j,c] V[j,k,c] W[k,l,c]
        T_comp = np.einsum('ijc,jkc,klc->ijkl', U3, V3, W3)
        
        diff = T_comp - T
        return np.sum(diff**2)
        
    res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000})
    
    if res.fun < 1e-6:
        U = res.x[:n*m*rank].reshape(n*m, rank)
        V = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        return {
            "U": U_rounded.tolist(),
            "V": V_rounded.tolist(),
            "W": W_rounded.tolist()
        }
        
    return None

def _verify_decomposition(result, n, m, p, T):
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    rank = U.shape[1]
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    T_comp = np.einsum('ijc,jkc,klc->ijkl', U3, V3, W3)
    
    if not np.allclose(T_comp, T, atol=1e-9):
        return False
        
    return True

# EVOLVE-BLOCK-END
