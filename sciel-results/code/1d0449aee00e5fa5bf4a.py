"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x5 matrix by a 5x5 matrix.
    
    The schoolbook rank is n*m*p = 50.
    We attempt to find a lower rank decomposition using numerical optimization
    if time permits, falling back to the schoolbook decomposition otherwise.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    schoolbook_rank = n * m * p  # 50
    
    # Build the target tensor T
    # T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook fallback
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = make_schoolbook()
    best_rank = n * m * p

    def try_rank(r, max_trials=3):
        """Try to find a rank-r decomposition using gradient descent."""
        if time.time() > deadline - 15:
            return None
            
        # Define objective and gradient for this rank r
        def objective(x):
            U = x[:n*m*r].reshape(n*m, r)
            V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W = x[n*m*r + m*p*r :].reshape(p*n, r)
            Rec = np.einsum('ik,jk,lk->ijl', U, V, W)
            Res = T - Rec
            return np.sum(Res * Res)

        def gradient(x):
            U = x[:n*m*r].reshape(n*m, r)
            V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W = x[n*m*r + m*p*r :].reshape(p*n, r)
            Rec = np.einsum('ik,jk,lk->ijl', U, V, W)
            Res = T - Rec
            Grad_U = np.einsum('ijl,jc,lc->ic', Res, V, W)
            Grad_V = np.einsum('ijl,ic,lc->jc', Res, U, W)
            Grad_W = np.einsum('ijl,ic,jc->lc', Res, U, V)
            return np.concatenate([Grad_U.flatten(), Grad_V.flatten(), Grad_W.flatten()])

        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(42 + r * 100 + trial)
            U0 = np.random.randn(n * m, r) * 0.5
            V0 = np.random.randn(m * p, r) * 0.5
            W0 = np.random.randn(p * n, r) * 0.5
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                              options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8})
                
                if res.success or (res.fun < 1e-6):
                    U = res.x[:n*m*r].reshape(n*m, r)
                    V = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W = res.x[n*m*r + m*p*r :].reshape(p*n, r)
                    
                    # Verify the decomposition
                    Rec = np.einsum('ik,jk,lk->ijl', U, V, W)
                    if np.max(np.abs(Rec - T)) < 1e-8:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            except Exception:
                pass
        
        return None

    # Try ranks from 49 down to 40
    for target_rank in range(schoolbook_rank - 1, max(35, schoolbook_rank - 15), -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(target_rank)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break  # Found a valid lower rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
