import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 20
    # Try to find a decomposition with this rank
    # If it fails, fall back to higher ranks or schoolbook
    
    # Initialize with random values
    # We will try a few restarts
    
    best_result = None
    best_loss = float('inf')
    
    start_time = time.time()
    deadline = start_time + 100 # 100s budget
    
    # Define the loss function
    def loss_func(params):
        U = params[0:n*m*target_rank].reshape(n, m, target_rank)
        V = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m, p, target_rank)
        W = params[n*m*target_rank + m*p*target_rank : ].reshape(p, n, target_rank)
        
        # Compute S[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        S = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        T = np.ones((n, m, p))
        return np.sum((S - T)**2)
    
    # Gradient is expensive to compute analytically, let's use numerical or scipy's automatic
    # scipy.optimize.minimize can compute gradients numerically or we can provide them.
    # For speed, let's provide gradients if possible, or use BFGS with numerical gradients.
    
    # Actually, let's just use a simple gradient descent or scipy's L-BFGS-B.
    # Given the small number of variables (760), L-BFGS-B should be fine.
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Random initialization
        scale = 1.0
        params0 = np.random.randn(n*m*target_rank + m*p*target_rank + p*n*target_rank) * scale
        
        # Optimize
        try:
            res = minimize(loss_func, params0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            if res.fun < best_loss:
                best_loss = res.fun
                best_result = res.x
        except Exception as e:
            pass
            
    if best_result is not None and best_loss < 1e-6:
        # We found a solution. Now try to snap to half-integers.
        U = best_result[0:n*m*target_rank].reshape(n, m, target_rank)
        V = best_result[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m, p, target_rank)
        W = best_result[n*m*target_rank + m*p*target_rank : ].reshape(p, n, target_rank)
        
        # Snap to nearest 0.5
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Verify
        S_snap = np.einsum('ijc,jkc,kic->ijk', U_snap, V_snap, W_snap)
        T = np.ones((n, m, p))
        if np.allclose(S_snap, T, atol=1e-6):
            # Convert to lists
            U_list = U_snap.reshape(n*m, target_rank).tolist()
            V_list = V_snap.reshape(m*p, target_rank).tolist()
            W_list = W_snap.reshape(p*n, target_rank).tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # If we didn't find a rank 20 solution, try rank 22 (known upper bound)
    # Or just return schoolbook if time is tight?
    # Let's try to construct the rank 22 solution analytically if possible, or just use schoolbook.
    # Given the time, returning schoolbook is safe but score 1.0.
    # We should try to at least return a better bound if found.
    
    # Fallback: Schoolbook
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
