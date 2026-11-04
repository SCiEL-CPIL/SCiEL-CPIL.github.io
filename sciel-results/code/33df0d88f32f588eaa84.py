import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Shapes:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Target tensor T[i, j, k] = 1 if j == k else 0
    # T is indexed by (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Fallback: Standard rank 18
    U18 = np.zeros((n*m, 18))
    V18 = np.zeros((m*p, 18))
    W18 = np.zeros((p*n, 18))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U18[i*m + j, col] = 1
                V18[j*p + k, col] = 1
                W18[k*n + i, col] = 1
                col += 1
    
    best_solution = {'U': U18.tolist(), 'V': V18.tolist(), 'W': W18.tolist()}
    best_rank = 18
    
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s buffer for 150s limit

    def check_solution_exact(U, V, W, R):
        # Snap to nearest 0.5
        Us = np.round(U * 2) / 2
        Vs = np.round(V * 2) / 2
        Ws = np.round(W * 2) / 2
        
        # Check exactness
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target = 1.0 if j == k else 0.0
                    val = 0.0
                    for r in range(R):
                        val += Us[i*m + j, r] * Vs[j*p + k, r] * Ws[k*n + i, r]
                    if abs(val - target) > 1e-9:
                        return False, None, None, None
        return True, Us, Vs, Ws

    def try_rank(R):
        nonlocal best_solution, best_rank
        if R >= best_rank:
            return
        
        # Objective function for continuous optimization
        def objective(params):
            U = params[0:n*m*R].reshape(n*m, R)
            V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W = params[n*m*R + m*p*R:].reshape(p*n, R)
            
            # Calculate MSE
            mse = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target = 1.0 if j == k else 0.0
                        val = 0.0
                        for r in range(R):
                            val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                        mse += (val - target)**2
                        
            # Regularization to snap to half-integers
            # This helps the optimizer stay near the solution manifold
            reg = 0.0
            # Small weight so it doesn't dominate, but guides
            # We want to encourage values to be close to 0.5 * integer
            # Penalty: (2*x - round(2*x))^2
            # This is non-differentiable at boundaries, but scipy can handle it or we use a smooth approx.
            # Smooth approx: 0.5 * (1 - cos(2*pi*(2*x)))? No, let's just use the raw penalty, 
            # gradient-free methods like Powell can handle it, or we use a small weight.
            
            # Let's try a simple L2 penalty on distance to nearest 0.5
            # But this is non-smooth.
            # Better: Just minimize MSE. If MSE is 0, snap and check.
            # If MSE is small, snap and check.
            
            return mse

        # We will use multiple restarts
        n_restarts = 100
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return
            
            # Initialize
            # Start from Rank 18 solution and truncate/randomize
            # This is a good heuristic: the rank 17 solution is "close" to rank 18
            U_init = np.zeros((n*m, R))
            V_init = np.zeros((m*p, R))
            W_init = np.zeros((p*n, R))
            
            # Copy first R-1 terms from rank 18
            for r in range(min(R-1, 18)):
                U_init[:, r] = U18[:, r]
                V_init[:, r] = V18[:, r]
                W_init[:, r] = W18[:, r]
            
            # The last term is random
            U_init[:, R-1] = np.random.randn(n*m) * 0.1
            V_init[:, R-1] = np.random.randn(m*p) * 0.1
            W_init[:, R-1] = np.random.randn(p*n) * 0.1
            
            # Sometimes, fully random start is better
            if restart % 5 == 0:
                U_init = np.random.randn(n*m, R) * 0.5
                V_init = np.random.randn(m*p, R) * 0.5
                W_init = np.random.randn(p*n, R) * 0.5

            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimize
            # Use Powell method which is derivative-free and robust to non-smoothness
            # If we add regularization, it might be non-smooth.
            # Let's stick to MSE first.
            res = minimize(objective, params_init, method='Powell', options={'maxiter': 500, 'disp': False})
            
            if res.success or res.fun < 1e-8:
                U_opt = res.x[0:n*m*R].reshape(n*m, R)
                V_opt = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_opt = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                is_sol, Us, Vs, Ws = check_solution_exact(U_opt, V_opt, W_opt, R)
                if is_sol:
                    best_solution = {'U': Us.tolist(), 'V': Vs.tolist(), 'W': Ws.tolist()}
                    best_rank = R
                    return
            
            # Also try L-BFGS-B with gradients? 
            # Powell is slow but robust.
            # Let's try a few L-BFGS-B runs as well if time permits.
            # But Powell is sufficient for small dimensions.

    # Try rank 17
    try_rank(17)
    
    # If 17 failed, we are stuck with 18
    return best_solution
