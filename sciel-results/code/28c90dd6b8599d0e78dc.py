# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 3, 3, 7
    
    # Try to find a rank 4 decomposition for the 3x3x1 sub-problem
    # If successful, total rank is 7 * 4 = 28
    # Otherwise, fallback to schoolbook rank 63
    
    def solve_subproblem(rank_target):
        # Tensors for 3x3x1
        # Indices: a (0..8), b (0..2), c (0..2)
        # T[a,b,c] = 1 if a = 3*c + b else 0
        
        # We want U (9xR), V (3xR), W (3xR)
        # such that sum_t U[a,t] V[b,t] W[c,t] == T[a,b,c]
        
        # Initialize randomly
        np.random.seed(42)
        
        best_loss = float('inf')
        best_factors = None
        
        for attempt in range(20):
            U = np.random.randn(9, rank_target)
            V = np.random.randn(3, rank_target)
            W = np.random.randn(3, rank_target)
            
            # Normalize to help optimization
            # Not strictly necessary if using proper scaling
            
            def loss(params):
                U_flat = params[:9*rank_target].reshape(9, rank_target)
                V_flat = params[9*rank_target:9*rank_target+3*rank_target].reshape(3, rank_target)
                W_flat = params[9*rank_target+3*rank_target:].reshape(3, rank_target)
                
                # Compute tensor reconstruction
                # T_recon[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
                # Efficiently:
                # Temp[a,b] = sum_t U[a,t] V[b,t]
                # T_recon[a,b,c] = sum_t Temp[a,b,t] W[c,t]? No.
                # T_recon[a,b,c] = sum_t (U[a,t] V[b,t]) W[c,t]
                
                # Use einsum
                # U: 9xR, V: 3xR, W: 3xR
                # UV: 9x3xR (U[a,t] V[b,t])
                UV = np.einsum('at,bt->abt', U_flat, V_flat)
                # T_recon: 9x3x3 (sum_t UV[a,b,t] W[c,t])
                T_recon = np.einsum('abt,ct->abc', UV, W_flat)
                
                # Target T
                # T[a,b,c] = 1 if a == 3*c + b
                T_target = np.zeros((9, 3, 3))
                for c in range(3):
                    for b in range(3):
                        a = 3*c + b
                        T_target[a, b, c] = 1.0
                
                # Loss: sum of squared errors
                err = T_recon - T_target
                return np.sum(err**2)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use Nelder-Mead or L-BFGS-B
            # L-BFGS-B is better for smooth functions
            res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_factors = res.x
        
        if best_loss < 1e-8:
            U = best_factors[:9*rank_target].reshape(9, rank_target)
            V = best_factors[9*rank_target:9*rank_target+3*rank_target].reshape(3, rank_target)
            W = best_factors[9*rank_target+3*rank_target:].reshape(3, rank_target)
            return U, V, W
        return None

    # Try rank 4
    sub_factors = solve_subproblem(4)
    
    if sub_factors is not None:
        U_sub, V_sub, W_sub = sub_factors
        rank_total = 7 * 4
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]
        
        # Map sub-problem factors to full problem
        # For each k in 0..6:
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i
        
        for k in range(7):
            for t in range(4):
                # U part
                for i in range(3):
                    for j in range(3):
                        a_idx = i * m + j
                        # U_sub[a_idx, t]
                        U[a_idx][k * 4 + t] = U_sub[a_idx, t]
                        
                # V part
                for j in range(3):
                    b_idx = j * p + k
                    V[b_idx][k * 4 + t] = V_sub[j, t]
                    
                # W part
                for i in range(3):
                    c_idx = k * n + i
                    W[c_idx][k * 4 + t] = W_sub[i, t]
                    
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
