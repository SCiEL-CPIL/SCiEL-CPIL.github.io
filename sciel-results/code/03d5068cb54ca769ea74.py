import numpy as np
import time
from fractions import Fraction
import random

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Target tensor T[a, b, c]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Fallback: Schoolbook decomposition
    def schoolbook():
        rank = nm * mp * pn // p  # Wait, rank is n*m*p = 3*3*7 = 63
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Verification function
    def verify(U, V, W, tol=1e-8):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # T_comp[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        T_comp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        if np.allclose(T_comp, T, atol=tol):
            return True
        
        return False

    # Strategy: Randomized optimization for low rank decompositions
    # We iterate ranks from low to high.
    
    # Lower bound:
    # Rank >= max(nm, mp, pn) is not a bound.
    # For 3x3x7, the trivial bound is 63.
    # Known lower bound for 3x3x3 is 23.
    # For 3x3x7, let's guess a lower bound around 40-50.
    # Let's try ranks from 40 to 62.
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    # Ranks to try. We start from a reasonable lower bound.
    # If we find a solution, we can stop immediately since we go low to high.
    # However, finding a solution at rank R doesn't prove it's minimal if we skipped lower ranks.
    # So we must check all lower ranks first.
    
    ranks_to_try = list(range(40, 63))
    
    # To save time, we might want to prioritize certain ranks or use fewer trials for higher ranks.
    # But since we want the MINIMAL rank, we must try 40, 41, 42... in order.
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Number of trials:
        # Lower ranks are harder to find, so maybe more trials?
        # Or maybe lower ranks are easier if the structure allows?
        # Let's use a fixed number of trials, say 3, and maybe increase if time permits.
        # Given 120s limit, and ~20 ranks, we have ~6s per rank.
        # L-BFGS-B with gradient is fast.
        # Let's try 3 trials per rank.
        
        num_trials = 3
        
        found = False
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + R * 100 + trial)
            
            # Initialize U, V, W
            # Scaling: The tensor entries are 1.
            # If U, V, W are O(1), the sum of R terms is O(sqrt(R)) or O(R) depending on signs.
            # To get 1, we might need scaling 1/sqrt(R) or similar.
            # Let's try standard normal, which is O(1).
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # Objective: Minimize ||T - T_comp||_F^2
            # T_comp = sum_r U[:,r] V[:,r]^T W[:,r]^T (outer product)
            # Actually T_comp is a 3-tensor.
            
            def objective_and_grad(params):
                size_U = nm * R
                size_V = mp * R
                U_flat = params[0:size_U]
                V_flat = params[size_U:size_U+size_V]
                W_flat = params[size_U+size_V:]
                
                U_mat = U_flat.reshape(nm, R)
                V_mat = V_flat.reshape(mp, R)
                W_mat = W_flat.reshape(pn, R)
                
                T_comp = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
                diff = T - T_comp
                loss = np.sum(diff * diff)
                
                # Gradients
                # dL/dU[a,r] = -2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
                # dL/dV[b,r] = -2 * sum_{a,c} diff[a,b,c] * U[a,r] * W[c,r]
                # dL/dW[c,r] = -2 * sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
                
                grad_U = -2 * np.einsum('abc,br,cr->ar', diff, V_mat, W_mat)
                grad_V = -2 * np.einsum('abc,ar,cr->br', diff, U_mat, W_mat)
                grad_W = -2 * np.einsum('abc,ar,br->cr', diff, U_mat, V_mat)
                
                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                
                return loss, grad
            
            from scipy.optimize import minimize
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                # Use L-BFGS-B
                # maxiter: 50 is usually enough for these small problems if close.
                # If not close, it will fail.
                res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-6:
                    U_opt = res.x[0:nm*R].reshape(nm, R)
                    V_opt = res.x[nm*R:nm*R+mp*R].reshape(mp, R)
                    W_opt = res.x[nm*R+mp*R:].reshape(pn, R)
                    
                    # Verify
                    if verify(U_opt, V_opt, W_opt, tol=1e-5):
                        best_rank = R
                        best_U = U_opt
                        best_V = V_opt
                        best_W = W_opt
                        found = True
                        break # Break trial loop, found solution for this rank
            
            except Exception:
                pass
        
        if found:
            # Since we are iterating R in increasing order, the first found is the minimal rank (within the range we tried).
            # We can break the outer loop.
            break

    # If no better solution found, return schoolbook
    if best_U is None:
        result = schoolbook()
        return result
    else:
        # Convert to lists
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
