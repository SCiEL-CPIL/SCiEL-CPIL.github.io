# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    # Symmetrize matrices to ensure the trace formulation is valid and symmetric
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Initial solution: Rank matching
    heavy = np.argsort(-a_sym.sum(axis=1), kind="stable")
    central = np.argsort(b_sym.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Convert permutation to matrix X
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p] = 1.0
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Projected Gradient Descent on the Birkhoff Polytope
    # Objective: tr(A X B X^T)
    # Gradient: B X^T A + A X B
    
    alpha = 0.1 # Initial step size
    # We might need to adjust alpha. Let's try a fixed schedule or adaptive.
    # For safety, we'll use a small step size and many iterations.
    
    current_cost = np.sum(a * b[np.ix_(p, p)])
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute for cost calculation
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
        
    # ADMM / PGD Loop
    # We will run PGD until time is up, keeping the best permutation found.
    # Since X is projected to a permutation matrix at each step, we can evaluate cost directly.
    
    X_current = X.copy()
    
    # To speed up projection, we use linear_sum_assignment on -X (to maximize X)
    # Note: linear_sum_assignment minimizes cost, so we pass -X.
    
    for _ in range(10000): # Max iterations, will be cut by time
        if time.time() > deadline:
            break
            
        # Compute Gradient
        # G = b_sym @ X_current.T @ a_sym + a_sym @ X_current @ b_sym
        # Note: X_current is a permutation matrix (0/1) at the start of each iter due to projection.
        # But during the update step, X - alpha*G is not 0/1.
        # We project after the update.
        
        # Gradient calculation
        # Term 1: b_sym @ X_current.T @ a_sym
        # Term 2: a_sym @ X_current @ b_sym
        G = b_sym @ X_current.T @ a_sym + a_sym @ X_current @ b_sym
        
        # Update
        X_new = X_current - alpha * G
        
        # Project onto doubly stochastic matrices
        # Use Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-X_new)
        X_proj = np.zeros((n, n), dtype=np.float64)
        X_proj[row_ind, col_ind] = 1.0
        
        # Check if we improved
        p_new = col_ind.astype(np.int64) # Facility i is at location p_new[i]? 
        # linear_sum_assignment returns indices (row, col) such that cost[row, col] is minimized.
        # We minimized -X_new, so we maximize X_new.
        # row_ind is the facility index, col_ind is the location index.
        # So facility row_ind[k] is assigned to location col_ind[k].
        # We need p such that p[facility] = location.
        # So p[row_ind] = col_ind.
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        cost_new = compute_cost(p_new)
        
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new.copy()
            
        X_current = X_proj
        
        # Optional: Reduce step size slowly?
        # alpha *= 0.999?
        
    # Polish with 2-swap local search on the best solution found by PGD
    p = best_p
    current_cost = best_cost
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            if time.time() > deadline:
                break
            u = p[i]
            b_u_p = b[u, p]
            b_p_u = b[p, u]
            
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                v = p[j]
                
                # Fast delta calculation
                # Using precomputed vectors for u, but v changes.
                # We can vectorize the j-loop for a fixed i.
                # Let's stick to the O(n) calculation but optimized.
                
                b_v_p = b[v, p]
                b_p_v = b[p, v]
                
                # s_i_u = a[i, :] @ b_u_p
                # ... (same as before)
                
                s_i_u = a[i, :] @ b_u_p
                s_i_v = a[i, :] @ b_v_p
                s_j_u = a[j, :] @ b_u_p
                s_j_v = a[j, :] @ b_v_p
                
                s_u_i = a[:, i] @ b_p_u
                s_v_i = a[:, i] @ b_p_v
                s_u_j = a[:, j] @ b_p_u
                s_v_j = a[:, j] @ b_p_v
                
                s_i_u_clean = s_i_u - a[i, i]*b[u, u] - a[i, j]*b[u, v]
                s_i_v_clean = s_i_v - a[i, i]*b[v, u] - a[i, j]*b[v, v]
                s_j_u_clean = s_j_u - a[j, i]*b[u, u] - a[j, j]*b[u, v]
                s_j_v_clean = s_j_v - a[j, i]*b[v, u] - a[j, j]*b[v, v]
                
                s_u_i_clean = s_u_i - a[i, i]*b[u, u] - a[j, i]*b[v, u]
                s_v_i_clean = s_v_i - a[i, i]*b[u, v] - a[j, i]*b[v, v]
                s_u_j_clean = s_u_j - a[i, j]*b[u, u] - a[j, j]*b[v, u]
                s_v_j_clean = s_v_j - a[i, j]*b[u, v] - a[j, j]*b[v, v]
                
                delta = (s_i_v_clean + s_v_i_clean + s_j_u_clean + s_u_j_clean) - \
                        (s_i_u_clean + s_u_i_clean + s_j_v_clean + s_v_j_clean) + \
                        (a[i,i]*b[v,v] + a[i,j]*b[v,u] + a[j,i]*b[u,v] + a[j,j]*b[u,u]) - \
                        (a[i,i]*b[u,u] + a[i,j]*b[u,v] + a[j,i]*b[v,u] + a[j,j]*b[v,v])
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_delta < 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            improved = True
        else:
            break
            
    return p.tolist()
# EVOLVE-BLOCK-END
