# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    
    # 1. Initial Heuristic: Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    fac_order = np.argsort(-a_row_sums, kind="stable")
    loc_order = np.argsort(b_row_sums, kind="stable")
    
    p_init = np.empty(n, dtype=np.int64)
    p_init[fac_order] = loc_order
    
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = compute_cost(p_init)
    best_cost = current_cost
    best_p = p_init.copy()
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 2. ADMM on Birkhoff Polytope
    # Minimize tr(A X B X^T) subject to X in Birkhoff polytope
    # ADMM: X^{k+1} = argmin tr(A X B X^T) + (rho/2)||X - Z^k + U^k||_F^2
    #        Z^{k+1} = proj_Birkhoff(X^{k+1} - U^k)
    #        U^{k+1} = U^k + X^{k+1} - Z^{k+1}
    
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0
    
    # Start from rank-matching solution
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p_init] = 1.0
    
    rho = 1.0  # ADMM penalty parameter
    U = np.zeros((n, n), dtype=np.float64)  # Dual variable
    
    # We'll track the best solution from ADMM iterations
    admm_best_p = p_init.copy()
    admm_best_cost = best_cost
    
    # Precompute for the QP subproblem
    # The subproblem is: min_X tr(A X B X^T) + (rho/2)||X - (Z-U)||_F^2
    # This is a convex QP. We can solve it via gradient descent or use a spectral approach.
    # For simplicity and speed, we'll use a few steps of projected gradient on the QP,
    # then project to Birkhoff for the Z-update.
    
    # Actually, let's use a simpler approach: 
    # X-update: solve the QP approximately via a few gradient steps
    # Z-update: project to Birkhoff via Hungarian
    
    n_admm_iters = 200
    n_qp_steps = 5  # Inner QP iterations
    
    for k in range(n_admm_iters):
        if time.time() > deadline - 60.0:
            break
        
        # X-update: minimize tr(A X B X^T) + (rho/2)||X - (Z-U)||_F^2
        # Let W = Z - U
        W = X - U  # Initially Z=X, so W=0, but we'll maintain Z separately
        
        # Actually, let's maintain Z explicitly
        Z = X - U  # This is the current Z estimate
        
        # X-update: solve min_X tr(A X B X^T) + (rho/2)||X - Z||_F^2
        # Gradient: 2(A X B + B X A) + rho(X - Z) = 0
        # Use a few steps of gradient descent with step size 1/(2*L + rho)
        
        # Estimate Lipschitz constant of the gradient
        # ||2(A X B + B X A)|| <= 4 * ||A|| * ||B|| * ||X||
        # For X in Birkhoff, ||X||_F <= sqrt(n)
        # Let's use a conservative step size
        
        # Precompute A@B and B@A for efficiency
        AB = A_sym @ B_sym
        BA = B_sym @ A_sym
        
        # Spectral norm estimate for step size
        # ||A||_2 * ||B||_2 can be estimated
        # For simplicity, use a fixed step size that works well empirically
        step_size = 0.5 / (np.linalg.norm(A_sym, ord=2) * np.linalg.norm(B_sym, ord=2) + rho)
        
        X_new = X.copy()
        for inner in range(n_qp_steps):
            # Gradient of tr(A X B X^T) is A X B + B X A (for symmetric A,B)
            AX = A_sym @ X_new
            grad_qp = AX @ B_sym + B_sym @ AX + rho * (X_new - Z)
            X_new -= step_size * grad_qp
        
        # Project X_new to Birkhoff for the Z-update
        row_ind, col_ind = linear_sum_assignment(-X_new)
        Z_new = np.zeros((n, n), dtype=np.float64)
        Z_new[row_ind, col_ind] = 1.0
        
        # Extract permutation from Z_new
        p_admm = np.empty(n, dtype=np.int64)
        p_admm[row_ind] = col_ind
        cost_admm = compute_cost(p_admm)
        
        if cost_admm < admm_best_cost:
            admm_best_cost = cost_admm
            admm_best_p = p_admm.copy()
            if cost_admm < best_cost:
                best_cost = cost_admm
                best_p = p_admm.copy()
        
        # U-update
        U = U + X_new - Z_new
        
        X = Z_new  # Set X to the projected solution for next iteration
    
    # Update best from ADMM
    if admm_best_cost < best_cost:
        best_cost = admm_best_cost
        best_p = admm_best_p.copy()
    
    # 3. Local Search: Vectorized 2-Swap
    p = best_p.copy()
    current_cost = best_cost
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
                
            u = p[i]
            A_i = a[i, :]
            A_iT = a[:, i]
            
            for j in range(i + 1, n):
                if time.time() > deadline - 5.0:
                    break
                
                v = p[j]
                A_j = a[j, :]
                A_jT = a[:, j]
                
                D1 = A_i - A_j
                D2 = A_iT - A_jT
                
                diff_b_vu = b[v, p] - b[u, p]
                diff_b_pv_u = b[p, v] - b[p, u]
                
                Sum1_all = D1 @ diff_b_vu
                Sum2_all = D2 @ diff_b_pv_u
                
                Sub_i_1 = D1[i] * (b[v, u] - b[u, u])
                Sub_i_2 = D2[i] * (b[u, v] - b[u, u])
                
                Sub_j_1 = D1[j] * (b[v, v] - b[u, v])
                Sub_j_2 = D2[j] * (b[v, v] - b[v, u])
                
                Sum1 = Sum1_all - Sub_i_1 - Sub_j_1
                Sum2 = Sum2_all - Sub_i_2 - Sub_j_2
                
                Cross = a[i,j] * (b[v,u] - b[u,v]) + a[j,i] * (b[u,v] - b[v,u])
                
                delta = Cross + Sum1 + Sum2
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_delta < 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            improved = True
        else:
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
