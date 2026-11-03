"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initialization: Rank matching
    # Heavy flows to short distances
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy_order = np.argsort(-a_sum, kind="stable")
    # Sort locations by ascending distance sum
    central_order = np.argsort(b_sum, kind="stable")
    
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy_order] = central_order
    
    # Calculate initial cost
    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
        
    best_cost = calc_cost(p_init)
    best_p = p_init.copy()
    
    # 2. Continuous Relaxation (Birkhoff Polytope)
    # We formulate the QAP as minimizing Tr(P^T A P B) where P is a permutation matrix.
    # We relax P to be a doubly stochastic matrix X (X in Birkhoff polytope).
    # Objective: f(X) = sum_{i,j,k,l} A[i,j] X[i,k] B[k,l] X[j,l]
    # Gradient wrt X[i,k]: 
    # grad[i,k] = sum_{j,l} A[i,j] B[k,l] X[j,l] + sum_{j,l} A[j,i] B[l,k] X[j,l]
    # Using matrix multiplication:
    # Let M1 = A @ X @ B.T
    # Let M2 = A.T @ X @ B
    # Then grad = M1 + M2
    
    # Initialize X as a doubly stochastic matrix based on p_init
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p_init] = 1.0
    
    # Precompute constants for gradient
    # grad = A @ X @ B.T + A.T @ X @ B
    # To save time, we can update X and recompute gradient.
    
    # Projected Gradient Descent
    # Step size needs to be small. 
    # We will use a simple step size decay.
    step_size = 0.01
    min_step_size = 0.0001
    
    # Helper: Project onto Birkhoff Polytope (Doubly Stochastic)
    # Using Sinkhorn-Knopp algorithm (iterative scaling)
    def project_birkhoff(X, max_iters=100, tol=1e-6):
        # Ensure non-negative
        X = np.maximum(X, 1e-12)
        
        # Normalize rows and columns
        for _ in range(max_iters):
            # Row sums
            r = X.sum(axis=1, keepdims=True)
            r[r == 0] = 1
            X = X / r
            # Column sums
            c = X.sum(axis=0, keepdims=True)
            c[c == 0] = 1
            X = X / c
            
            # Check convergence
            if np.max(np.abs(X.sum(axis=1) - 1)) < tol and np.max(np.abs(X.sum(axis=0) - 1)) < tol:
                break
                
        # Final correction to ensure exact sums if possible (Hungarian on the result is better)
        # For the purpose of gradient descent, approximate projection is okay,
        # but for the final integer recovery, we use Hungarian.
        return X

    # Main Loop for Continuous Relaxation
    # We alternate between continuous relaxation and integer recovery (Local Search)
    # This is a "Hybrid" approach.
    
    current_X = X.copy()
    no_improve_count = 0
    max_no_improve = 50
    
    # We will run a few rounds of PGM (Projected Gradient Method)
    # Each round:
    # 1. Do some PGM steps
    # 2. Recover integer solution via Hungarian
    # 3. If improved, keep it. If not, perturb.
    
    for _ in range(20): # Max 20 rounds
        if time.time() > deadline:
            break
            
        # PGM Phase
        for _ in range(20): # 20 PGM steps per round
            if time.time() > deadline:
                break
            
            # Compute Gradient
            # grad = A @ X @ B.T + A.T @ X @ B
            # Note: A and B are symmetric in QAP usually, but let's be general.
            # QAP cost is sum a_ij b_kl x_ik x_jl
            # d/dx_ik = sum_j,l a_ij b_kl x_jl + sum_j,l a_ji b_lk x_jl
            # = (A @ X @ B.T)[i,k] + (A.T @ X @ B)[i,k]
            
            # Optimization: 
            # T1 = A @ X
            # T2 = T1 @ B.T
            # T3 = A.T @ X
            # T4 = T3 @ B
            # Grad = T2 + T4
            
            T1 = a @ current_X
            T2 = T1 @ b.T
            T3 = a.T @ current_X
            T4 = T3 @ b
            
            grad = T2 + T4
            
            # Update X
            current_X = current_X - step_size * grad
            
            # Project to Birkhoff
            current_X = project_birkhoff(current_X, max_iters=50)
            
            # Decay step size
            step_size = max(min_step_size, step_size * 0.95)
            
        # Integer Recovery Phase
        # Solve linear assignment problem on the cost matrix derived from X
        # The cost for assigning facility i to location k in the linear relaxation is roughly
        # related to the gradient or the "reduced cost".
        # A common heuristic is to use the matrix M = A @ X @ B.T + A.T @ X @ B as the cost matrix
        # for the Hungarian algorithm, or simply use the current X as a hint.
        # Actually, the standard approach is to minimize the linearization.
        # Let's use the gradient matrix as the cost for Hungarian, or better,
        # just run Hungarian on a cost matrix that approximates the QAP cost given X.
        
        # A better recovery:
        # The cost of permutation p is Tr(P^T A P B).
        # We can approximate this by linearizing around X.
        # Or simply, we can use the matrix C = A @ X @ B.T + A.T @ X @ B
        # and find the permutation that minimizes sum C[i, p[i]].
        # This is a valid heuristic.
        
        cost_matrix = T2 + T4 # Reuse from last step
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        p_candidate = col_ind.copy()
        
        # Calculate cost
        cost_candidate = calc_cost(p_candidate)
        
        if cost_candidate < best_cost:
            best_cost = cost_candidate
            best_p = p_candidate.copy()
            no_improve_count = 0
            # Restart X from this best solution
            current_X = np.zeros((n, n), dtype=np.float64)
            current_X[np.arange(n), best_p] = 1.0
            step_size = 0.01 # Reset step size
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturb best solution and restart
                p_perturbed = best_p.copy()
                for _ in range(10):
                    i = np.random.randint(n)
                    j = np.random.randint(n)
                    if i != j:
                        p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
                
                cost_perturbed = calc_cost(p_perturbed)
                if cost_perturbed < best_cost:
                    best_cost = cost_perturbed
                    best_p = p_perturbed.copy()
                
                current_X = np.zeros((n, n), dtype=np.float64)
                current_X[np.arange(n), best_p] = 1.0
                step_size = 0.01
                no_improve_count = 0

    # Final Polish: Local Search on the best solution found
    # Use the swap-based local search for the remaining time to polish the result
    p = best_p.copy()
    current_cost = best_cost
    
    tabu_size = 30
    tabu_list = {}
    current_step = 0
    no_improve_ls = 0
    max_no_improve_ls = 1000
    
    while time.time() < deadline:
        if no_improve_ls > max_no_improve_ls:
            # Perturb
            for _ in range(10):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = calc_cost(p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            no_improve_ls = 0
            tabu_list.clear()
            continue
            
        Bp = b[:, p]
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        for i in range(n):
            u = p[i]
            ai = a[i]
            for j in range(i + 1, n):
                key = (i, j)
                if key in tabu_list and tabu_list[key] > current_step:
                    continue
                
                v = p[j]
                delta = np.dot(ai - a[j], Bp[v] - Bp[u])
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
                    
        if best_i == -1:
            tabu_list.clear()
            continue
            
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += int(best_delta)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_ls = 0
        else:
            no_improve_ls += 1
            
        tabu_list[(best_i, best_j)] = current_step + tabu_size
        tabu_list[(best_j, best_i)] = current_step + tabu_size
        
        current_step += 1
        
        if current_step % 100 == 0:
            keys_to_remove = [k for k, v in tabu_list.items() if v <= current_step]
            for k in keys_to_remove:
                del tabu_list[k]

    return best_p.tolist()
# EVOLVE-BLOCK-END
