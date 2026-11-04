"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Helper: Calculate exact cost
    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
        
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
    
    # Precompute structures for fast local search
    # dot_row[i, l] = sum_k a[i,k] * b[l, p[k]]
    # dot_col[i, l] = sum_k a[k,i] * b[p[k], l]
    
    def recompute_precomp(p):
        b_col_p = b[:, p]  # (n, n): [l, k] = b[l, p[k]]
        b_row_p = b[p, :]  # (n, n): [k, l] = b[p[k], l]
        dot_row = a @ b_col_p.T
        dot_col = a.T @ b_row_p
        return dot_row, dot_col
    
    dot_row, dot_col = recompute_precomp(p)
    b_diag = np.diag(b)
    
    tabu_size = 30
    tabu_list = {}
    current_step = 0
    no_improve_ls = 0
    max_no_improve_ls = 500 # Lower threshold to allow more perturbations if stuck
    
    while time.time() < deadline:
        if no_improve_ls > max_no_improve_ls:
            # Perturb
            for _ in range(10):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = calc_cost(p)
            dot_row, dot_col = recompute_precomp(p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            no_improve_ls = 0
            tabu_list.clear()
            continue
            
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        # Sample a subset of pairs to speed up search if n is large
        # For n=150, full scan is 150*149/2 ~ 11k pairs.
        # Each delta is O(1) with precomputation. 11k O(1) ops is fast.
        # Let's try full scan first, but with a time check.
        
        for i in range(n):
            if time.time() > deadline:
                break
            u = p[i]
            ai = a[i]
            for j in range(i + 1, n):
                key = (i, j)
                if key in tabu_list and tabu_list[key] > current_step:
                    continue
                
                v = p[j]
                
                # Fast delta calculation using precomputed dot_row and dot_col
                # delta = (dot_row[i, v] - dot_row[i, u]) + (dot_row[j, u] - dot_row[j, v])
                #       + (dot_col[i, v] - dot_col[i, u]) + (dot_col[j, u] - dot_col[j, v])
                #       - correction terms for i,j interactions
                # 
                # Let's use the vectorized dot product approach from the inspiration
                # which is robust:
                # delta = np.dot(ai - a[j], b[v, p] - b[u, p]) ... wait, b[p] is the permuted matrix.
                # Actually, the inspiration's formula was:
                # delta = np.dot(ai - a[j], Bp[v] - Bp[u]) where Bp = b[:, p]
                # Let's verify this.
                # Cost = sum_{k,l} a[k,l] b[p[k], p[l]]
                # Swap p[i] <-> p[j]. u=p[i], v=p[j].
                # Change in cost:
                # Terms involving i or j change.
                # sum_k a[i,k] b[p[i], p[k]] -> sum_k a[i,k] b[v, p[k]] (if k!=i) + a[i,i]b[v,v]
                # sum_k a[k,i] b[p[k], p[i]] -> sum_k a[k,i] b[p[k], v] (if k!=i) + a[i,i]b[v,v]
                # This is getting complicated to derive from scratch quickly.
                # The vectorized dot product:
                # Let Bp = b[:, p]. Then Bp[l, k] = b[l, p[k]].
                # The term sum_k a[i,k] b[p[i], p[k]] is dot(a[i, :], Bp[p[i], :]).
                # After swap, p[i]=v. New term is dot(a[i, :], Bp[v, :]).
                # But a[i,j] and a[j,i] terms are handled within the vectors.
                # Actually, the formula delta = np.dot(ai - a[j], Bp[v] - Bp[u])
                # assumes symmetry or specific structure?
                # Let's stick to the proven O(1) incremental update logic from inspiration 1
                # or just use the vectorized dot product which is O(n) per pair.
                # With n=150, O(n) per pair is 150 ops. 11k pairs * 150 ops = 1.6M ops per full scan.
                # 1.6M ops is very fast in NumPy (vectorized).
                # So we can do a full scan every iteration.
                
                # Calculate delta using vectorized operations
                # Bp = b[:, p]
                # We need Bp[v] and Bp[u]
                # Bp is (n, n). Bp[v] is the row v of Bp, which is b[v, p].
                # Bp[u] is b[u, p].
                
                # Wait, the formula in inspiration 1:
                # delta = np.dot(ai - a[j], Bp[v] - Bp[u])
                # Let's check dimensions.
                # ai is (n,). Bp[v] is (n,). Dot is scalar.
                # Is this correct?
                # Cost contribution of row i: sum_k a[i,k] b[p[i], p[k]]
                # Cost contribution of col i: sum_k a[k,i] b[p[k], p[i]]
                # If we swap p[i] and p[j]:
                # New row i cost: sum_k a[i,k] b[v, p[k]] (with p[j]=u now)
                # Old row i cost: sum_k a[i,k] b[u, p[k]] (with p[j]=v now)
                # Difference: sum_k a[i,k] (b[v, p[k]] - b[u, p[k]])
                # Note: for k=j, p[j] changes from v to u.
                # For k!=j, p[k] is unchanged.
                # So sum_{k!=j} a[i,k] (b[v, p[k]] - b[u, p[k]]) + a[i,j] (b[v, u] - b[u, v])
                
                # This is getting messy. The vectorized dot product `np.dot(ai, Bp[v])`
                # calculates sum_k a[i,k] b[v, p[k]].
                # This includes the term k=j: a[i,j] b[v, p[j]] = a[i,j] b[v, u] (since p[j] becomes u).
                # And k=i: a[i,i] b[v, p[i]] = a[i,i] b[v, v] (since p[i] becomes v).
                # So `np.dot(ai, Bp[v])` is exactly the new row i cost.
                # Similarly `np.dot(ai, Bp[u])` is the old row i cost.
                # So RowDiff = np.dot(ai, Bp[v] - Bp[u]).
                
                # What about column i?
                # Old col i cost: sum_k a[k,i] b[p[k], u]
                # New col i cost: sum_k a[k,i] b[p[k], v]
                # ColDiff = sum_k a[k,i] (b[p[k], v] - b[p[k], u])
                # This is `np.dot(a[:, i], Bp[:, v] - Bp[:, u])`?
                # Bp[:, v] is the column v of Bp, which is b[:, p[v]] = b[:, u] (since p[v] is not p[i] or p[j]... wait).
                # Bp[l, k] = b[l, p[k]].
                # Bp[:, v] means l varies, k=v. So Bp[l, v] = b[l, p[v]].
                # This is NOT b[p[k], v].
                
                # Let's use the explicit O(n) calculation which is safer.
                # delta = sum_{k!=i,j} a[i,k](b[v,p[k]]-b[u,p[k]]) + a[i,i](b[v,v]-b[u,u]) + a[i,j](b[v,u]-b[u,v])
                #       + sum_{k!=i,j} a[k,i](b[p[k],v]-b[p[k],u]) + a[i,i](b[v,v]-b[u,u]) + a[j,i](b[u,v]-b[v,u])
                #       + sum_{k!=i,j} a[j,k](b[u,p[k]]-b[v,p[k]]) + a[j,j](b[u,u]-b[v,v]) + a[j,i](b[u,v]-b[v,u])
                #       + sum_{k!=i,j} a[k,j](b[p[k],u]-b[p[k],v]) + a[i,j](b[u,v]-b[v,u]) + a[j,j](b[u,u]-b[v,v])
                
                # This is 4 sums of O(n) terms.
                # We can compute these sums using precomputed dot_row and dot_col.
                
                # dot_row[i, l] = sum_k a[i,k] b[l, p[k]]
                # sum_{k!=i,j} a[i,k] b[v, p[k]] = dot_row[i, v] - a[i,i]b[v, p[i]] - a[i,j]b[v, p[j]]
                # p[i]=u, p[j]=v.
                # = dot_row[i, v] - a[i,i]b[v, u] - a[i,j]b[v, v]
                
                # sum_{k!=i,j} a[i,k] b[u, p[k]] = dot_row[i, u] - a[i,i]b[u, u] - a[i,j]b[u, v]
                
                # So RowPart_i = (dot_row[i, v] - a[i,i]b[v, u] - a[i,j]b[v, v]) - (dot_row[i, u] - a[i,i]b[u, u] - a[i,j]b[u, v])
                # = dot_row[i, v] - dot_row[i, u] - a[i,i](b[v, u] - b[u, u]) - a[i,j](b[v, v] - b[u, v])
                
                # Similarly for RowPart_j:
                # Old row j: sum a[j,k] b[v, p[k]] (p[i]=u, p[j]=v)
                # New row j: sum a[j,k] b[u, p[k]] (p[i]=v, p[j]=u)
                # RowPart_j = (dot_row[j, u] - a[j,i]b[u, v] - a[j,j]b[u, u]) - (dot_row[j, v] - a[j,i]b[v, u] - a[j,j]b[v, v])
                # = dot_row[j, u] - dot_row[j, v] - a[j,i](b[u, v] - b[v, u]) - a[j,j](b[u, u] - b[v, v])
                
                # Now Column parts.
                # dot_col[i, l] = sum_k a[k,i] b[p[k], l]
                # sum_{k!=i,j} a[k,i] b[p[k], v] = dot_col[i, v] - a[i,i]b[p[i], v] - a[j,i]b[p[j], v]
                # p[i]=u, p[j]=v.
                # = dot_col[i, v] - a[i,i]b[u, v] - a[j,i]b[v, v]
                
                # sum_{k!=i,j} a[k,i] b[p[k], u] = dot_col[i, u] - a[i,i]b[u, u] - a[j,i]b[v, u]
                
                # ColPart_i = (dot_col[i, v] - a[i,i]b[u, v] - a[j,i]b[v, v]) - (dot_col[i, u] - a[i,i]b[u, u] - a[j,i]b[v, u])
                # = dot_col[i, v] - dot_col[i, u] - a[i,i](b[u, v] - b[u, u]) - a[j,i](b[v, v] - b[v, u])
                
                # ColPart_j:
                # Old col j: sum a[k,j] b[p[k], v]
                # New col j: sum a[k,j] b[p[k], u]
                # ColPart_j = (dot_col[j, u] - a[i,j]b[p[i], u] - a[j,j]b[p[j], u]) - (dot_col[j, v] - a[i,j]b[p[i], v] - a[j,j]b[p[j], v])
                # p[i]=u, p[j]=v.
                # = (dot_col[j, u] - a[i,j]b[u, u] - a[j,j]b[v, u]) - (dot_col[j, v] - a[i,j]b[u, v] - a[j,j]b[v, v])
                # = dot_col[j, u] - dot_col[j, v] - a[i,j](b[u, u] - b[u, v]) - a[j,j](b[v, u] - b[v, v])
                
                # Total Delta = RowPart_i + RowPart_j + ColPart_i + ColPart_j
                
                buu = b_diag[u]
                bvv = b_diag[v]
                buv = b[u, v]
                bvu = b[v, u]
                
                delta = 0.0
                # RowPart_i
                delta += (dot_row[i, v] - dot_row[i, u]) - a[i,i]*(bvu - buu) - a[i,j]*(bvv - buv)
                # RowPart_j
                delta += (dot_row[j, u] - dot_row[j, v]) - a[j,i]*(buv - bvu) - a[j,j]*(buu - bvv)
                # ColPart_i
                delta += (dot_col[i, v] - dot_col[i, u]) - a[i,i]*(buv - buu) - a[j,i]*(bvv - bvu)
                # ColPart_j
                delta += (dot_col[j, u] - dot_col[j, v]) - a[i,j]*(buu - buv) - a[j,j]*(bvu - bvv)
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
                    
        if best_i == -1:
            tabu_list.clear()
            no_improve_ls += 1
            continue
            
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += int(best_delta)
        
        # Update dot_row and dot_col incrementally in O(n)
        u, v = p[best_j], p[best_i] # Note: p[best_i] is now v, p[best_j] is now u
        # Wait, we swapped p[best_i] and p[best_j].
        # Old u = p[best_i], Old v = p[best_j].
        # New p[best_i] = v, New p[best_j] = u.
        
        # dot_row[k, l] = sum_m a[k,m] b[l, p[m]]
        # Change in p: p[best_i] u->v, p[best_j] v->u.
        # For any k, dot_row[k, l] changes by:
        # a[k, best_i] * (b[l, v] - b[l, u]) + a[k, best_j] * (b[l, u] - b[l, v])
        # = (a[k, best_i] - a[k, best_j]) * (b[l, v] - b[l, u])
        
        # Let delta_p = b[:, v] - b[:, u] (vector of size n)
        # delta_row[k, l] = (a[k, best_i] - a[k, best_j]) * delta_p[l]
        # This is an outer product.
        delta_row_vec = a[:, best_i] - a[:, best_j]
        delta_p_vec = b[:, v] - b[:, u]
        dot_row += np.outer(delta_row_vec, delta_p_vec)
        
        # dot_col[k, l] = sum_m a[m, k] b[p[m], l]
        # Change in p: p[best_i] u->v, p[best_j] v->u.
        # dot_col[k, l] changes by:
        # a[best_i, k] * (b[v, l] - b[u, l]) + a[best_j, k] * (b[u, l] - b[v, l])
        # = (a[best_i, k] - a[best_j, k]) * (b[v, l] - b[u, l])
        
        # Let delta_col_vec = a[best_i, :] - a[best_j, :] (vector of size n)
        # Let delta_p_row_vec = b[v, :] - b[u, :] (vector of size n)
        # dot_col[k, l] += delta_col_vec[k] * delta_p_row_vec[l]
        
        delta_col_vec = a[best_i, :] - a[best_j, :]
        delta_p_row_vec = b[v, :] - b[u, :]
        dot_col += np.outer(delta_col_vec, delta_p_row_vec)
        
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
            keys_to_remove = [k for k, v_val in tabu_list.items() if v_val <= current_step]
            for k in keys_to_remove:
                del tabu_list[k]

    return best_p.tolist()
# EVOLVE-BLOCK-END
