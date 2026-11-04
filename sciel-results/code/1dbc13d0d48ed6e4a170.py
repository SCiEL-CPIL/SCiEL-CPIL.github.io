"""QAP solver for tho150: discrete local search + continuous Birkhoff relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """Compute delta in cost when swapping positions i and j in permutation p.
    p[i] and p[j] are the facilities at locations i and j.
    After swap: facility p[j] goes to location i, facility p[i] goes to location j.
    """
    pi, pj = p[i], p[j]
    # Only terms involving i or j change
    # Cost contribution of (i,j) pair before: a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
    # After: a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
    # Plus interactions with all k != i,j:
    # Before: a[i,k]*b[pi,p[k]] + a[k,i]*b[p[k],pi] + a[j,k]*b[pj,p[k]] + a[k,j]*b[p[k],pj]
    # After:  a[i,k]*b[pj,p[k]] + a[k,i]*b[p[k],pj] + a[j,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi]
    
    # Let's compute delta = new_cost - old_cost for the affected terms
    # Terms involving i (with k != i,j):
    # a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + a[k,i]*(b[p[k],pj] - b[p[k],pi])
    # Terms involving j (with k != i,j):
    # a[j,k]*(b[pi,p[k]] - b[pj,p[k]]) + a[k,j]*(b[p[k],pi] - b[p[k],pj])
    # Terms (i,j) and (j,i):
    # a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
    
    # Combine:
    # For k != i,j:
    # a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + a[k,i]*(b[p[k],pj] - b[p[k],pi])
    # + a[j,k]*(b[pi,p[k]] - b[pj,p[k]]) + a[k,j]*(b[p[k],pi] - b[p[k],pj])
    # = (a[i,k] - a[j,k])*(b[pj,p[k]] - b[pi,p[k]]) + (a[k,i] - a[k,j])*(b[p[k],pj] - b[p[k],pi])
    
    # This is O(n) to compute. Let's vectorize.
    # b_diff_pi_pj = b[pi, :] - b[pj, :]  (shape n)
    # b_diff_pj_pi = b[:, pi] - b[:, pj]  (shape n)
    # a_row_diff_i_j = a[i, :] - a[j, :]  (shape n)
    # a_col_diff_i_j = a[:, i] - a[:, j]  (shape n)
    
    # But we need to exclude k=i and k=j from the sum.
    # Actually, let's just compute the full sum and it works out.
    
    b_pi_pj = b[pi, :]  # b[pi, k] for all k
    b_pj_pi = b[pj, :]  # b[pj, k] for all k
    b_col_pi = b[:, pi]  # b[k, pi] for all k
    b_col_pj = b[:, pj]  # b[k, pj] for all k
    
    a_i_j = a[i, :] - a[j, :]  # a[i,k] - a[j,k]
    a_col_i_j = a[:, i] - a[:, j]  # a[k,i] - a[k,j]
    
    # Sum over all k (including i and j, but those terms will be handled separately or cancel)
    # Actually let's just compute the full delta including all k, and the i,j terms are included naturally.
    
    delta = np.dot(a_i_j, b_pi_pj - b_pj_pi) + np.dot(a_col_i_j, b_col_pj - b_col_pi)
    
    return int(delta)

def _local_search(a, b, p, max_iter=5000, time_limit=None, start_time=None):
    """2-opt local search: try all swaps, take best improving swap."""
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    
    if start_time is None:
        start_time = time.time()
    
    for iteration in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            for j in range(i+1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            break  # Local minimum
        
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
    
    return p, current_cost

def _birkhoff_relaxation(a, b, p, n_iter=200, time_limit=None, start_time=None):
    """Continuous relaxation: project to Birkhoff polytope using Sinkhorn-like iterations.
    
    We work with the assignment matrix X where X[i, p[i]] = 1.
    The QAP objective in matrix form: sum_{i,j,k,l} a[i,j] * b[k,l] * X[i,k] * X[j,l]
    
    We use a gradient-based approach on the Birkhoff polytope.
    """
    n = len(p)
    
    # Build initial assignment matrix
    X = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        X[i, p[i]] = 1.0
    
    if start_time is None:
        start_time = time.time()
    
    # Compute gradient of QAP objective w.r.t. X
    # f(X) = sum_{i,j,k,l} a[i,j] * b[k,l] * X[i,k] * X[j,l]
    # df/dX[i,k] = sum_{j,l} a[i,j] * b[k,l] * X[j,l] + sum_{j,l} a[j,i] * b[k,l] * X[j,l]
    #            = sum_j a[i,j] * (b @ X)[k, j] ... let me re-derive
    
    # f(X) = trace(X^T * A * X * B^T) where A=a, B=b
    # Actually: f = sum_{i,k} X[i,k] * sum_{j,l} a[i,j] * X[j,l] * b[k,l]
    # Let C = A @ X @ B^T, so C[i,k] = sum_{j,l} a[i,j] * X[j,l] * b[k,l]
    # Then f = sum_{i,k} X[i,k] * C[i,k] = trace(X^T C)
    # df/dX[i,k] = C[i,k] + (conjugate symmetry terms)
    
    # Actually for the QAP: f(X) = sum_{i,j} a[i,j] * sum_{k,l} b[k,l] * X[i,k] * X[j,l]
    # = sum_{i,j,k,l} a[i,j] * b[k,l] * X[i,k] * X[j,l]
    
    # df/dX[i,k] = sum_{j,l} a[i,j] * b[k,l] * X[j,l] + sum_{j,l} a[j,i] * b[l,k] * X[j,l]
    #             = sum_j a[i,j] * (B @ X^T)[k, j] ... 
    
    # Let me define: M[i,k] = sum_{j,l} a[i,j] * X[j,l] * b[k,l]
    # This is (A @ X @ B^T)[i,k]
    
    # And the second term: sum_{j,l} a[j,i] * b[l,k] * X[j,l] = (A^T @ X @ B)[i,k]
    
    # So gradient = A @ X @ B^T + A^T @ X @ B
    
    best_X = X.copy()
    best_cost = _qap_cost(a, b, p)
    
    for it in range(n_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Compute gradient
        C1 = a @ X @ b.T  # (n,n)
        C2 = a.T @ X @ b   # (n,n)
        grad = C1 + C2
        
        # Project gradient to Birkhoff polytope: we want to minimize, so move in -grad direction
        # Use Sinkhorn to project (X - eta * grad) to Birkhoff polytope
        # Simpler: use a multiplicative update / Sinkhorn on (X * exp(-eta * grad))
        
        # Adaptive step size
        eta = 0.01
        
        # Sinkhorn-Knopp: find doubly stochastic Y closest to exp(-eta * grad) in some sense
        # Actually, let's use: Y = exp(-eta * grad), then Sinkhorn normalize
        # But this might overflow. Let's use a different approach.
        
        # Alternative: use the fact that the Euclidean projection onto Birkhoff polytope
        # can be approximated by Sinkhorn iterations on exp(-grad/tau)
        
        # For now, let's use a simple gradient step with Birkhoff projection via Sinkhorn
        log_Y = -eta * grad
        # Shift for numerical stability
        log_Y -= log_Y.max()
        Y = np.exp(log_Y)
        
        # Sinkhorn iterations to make Y doubly stochastic
        for _ in range(50):
            Y /= Y.sum(axis=1, keepdims=True)
            Y /= Y.sum(axis=0, keepdims=True)
        
        X = Y
        
        # Extract permutation from X using Hungarian
        row_ind, col_ind = linear_sum_assignment(-X)
        p_new = np.zeros(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        cost_new = _qap_cost(a, b, p_new)
        if cost_new < best_cost:
            best_cost = cost_new
            best_X = X.copy()
    
    # Final extraction
    row_ind, col_ind = linear_sum_assignment(-best_X)
    p_final = np.zeros(n, dtype=np.int64)
    p_final[row_ind] = col_ind
    final_cost = _qap_cost(a, b, p_final)
    
    if final_cost < best_cost:
        best_cost = final_cost
    
    return p_final, best_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s kill
    
    # Initial solution: rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.tolist()
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, np.array(p))
    
    # Phase 1: Local search from initial
    remaining = deadline - (time.time() - start_time)
    if remaining > 5:
        p_ls, cost_ls = _local_search(a, b, np.array(p), max_iter=10000, 
                                        time_limit=remaining - 2, start_time=start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.tolist()
    
    # Phase 2: Birkhoff relaxation from best discrete solution
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        p_birk, cost_birk = _birkhoff_relaxation(a, b, np.array(best_p), 
                                                   n_iter=100,
                                                   time_limit=remaining - 2, 
                                                   start_time=start_time)
        if cost_birk < best_cost:
            best_cost = cost_birk
            best_p = p_birk.tolist()
    
    # Phase 3: Local search from Birkhoff result
    remaining = deadline - (time.time() - start_time)
    if remaining > 5:
        p_ls2, cost_ls2 = _local_search(a, b, np.array(best_p), max_iter=20000,
                                          time_limit=remaining - 1, start_time=start_time)
        if cost_ls2 < best_cost:
            best_cost = cost_ls2
            best_p = p_ls2.tolist()
    
    # Phase 4: Another Birkhoff + local search if time permits
    remaining = deadline - (time.time() - start_time)
    if remaining > 15:
        p_birk2, cost_birk2 = _birkhoff_relaxation(a, b, np.array(best_p),
                                                     n_iter=80,
                                                     time_limit=remaining * 0.4,
                                                     start_time=start_time)
        if cost_birk2 < best_cost:
            best_cost = cost_birk2
            best_p = p_birk2.tolist()
        
        remaining = deadline - (time.time() - start_time)
        if remaining > 5:
            p_ls3, cost_ls3 = _local_search(a, b, np.array(best_p), max_iter=20000,
                                              time_limit=remaining - 1, start_time=start_time)
            if cost_ls3 < best_cost:
                best_cost = cost_ls3
                best_p = p_ls3.tolist()
    
    # Final check
    remaining = deadline - (time.time() - start_time)
    if remaining > 3:
        p_final, cost_final = _local_search(a, b, np.array(best_p), max_iter=5000,
                                              time_limit=remaining - 0.5, start_time=start_time)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.tolist()
    
    return best_p
# EVOLVE-BLOCK-END
