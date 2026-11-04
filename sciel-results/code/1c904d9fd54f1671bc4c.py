"""QAP tai256c solver: vectorized 2-opt with O(n) swap deltas, Hungarian re-optimization, and iterated local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 120s
    rng = random.Random(42)

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def fast_swap_delta(p, u, v):
        """Compute cost delta for swapping p[u] and p[v] in O(n)."""
        pu = p[u]
        pv = p[v]
        
        # Vectorized O(n) delta computation
        # delta = sum_j a[u,j]*(b[pv,p[j]] - b[pu,p[j]]) 
        #       + sum_i a[i,u]*(b[p[i],pv] - b[p[i],pu])
        #       + sum_j a[v,j]*(b[pu,p[j]] - b[pv,p[j]])
        #       + sum_i a[i,v]*(b[p[i],pu] - b[p[i],pv])
        #       - 2*a[u,v]*(b[pv,pu] - b[pu,pv])
        #       - 2*a[v,u]*(b[pu,pv] - b[pv,pu])
        #       + a[u,u]*(b[pv,pv] - b[pu,pu])
        #       + a[v,v]*(b[pu,pu] - b[pv,pv])
        
        # Let's compute this more carefully using the full expansion
        # The affected terms are all (i,j) where i in {u,v} or j in {u,v}
        
        # Before swap:
        # T_before = sum_j a[u,j]*b[pu,p[j]] + sum_i a[i,u]*b[p[i],pu] 
        #         + sum_j a[v,j]*b[pv,p[j]] + sum_i a[i,v]*b[p[i],pv]
        #         - a[u,u]*b[pu,pu] - a[v,v]*b[pv,pv]  # subtract diagonal double-count
        #         - a[u,v]*b[pu,pv] - a[v,u]*b[pv,pu]  # subtract cross double-count
        
        # After swap (p[u]=pv, p[v]=pu):
        # T_after = sum_j a[u,j]*b[pv,p[j]] + sum_i a[i,u]*b[p[i],pv] 
        #         + sum_j a[v,j]*b[pu,p[j]] + sum_i a[i,v]*b[p[i],pu]
        #         - a[u,u]*b[pv,pv] - a[v,v]*b[pu,pu]
        #         - a[u,v]*b[pv,pu] - a[v,u]*b[pu,pv]
        
        # delta = T_after - T_before
        
        # Vectorized computation:
        # Row u before: a[u,:] @ b[pu, p] -> but p is the location array
        # Let's use: sum_j a[u,j]*b[pu,p[j]] = a[u,:] @ b[pu, p]
        
        row_u_before = np.dot(a[u, :], b[pu, p])
        row_u_after = np.dot(a[u, :], b[pv, p])
        
        col_u_before = np.dot(a[:, u], b[p, pu])
        col_u_after = np.dot(a[:, u], b[p, pv])
        
        row_v_before = np.dot(a[v, :], b[pv, p])
        row_v_after = np.dot(a[v, :], b[pu, p])
        
        col_v_before = np.dot(a[:, v], b[p, pv])
        col_v_after = np.dot(a[:, v], b[p, pu])
        
        # Diagonal corrections
        diag_u_before = a[u, u] * b[pu, pu]
        diag_u_after = a[u, u] * b[pv, pv]
        
        diag_v_before = a[v, v] * b[pv, pv]
        diag_v_after = a[v, v] * b[pu, pu]
        
        # Cross corrections
        cross_uv_before = a[u, v] * b[pu, pv] + a[v, u] * b[pv, pu]
        cross_uv_after = a[u, v] * b[pv, pu] + a[v, u] * b[pu, pv]
        
        delta = (row_u_after + col_u_after + row_v_after + col_v_after 
                 - diag_u_after - diag_v_after - cross_uv_after) - \
                (row_u_before + col_u_before + row_v_before + col_v_before 
                 - diag_u_before - diag_v_before - cross_uv_before)
        
        return int(delta)

    def two_opt_pass(p, cost, max_passes=10):
        """Multiple passes of 2-opt with O(n) delta computation."""
        for _pass in range(max_passes):
            if time.time() > deadline - 1.0:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline - 1.0:
                    return p, cost
                for j in range(i + 1, n):
                    if time.time() > deadline - 1.0:
                        return p, cost
                    delta = fast_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
            if not improved:
                break
        return p, cost

    def hungarian_reassign(p):
        """Use Hungarian algorithm to find a better assignment."""
        # Compute the cost matrix C[i,j] = marginal cost of placing facility i at location j
        # C[i,j] = sum_k a[i,k]*b[j,p[k]] + sum_k a[k,i]*b[p[k],j] - a[i,i]*b[j,j]
        
        # Vectorized computation
        # b[j, p] gives us b[j, p[k]] for all k, so b[:, p] is n x n
        # a[:, i] is the column i of a
        # So sum_k a[i,k]*b[j,p[k]] = b[j,:] @ a[:,i]... wait, that's not right
        # sum_k a[i,k]*b[j,p[k]] = (a[i,:] * b[j,p]).sum()
        
        # For all i,j: C[i,j] = sum_k a[i,k]*b[j,p[k]] + sum_k a[k,i]*b[p[k],j] - a[i,i]*b[j,j]
        
        # First term: for fixed j, sum_k a[i,k]*b[j,p[k]] = a[i,:] @ b[j,p]
        # So first_term[i,j] = a[i,:] @ b[j,p] for all i
        # This is: a @ b[:, p].T ... no
        # b[j,p] is a vector of length n (the row j of b indexed by p)
        # a[i,:] @ b[j,p] = sum_k a[i,k]*b[j,p[k]]
        # For all i,j: this is a @ (b[:, p]).T? No.
        # Let B_p = b[:, p] which is n x n where B_p[j,k] = b[j,p[k]]
        # Then first_term[i,j] = sum_k a[i,k]*B_p[j,k] = a[i,:] @ B_p[j,:]
        # So first_term = a @ B_p.T = a @ b[:, p].T
        
        B_p = b[:, p]  # B_p[j,k] = b[j, p[k]]
        first_term = a @ B_p.T  # first_term[i,j] = sum_k a[i,k]*b[j,p[k]]
        
        # Second term: sum_k a[k,i]*b[p[k],j]
        # For fixed i,j: sum_k a[k,i]*b[p[k],j]
        # a[k,i] is column i of a
        # b[p[k],j] is column j of b indexed by p
        # Let B_p2 = b[p, :] which is n x n where B_p2[k,j] = b[p[k],j]
        # Then sum_k a[k,i]*B_p2[k,j] = (a[:,i].T @ B_p2[:,j])... 
        # Actually: sum_k a[k,i]*b[p[k],j] = a[:,i].T @ b[p,j]
        # For all i,j: second_term[i,j] = a[:,i].T @ b[p,j]
        # This is: a.T @ b[p, :].T ... let me think
        # a[:,i].T is a row vector, b[p,j] is a column vector
        # sum_k a[k,i]*b[p[k],j] = (a[:,i] * b[p,j]).sum()
        # For all i,j: second_term = (a.T * b[p, :].T).sum(axis=1)... no
        # Let me just do: second_term[i,j] = np.dot(a[:,i], b[p,j])
        # Vectorized: second_term = a.T @ b[p, :].T? 
        # a.T[i,k] = a[k,i], b[p,:].T[k,j] = b[p[k],j]
        # (a.T @ b[p,:].T)[i,j] = sum_k a.T[i,k]*b[p,:].T[k,j] = sum_k a[k,i]*b[p[k],j] ✓
        
        second_term = a.T @ b[p, :].T
        
        # Third term: -a[i,i]*b[j,j]
        diag_a = np.diag(a)
        diag_b = np.diag(b)
        third_term = -np.outer(diag_a, diag_b)
        
        C = first_term + second_term + third_term
        
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[col_ind] = row_ind
        return new_p

    # Initial solution: rank matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    
    # Initial 2-opt
    p, cost = two_opt_pass(p.copy(), best_cost, max_passes=5)
    if cost < best_cost:
        best_p = p.copy()
        best_cost = cost
    
    # Iterated local search
    while time.time() < deadline - 5.0:
        # Perturb: random swaps
        p = best_p.copy()
        n_swaps = rng.randint(2, 6)
        for _ in range(n_swaps):
            i, j = rng.sample(range(n), 2)
            p[i], p[j] = p[j], p[i]
        
        # Local search from perturbed solution
        p, cost = two_opt_pass(p, compute_cost(p), max_passes=3)
        
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
        
        # Hungarian reassignment periodically
        if time.time() < deadline - 10.0:
            new_p = hungarian_reassign(p)
            new_cost = compute_cost(new_p)
            if new_cost < best_cost:
                best_p = new_p.copy()
                best_cost = new_cost
                # Polish the Hungarian solution
                p2, c2 = two_opt_pass(new_p.copy(), best_cost, max_passes=3)
                if c2 < best_cost:
                    best_p = p2.copy()
                    best_cost = c2
    
    # Final high-precision polish: multiple 2-opt passes on the best solution
    if time.time() < deadline - 1.0:
        p_final = best_p.copy()
        c_final = best_cost
        for _ in range(5):
            if time.time() > deadline - 0.5:
                break
            p_final, c_final = two_opt_pass(p_final, c_final, max_passes=2)
        if c_final < best_cost:
            best_p = p_final
            best_cost = c_final
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
