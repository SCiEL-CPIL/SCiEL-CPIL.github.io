"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def solve(a, b):
    n = len(a)
    
    # Convert to float for numerical stability
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    
    start_time = time.time()
    time_budget = 115.0  # Leave 5 seconds for final operations
    
    # --- Phase 1: Initial heuristic assignment ---
    # Match high-flow facilities to low-distance locations
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # --- Phase 2: Sinkhorn-based optimization on Birkhoff polytope ---
    def sinkhorn_kernel(K, num_iters, eps):
        """Sinkhorn-Knopp to project onto Birkhoff polytope with entropic regularization."""
        n_local = K.shape[0]
        K_safe = np.maximum(K, 1e-12)
        K_safe = K_safe / (K_safe.max() + 1e-12)
        
        u = np.ones(n_local)
        v = np.ones(n_local)
        
        for _ in range(num_iters):
            u = 1.0 / (eps * K_safe @ v + 1e-12)
            v = 1.0 / (eps * K_safe.T @ u + 1e-12)
        
        P = np.outer(u, v) * K_safe
        return P
    
    def extract_permutation(P):
        """Extract permutation from doubly stochastic matrix via Hungarian."""
        cost_matrix = -P
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p_out = np.zeros(n, dtype=np.int64)
        p_out[row_ind] = col_ind
        return p_out
    
    def compute_cost(p):
        """Compute exact QAP cost for a permutation."""
        Bp = b[p, :]
        return int(np.sum(a * Bp[p, :]))
    
    # Run Sinkhorn annealing to refine the initial assignment
    current_P = np.zeros((n, n))
    for i in range(n):
        current_P[i, p[i]] = 1.0
    
    # Annealing schedule
    num_annealing_steps = 15
    eps_schedule = np.geomspace(0.5, 0.05, num_annealing_steps)
    
    for step, eps in enumerate(eps_schedule):
        if time.time() - start_time > time_budget * 0.4:
            break
        
        # Create kernel: mix current P with gradient information
        # Gradient of tr(P^T A P B) w.r.t. P is A P B^T + A^T P B
        grad = A @ current_P @ B + A.T @ current_P @ B.T
        
        # Use negative gradient as kernel (we want to minimize)
        # Add small noise for exploration
        noise = np.random.exponential(0.05, size=(n, n))
        K = np.maximum(-grad, 0) + noise + 1e-6
        
        # Run Sinkhorn
        num_iters = max(30, 100 - step * 5)
        new_P = sinkhorn_kernel(K, num_iters, eps)
        
        # Extract permutation and check if it improves
        new_p = extract_permutation(new_P)
        new_cost = compute_cost(new_p)
        curr_cost = compute_cost(p)
        
        if new_cost < curr_cost:
            p = new_p
            current_P = np.zeros((n, n))
            for i in range(n):
                current_P[i, p[i]] = 1.0
        else:
            # Keep current but update P for next iteration
            current_P = new_P
    
    # --- Phase 3: Fast 2-opt local search ---
    def fast_local_search(p, time_limit):
        """Efficient 2-opt local search using vectorized best-swap search."""
        p = p.copy()
        current_cost = compute_cost(p)
        
        # Precompute for fast swap evaluation
        # For each i, we can compute the effect of swapping p[i] with p[j]
        # delta(i,j) = sum_k [a[i,k]*(b[p[j],p[k]]-b[p[i],p[k]]) + a[k,i]*(b[p[k],p[j]]-b[p[k],p[i]])]
        #             + sum_k [a[j,k]*(b[p[i],p[k]]-b[p[j],p[k]]) + a[k,j]*(b[p[k],p[i]]-b[p[k],p[j]])]
        #             + a[i,j]*(b[p[j],p[i]]-b[p[i],p[j]]) + a[j,i]*(b[p[i],p[j]]-b[p[j],p[i]])
        
        # Vectorized approach: for fixed i, compute delta for all j at once
        # Let pi = p[i]
        # For each j, pj = p[j]
        # delta(i,j) involves:
        #   a[i,:] * (b[pj,:] - b[pi,:]) summed over k!=i,j
        #   a[:,i] * (b[:,pj] - b[:,pi]) summed over k!=i,j
        #   a[j,:] * (b[pi,:] - b[pj,:]) summed over k!=i,j
        #   a[:,j] * (b[:,pi] - b[:,pj]) summed over k!=i,j
        #   a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
        
        # Simplify: for fixed i, define:
        #   vec_i = a[i,:] + a[:,i]  (combined row/col of i in A)
        #   For each j: delta involves vec_i dot (b[pj,:] - b[pi,:]) adjusted for k=i,j
        #   and vec_j dot (b[pi,:] - b[pj,:]) adjusted
        #   plus the i-j direct terms
        
        improved = True
        max_passes = 20
        pass_count = 0
        
        while improved and pass_count < max_passes:
            if time.time() - start_time > time_limit:
                break
            improved = False
            pass_count += 1
            
            # For each i, find best j to swap with
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                
                pi = p[i]
                # Compute contribution of i to all potential swaps
                # a[i, k] * b[pj, k] for all k -> a[i,:] @ b[pj, :]
                # a[k, i] * b[k, pj] for all k -> b[:, pj] @ a[:, i] = a[:, i] @ b[:, pj]
                
                # For all j, compute:
                # term1[j] = a[i, :] @ b[p[j], :]  (sum_k a[i,k]*b[pj,k])
                # term2[j] = a[:, i] @ b[:, p[j]]  (sum_k a[k,i]*b[k,pj])
                
                # Vectorized: for all j at once
                # b[p, :] is Bp where Bp[j, :] = b[p[j], :]
                Bp = b[p, :]  # shape (n, n), Bp[j, k] = b[p[j], k]
                
                # term1[j] = sum_k a[i,k] * Bp[j,k] = a[i,:] @ Bp[j,:]
                # This is: Bp @ a[i, :] gives vector where element j = Bp[j,:] @ a[i,:] = sum_k b[p[j],k]*a[i,k]
                term1 = Bp @ a[i, :]  # shape (n,)
                
                # term2[j] = sum_k a[k,i] * b[k, p[j]] = sum_k a[k,i] * Bp.T[k,j] = a[:,i] @ Bp.T[:,j]
                # Bp.T[k,j] = b[p[j],k], so a[:,i] @ Bp.T[:,j] = sum_k a[k,i]*b[p[j],k]
                term2 = a[:, i] @ Bp.T  # shape (n,)
                
                # For current assignment (j=i):
                term1_i = term1[i]
                term2_i = term2[i]
                
                # For swap (i,j), the change in the "i part" is:
                # (term1[j] - term1[i]) + (term2[j] - term2[i])
                # But we also need to account for the j part and the direct i-j terms
                
                # The j part when j gets pi instead of pj:
                # a[j,:] @ (b[pi,:] - b[pj,:]) = a[j,:] @ Bp[i,:] - a[j,:] @ Bp[j,:]
                # a[:,j] @ (Bp[i,:] - Bp[j,:]).T = a[:,j] @ Bp[i,:] - a[:,j] @ Bp[j,:]
                
                # For each j:
                # j_part[j] = a[j,:] @ Bp[i,:] - a[j,:] @ Bp[j,:] + a[:,j] @ Bp[i,:] - a[:,j] @ Bp[j,:]
                
                # a[j,:] @ Bp[i,:] for all j: Bp[i,:] is a row, so this is (Bp[i,:]) @ a.T? No.
                # a[j,:] @ Bp[i,:] = sum_k a[j,k]*b[pi,k]
                # For all j: this is (a @ Bp[i,:].T) ... no, a @ Bp[i,:] doesn't work directly.
                # a[j,:] @ Bp[i,:] = (a @ Bp[i,:].reshape(n,1)).reshape(n) ... 
                # Actually: for all j, a[j,:] @ Bp[i,:] = sum_k a[j,k] * Bp[i,k]
                # This is: (a @ Bp[i, :].T).T? No.
                # It's: a @ Bp[i, :] where Bp[i,:] is shape (n,), so a @ Bp[i,:] gives shape (n,)
                # where element j = sum_k a[j,k] * Bp[i,k]. Yes!
                
                a_times_Bp_i = a @ Bp[i, :]  # shape (n,), element j = sum_k a[j,k]*b[pi,k]
                
                # a[j,:] @ Bp[j,:] for all j: this is diagonal of a @ Bp? 
                # Element j of (a @ Bp) is sum_k a[j,k]*Bp[k,:] ... no, that's a @ Bp which is (n,n)
                # We want for each j: a[j,:] @ Bp[j,:] = sum_k a[j,k]*b[p[j],k]
                # This is the diagonal of a @ Bp? (a @ Bp)[j,j] = sum_k a[j,k]*Bp[k,j] = sum_k a[j,k]*b[p[k],j]
                # That's not what we want. We want sum_k a[j,k]*b[p[j],k]
                # Which is: for each j, a[j,:] dot b[p[j],:]
                # = np.einsum('jk,jk->j', a, Bp)
                
                a_diag_Bp = np.einsum('jk,jk->j', a, Bp)  # element j = sum_k a[j,k]*b[p[j],k]
                
                # Similarly for a[:,j] @ Bp[i,:] and a[:,j] @ Bp[j,:]
                # a[:,j] @ Bp[i,:] = sum_k a[k,j]*b[pi,k]
                # For all j: (a.T @ Bp[i,:]) ... a.T @ Bp[i,:] doesn't work directly
                # a[k,j] for fixed j is column j of a. sum_k a[k,j]*Bp[i,k] = a[:,j] @ Bp[i,:]
                # For all j: this is (a.T @ Bp[i,:].T).T? 
                # a.T is (n,n) with a.T[j,k] = a[k,j]. So a.T @ Bp[i,:].T doesn't help.
                # Actually: for each j, sum_k a[k,j]*Bp[i,k] = (a.T)[j,:] @ Bp[i,:]
                # So for all j: a.T @ Bp[i,:] where the result has element j = sum_k a.T[j,k]*Bp[i,k] = sum_k a[k,j]*Bp[i,k]. Yes!
                
                aT_times_Bp_i = a.T @ Bp[i, :]  # shape (n,), element j = sum_k a[k,j]*b[pi,k]
                
                # a[:,j] @ Bp[j,:] for all j: sum_k a[k,j]*b[p[j],k]
                # = np.einsum('kj,jk->j', a, Bp)
                aT_diag_Bp = np.einsum('kj,jk->j', a, Bp)  # element j = sum_k a[k,j]*b[p[j],k]
                
                # Now j_part[j] = (a_times_Bp_i[j] - a_diag_Bp[j]) + (aT_times_Bp_i[j] - aT_diag_Bp[j])
                j_part = (a_times_Bp_i - a_diag_Bp) + (aT_times_Bp_i - aT_diag_Bp)
                
                # i_part[j] = (term1[j] - term1_i) + (term2[j] - term2_i)
                i_part = (term1 - term1_i) + (term2 - term2_i)
                
                # Direct i-j terms:
                # a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
                # = (a[i,j]-a[j,i]) * (b[pj,pi]-b[pi,pj])
                # For all j: 
                # b[pj,pi] for all j: b[p, pi] (vector of length n)
                # b[pi,pj] for all j: b[pi, p] (vector of length n)
                
                b_p_pi = b[p, pi]  # element j = b[p[j], pi]
                b_pi_p = b[pi, p]  # element j = b[pi, p[j]]
                
                direct = (a[i, :] - a[:, i]) * (b_p_pi - b_pi_p)
                direct[i] = 0  # No self-swap
                
                # Total delta[j] = i_part[j] + j_part[j] + direct[j]
                delta = i_part + j_part + direct
                delta[i] = np.inf  # Exclude self-swap
                
                # Find best j
                best_j = np.argmin(delta)
                if delta[best_j] < -1e-10:
                    p[i], p[best_j] = p[best_j], p[i]
                    current_cost += delta[best_j]
                    improved = True
        
        return p, current_cost
    
    # Run local search
    p, current_cost = fast_local_search(p, time_budget)
    
    # --- Phase 4: Multi-start with random perturbations if time remains ---
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 5.0:
        best_p = p.copy()
        best_cost = current_cost
        
        # Try a few random restarts with local search
        num_restarts = min(5, int(remaining_time / 5))
        for _ in range(num_restarts):
            if time.time() - start_time > time_budget:
                break
            
            # Random perturbation of current best
            p_try = best_p.copy()
            # Randomly swap a few pairs
            num_swaps = max(1, n // 50)
            for _ in range(num_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p_try[i], p_try[j] = p_try[j], p_try[i]
            
            # Local search from perturbed state
            p_try, cost_try = fast_local_search(p_try, time_budget)
            
            if cost_try < best_cost:
                best_cost = cost_try
                best_p = p_try
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
