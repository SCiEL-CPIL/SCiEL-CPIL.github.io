# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def calc_cost_full(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    def two_opt_delta(p, i, j):
        """Compute delta cost for swapping facilities i and j in permutation p."""
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0
        d = b[lj, p] - b[li, p]
        d[i] = 0
        d[j] = 0
        a_diff = a[i, :] - a[j, :] + a[:, i] - a[:, j]
        a_diff[i] = 0
        a_diff[j] = 0
        return int(np.dot(a_diff, d))

    def linearized_cost_matrix(p):
        """
        Compute the linearized cost matrix for Hungarian algorithm.
        cost[i,l] = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
        """
        C1 = b[:, p]  # C1[l, j] = b[l, p[j]]
        C2 = b[p, :]  # C2[j, l] = b[p[j], l]
        term1 = a @ C1.T  # term1[i,l] = sum_j a[i,j] * b[l, p[j]]
        term2 = a.T @ C2  # term2[i,l] = sum_j a[j,i] * b[p[j], l]
        return term1 + term2

    def two_opt_polish(p, max_time=None):
        """Greedy 2-opt local search. Returns (p, cost)."""
        cost = calc_cost_full(p)
        if max_time is None:
            max_time = deadline
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n):
                if time.time() > max_time:
                    break
                for j in range(i + 1, n):
                    if time.time() > max_time:
                        break
                    delta = two_opt_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    # --- Phase 1: Initial Solution via Rank Matching ---
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy_order = np.argsort(-a_row_sums, kind='stable')
    central_order = np.argsort(b_row_sums, kind='stable')
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    p = p.copy()

    # Initial polish
    p, current_cost = two_opt_polish(p, max_time=start_time + 15.0)
    best_p = p.copy()
    best_cost = current_cost

    # --- Phase 2: Frank-Wolfe on Birkhoff Polytope + Hungarian + Polish ---
    # X is a doubly stochastic matrix (relaxed assignment)
    # Start from the current permutation as a one-hot matrix
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p] = 1.0

    # Precompute A_sym and B_sym for the QAP relaxation
    # Q(X) = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]
    # = sum_{i,k} X[i,k] * (sum_j a[i,j] * sum_l b[k,l] X[j,l])
    # Let M = X^T @ B, then M[j,l] = sum_k X[k,j] b[k,l]
    # Then Q(X) = sum_{i,k} X[i,k] * (A @ M)[i,k]  ... wait let me re-derive
    
    # Q(X) = sum_{i,j} a[i,j] * sum_{k,l} b[k,l] X[i,k] X[j,l]
    # = sum_{i,k} X[i,k] * sum_j a[i,j] * sum_l b[k,l] X[j,l]
    # Let S = B @ X, so S[k,l] = sum_j b[k,j] X[j,l]  -- no that's not right
    # sum_l b[k,l] X[j,l] = (B @ X^T)[k,j]
    # So Q(X) = sum_{i,k} X[i,k] * sum_j a[i,j] * (B @ X^T)[k,j]
    # = sum_{i,k} X[i,k] * (A @ (B @ X^T)^T)[i,k]  ... 
    # (B @ X^T)^T = X @ B^T
    # So Q(X) = sum_{i,k} X[i,k] * (A @ X @ B^T)[i,k]
    # = trace((A @ X @ B^T)^T @ X) = trace(B @ X^T @ A^T @ X)
    
    # For Frank-Wolfe, we need the gradient:
    # grad[i,k] = 2 * (A @ X @ B^T)[i,k]  (assuming symmetric treatment)
    # Actually for Q(X) = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l]:
    # dQ/dX[i,k] = sum_{j,l} a[i,j] b[k,l] X[j,l] + sum_{j,l} a[j,i] b[l,k] X[j,l]
    #            = (A @ X @ B^T)[i,k] + (A^T @ X @ B)[i,k]
    
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    
    def frank_wolfe_step(X, max_iters=50, tol=1e-6):
        """Perform Frank-Wolfe iterations on the Birkhoff polytope."""
        for it in range(max_iters):
            if time.time() > deadline - 15.0:
                break
            # Compute gradient
            G1 = A @ X @ B.T
            G2 = A.T @ X @ B
            G = G1 + G2
            
            # Linear minimization over Birkhoff polytope: solve min sum G[i,k] X[i,k]
            row_ind, col_ind = linear_sum_assignment(G)
            S = np.zeros((n, n), dtype=np.float64)
            S[row_ind, col_ind] = 1.0
            
            # Check for convergence
            gap = np.sum(G * (X - S))
            if abs(gap) < tol:
                break
            
            # Line search
            # Try several step sizes
            best_step = 0.0
            best_val = np.inf
            for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                X_new = (1.0 - t) * X + t * S
                # Compute Q(X_new)
                Q1 = A @ X_new @ B.T
                Q2 = A.T @ X_new @ B
                val = np.sum(X_new * (Q1 + Q2))
                if val < best_val:
                    best_val = val
                    best_step = t
            
            X = (1.0 - best_step) * X + best_step * S
            
        return X

    # Run Frank-Wolfe from the initial permutation
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p] = 1.0
    
    fw_start = time.time()
    X = frank_wolfe_step(X, max_iters=100, tol=1e-4)
    fw_elapsed = time.time() - fw_start

    # Extract permutation from X using Hungarian on the relaxed solution
    # We want to find the permutation closest to X in some sense
    # Use the negative of X as cost to maximize overlap
    row_ind, col_ind = linear_sum_assignment(-X)
    p_fw = np.empty(n, dtype=np.int64)
    p_fw[row_ind] = col_ind
    
    fw_cost = calc_cost_full(p_fw)
    
    if fw_cost < best_cost:
        best_p = p_fw.copy()
        best_cost = fw_cost
    
    # Polish the Frank-Wolfe result
    p_fw_polished = p_fw.copy()
    p_fw_polished, fw_polished_cost = two_opt_polish(p_fw_polished, max_time=min(deadline - 10.0, time.time() + 10.0))
    if fw_polished_cost < best_cost:
        best_p = p_fw_polished.copy()
        best_cost = fw_polished_cost

    # --- Phase 3: Hungarian Reassignment Loop (combined with FW insights) ---
    p = best_p.copy()
    hungarian_count = 0
    no_improve_rounds = 0
    max_no_improve = 3

    while time.time() < deadline - 20.0:
        # Compute linearized cost and solve assignment
        cost_mat = linearized_cost_matrix(p)
        
        # Add small random perturbation to break ties and escape plateaus
        if hungarian_count > 0 and no_improve_rounds > 0:
            noise_scale = 1.0 + no_improve_rounds * 0.5
            cost_mat = cost_mat + np.random.randn(n, n) * noise_scale
        
        try:
            row_ind, col_ind = linear_sum_assignment(cost_mat)
        except Exception:
            break
            
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        new_cost = calc_cost_full(new_p)
        
        if new_cost < best_cost:
            p = new_p
            best_p = p.copy()
            best_cost = new_cost
            hungarian_count += 1
            no_improve_rounds = 0
            
            # Polish after a good Hungarian step
            p, p_cost = two_opt_polish(p, max_time=min(deadline - 10.0, time.time() + 8.0))
            if p_cost < best_cost:
                best_p = p.copy()
                best_cost = p_cost
        else:
            no_improve_rounds += 1
            
            # If stuck, try a perturbed restart from best
            if no_improve_rounds >= max_no_improve:
                p = best_p.copy()
                # Randomly swap a few pairs to escape local minimum
                num_swaps = np.random.randint(5, 15)
                for _ in range(num_swaps):
                    i = np.random.randint(n)
                    j = np.random.randint(n)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                p, p_cost = two_opt_polish(p, max_time=min(deadline - 10.0, time.time() + 5.0))
                if p_cost < best_cost:
                    best_p = p.copy()
                    best_cost = p_cost
                no_improve_rounds = 0
                hungarian_count = 0

    # --- Phase 4: Final Polish ---
    p_polish = best_p.copy()
    p_polish, final_cost = two_opt_polish(p_polish, max_time=deadline - 2.0)
    if final_cost < best_cost:
        best_p = p_polish
        best_cost = final_cost

    return best_p.tolist()

# EVOLVE-BLOCK-END
