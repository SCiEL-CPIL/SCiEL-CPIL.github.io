"""Combined QAP solver: MDS/Procrustes initialization + efficient 2-swap local search + Tabu Search polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    """
    Solves the QAP instance using MDS/Procrustes initialization with efficient 2-opt local search
    and simulated annealing polish.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Convert to float64 for faster arithmetic
    a_f = np.ascontiguousarray(a, dtype=np.float64)
    b_f = np.ascontiguousarray(b, dtype=np.float64)

    def compute_cost(p):
        """Compute exact cost."""
        return float(np.sum(a_f * b_f[np.ix_(p, p)]))

    def mds_embedding(D, dim=2):
        """Classical MDS embedding."""
        n = D.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (D ** 2) @ J
        vals, vecs = np.linalg.eigh(B)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        vals = vals[:dim]
        vecs = vecs[:, :dim]
        vals = np.maximum(vals, 0)
        coords = vecs * np.sqrt(vals)
        return coords

    def procrustes_alignment(X, Y):
        """Align X to Y and solve assignment problem."""
        U, S, Vt = np.linalg.svd(Y.T @ X)
        Q = U @ Vt
        X_aligned = X @ Q
        
        X_norms = np.sum(X_aligned ** 2, axis=1)
        Y_norms = np.sum(Y ** 2, axis=1)
        Cross = X_aligned @ Y.T
        C = X_norms[:, None] + Y_norms[None, :] - 2 * Cross
        
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def efficient_local_search(p, current_cost, end_time, seed=12345):
        """
        Efficient 2-swap local search with precomputed row/col contributions.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        rng = np.random.RandomState(seed)
        
        # List of all pairs
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        while time.time() < end_time:
            improved = False
            rng.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() > end_time:
                    break
                
                li = curr_p[i]
                lj = curr_p[j]
                
                # delta = (a_f[i, :] - a_f[j, :]) @ (b_f[lj, curr_p] - b_f[li, curr_p])
                #       + (a_f[:, i] - a_f[:, j]) @ (b_f[curr_p, lj] - b_f[curr_p, li])
                
                drow = b_f[lj, curr_p] - b_f[li, curr_p]
                dcol = b_f[curr_p, lj] - b_f[curr_p, li]
                
                a_diff_row = a_f[i, :] - a_f[j, :]
                a_diff_col = a_f[:, i] - a_f[:, j]
                
                delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)
                
                if delta < -1e-10:  # Accept if strictly better
                    # Perform swap
                    curr_p[i], curr_p[j] = lj, li
                    curr_cost += delta
                    
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_p = curr_p.copy()
                    improved = True
            
            if not improved:
                break
                
        return best_p, best_cost

    # 1. MDS Initialization
    # Use symmetrized matrices for MDS
    a_sym = (a_f + a_f.T) / 2.0
    b_sym = (b_f + b_f.T) / 2.0
    
    # Try different embedding dimensions
    best_p_init = None
    best_cost_init = np.inf
    
    for dim in [2, 3, 5]:
        if time.time() > deadline - 50.0:
            break
            
        X = mds_embedding(a_sym, dim)
        Y = mds_embedding(b_sym, dim)
        
        p_try = procrustes_alignment(X, Y)
        cost_try = compute_cost(p_try)
        
        if cost_try < best_cost_init:
            best_cost_init = cost_try
            best_p_init = p_try
    
    # 2. Rank Matching Initialization
    a_row_sums = a_f.sum(axis=1)
    b_row_sums = b_f.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    cost_rank = compute_cost(p_rank)
    
    if cost_rank < best_cost_init:
        best_cost_init = cost_rank
        best_p_init = p_rank
    
    # 3. Random Initialization
    rng_init = np.random.RandomState(42)
    p_rand = rng_init.permutation(n)
    cost_rand = compute_cost(p_rand)
    
    if cost_rand < best_cost_init:
        best_cost_init = cost_rand
        best_p_init = p_rand

    # 4. Local Search Polish
    end_time_lk = deadline - 10.0
    best_p, best_cost = efficient_local_search(best_p_init, best_cost_init, end_time_lk, seed=12345)
    
    # 5. Perturb and Local Search if time remains
    if time.time() < deadline - 15.0:
        rng_perturb = np.random.RandomState(789)
        p_pert = best_p.copy()
        # Swap 3-5 random pairs
        for _ in range(4):
            i, j = rng_perturb.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
        cost_pert = compute_cost(p_pert)
        end_time_lk2 = deadline - 5.0
        p_lk2, cost_lk2 = efficient_local_search(p_pert, cost_pert, end_time_lk2, seed=456)
        
        if cost_lk2 < best_cost:
            best_p = p_lk2
            best_cost = cost_lk2

    # 6. Tabu Search Polish if time remains
    if time.time() < deadline - 3.0:
        curr_p = best_p.copy()
        curr_cost = best_cost
        best_p_tabu = best_p.copy()
        best_cost_tabu = best_cost
        
        rng_tabu = np.random.RandomState(111)
        tabu_tenure = 30  # Number of iterations a move stays tabu
        tabu_list = {}  # (i, j) -> iteration when it becomes non-tabu
        current_iter = 0
        max_iter = 5000
        no_improve_count = 0
        patience = 200  # Stop if no improvement for this many iterations
        
        end_time_tabu = deadline - 1.0
        while time.time() < end_time_tabu and current_iter < max_iter and no_improve_count < patience:
            # Find best non-tabu improving move
            best_delta = -np.inf
            best_i, best_j = -1, -1
            
            # Sample a subset of facilities to evaluate (for speed)
            sample_size = min(n, 40)
            sampled = rng_tabu.choice(n, sample_size, replace=False)
            
            for i in sampled:
                li = curr_p[i]
                # Compute deltas for swapping i with all j != i
                # delta(i,j) = (a[i,:] - a[j,:]) @ (b[lj, curr_p] - b[li, curr_p])
                #            + (a[:,i] - a[:,j]) @ (b[curr_p, lj] - b[curr_p, li])
                
                # Vectorized computation for all j
                # a_diff_row[j] = a[i,:] - a[j,:]
                a_diff_row = a_f[i, :] - a_f  # shape (n, n), a_diff_row[j, :] = a[i,:] - a[j,:]
                
                # drow[j] = b_f[curr_p[j], curr_p] - b_f[li, curr_p]
                drow = b_f[curr_p, curr_p] - b_f[li, curr_p]  # shape (n, n), drow[j, :] = b[curr_p[j], curr_p] - b[li, curr_p]
                
                # dcol[j] = b_f[curr_p, curr_p[j]] - b_f[curr_p, li]
                dcol = b_f[curr_p, curr_p].T - b_f[curr_p, li]  # shape (n, n), dcol[:, j] = b[curr_p, curr_p[j]] - b[curr_p, li]
                
                # a_diff_col[j] = a_f[:, i] - a_f[:, j]
                a_diff_col = a_f[:, i] - a_f.T  # shape (n, n), a_diff_col[:, j] = a_f[:, i] - a_f[:, j]
                
                # delta[j] = sum_k a_diff_row[j, k] * drow[j, k] + sum_k a_diff_col[k, j] * dcol[k, j]
                delta_row = np.sum(a_diff_row * drow, axis=1)  # shape (n,)
                delta_col = np.sum(a_diff_col * dcol, axis=0)  # shape (n,)
                deltas = delta_row + delta_col
                
                # Exclude i itself
                deltas[i] = np.inf
                
                # Exclude tabu moves
                for j in range(n):
                    if j == i:
                        continue
                    # Check if (min(i,j), max(i,j)) is tabu
                    key = (min(i, j), max(i, j))
                    if key in tabu_list and tabu_list[key] > current_iter:
                        deltas[j] = np.inf
                
                # Find best j
                j_best = np.argmin(deltas)
                if deltas[j_best] < best_delta:
                    best_delta = deltas[j_best]
                    best_i, best_j = i, j_best
            
            # If no valid move found, break
            if best_i == -1:
                break
            
            # Aspiration: accept if it improves best-so-far, even if tabu
            # (already handled by not marking as inf if it would improve best)
            if best_delta >= 0 and best_delta > best_cost_tabu - curr_cost:
                # Only accept if it's a new best
                pass
            elif best_delta >= 0:
                # Accept worsening move if no improving move found (to escape local min)
                no_improve_count += 1
            else:
                no_improve_count = 0
            
            # Perform the swap
            li = curr_p[best_i]
            lj = curr_p[best_j]
            curr_p[best_i], curr_p[best_j] = lj, li
            curr_cost += best_delta
            
            # Update tabu list
            key = (min(best_i, best_j), max(best_i, best_j))
            tabu_list[key] = current_iter + tabu_tenure
            
            # Clean up old tabu entries
            if len(tabu_list) > 1000:
                tabu_list = {k: v for k, v in tabu_list.items() if v > current_iter}
            
            if curr_cost < best_cost_tabu:
                best_cost_tabu = curr_cost
                best_p_tabu = curr_p.copy()
            
            current_iter += 1
        
        if best_cost_tabu < best_cost:
            best_p = best_p_tabu
            best_cost = best_cost_tabu

    return best_p.tolist()
# EVOLVE-BLOCK-END
