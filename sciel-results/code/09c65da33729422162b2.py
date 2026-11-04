"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Convert to float64 for faster arithmetic
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def compute_cost_fast(p):
        """Compute cost using vectorized matrix multiplication."""
        bp = b_f[np.ix_(p, p)]
        return int(np.sum(a_f * bp))

    def efficient_local_search(p, current_cost, end_time, seed=12345):
        """
        Efficient 2-swap local search with precomputed row/col contributions.
        Uses the delta formula:
        delta = (a[i,:] - a[j,:]) @ (b[old_j, p] - b[old_i, p]) 
              + (a[:,i] - a[:,j]) @ (b[p, old_j] - b[p, old_i])
        where p is the current permutation (facility -> location).
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
                
                # drow = b_f[lj, curr_p] - b_f[li, curr_p]
                # dcol = b_f[curr_p, lj] - b_f[curr_p, li]
                
                drow = b_f[lj, curr_p] - b_f[li, curr_p]
                dcol = b_f[curr_p, lj] - b_f[curr_p, li]
                
                # delta = a_f[i, :] @ drow + a_f[:, i] @ dcol + a_f[j, :] @ (-drow) + a_f[:, j] @ (-dcol)
                #       = (a_f[i, :] - a_f[j, :]) @ drow + (a_f[:, i] - a_f[:, j]) @ dcol
                
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

    def simulated_annealing(p, best_cost, end_time, seed=42):
        """
        Perform Simulated Annealing to escape local minima.
        """
        curr_p = p.copy()
        curr_cost = best_cost
        best_p = p.copy()
        best_cost_sa = best_cost
        
        rng = np.random.RandomState(seed)
        
        # Estimate initial temperature
        deltas = []
        for _ in range(20):
            i, j = rng.choice(n, 2, replace=False)
            li = curr_p[i]
            lj = curr_p[j]
            drow = b_f[lj, curr_p] - b_f[li, curr_p]
            dcol = b_f[curr_p, lj] - b_f[curr_p, li]
            a_diff_row = a_f[i, :] - a_f[j, :]
            a_diff_col = a_f[:, i] - a_f[:, j]
            d = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)
            deltas.append(abs(d))
        avg_delta = np.mean(deltas) if deltas else 1.0
        if avg_delta == 0:
            avg_delta = 1.0
            
        T = avg_delta * 15.0
        cooling_rate = 0.9995
        min_T = 1e-5
        
        while time.time() < end_time:
            if T < min_T:
                break
                
            # Generate random move
            i, j = rng.choice(n, 2, replace=False)
            
            li = curr_p[i]
            lj = curr_p[j]
            
            drow = b_f[lj, curr_p] - b_f[li, curr_p]
            dcol = b_f[curr_p, lj] - b_f[curr_p, li]
            a_diff_row = a_f[i, :] - a_f[j, :]
            a_diff_col = a_f[:, i] - a_f[:, j]
            delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)
            
            # Acceptance criterion
            if delta < 0:
                accept = True
            else:
                # Avoid overflow in exp
                if T > 0:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                accept = rng.random_sample() < prob
            
            if accept:
                curr_p[i], curr_p[j] = lj, li
                curr_cost += delta
                
                if curr_cost < best_cost_sa:
                    best_cost_sa = curr_cost
                    best_p = curr_p.copy()
            
            T *= cooling_rate
        
        return best_p, best_cost_sa

    # 1. MDS/Procrustes Initialization
    def mds_embedding(D, dim=2):
        """Classical MDS embedding."""
        nn = D.shape[0]
        J = np.eye(nn) - np.ones((nn, nn)) / nn
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

    # Symmetrize for MDS
    a_sym = (a_f + a_f.T) / 2.0
    b_sym = (b_f + b_f.T) / 2.0

    best_p_init = None
    best_cost_init = np.inf

    # Try different embedding dimensions
    for dim in [2, 3, 5]:
        if time.time() > deadline - 50.0:
            break
            
        try:
            X = mds_embedding(a_sym, dim)
            Y = mds_embedding(b_sym, dim)
            
            p_try = procrustes_alignment(X, Y)
            cost_try = compute_cost_fast(p_try)
            
            if cost_try < best_cost_init:
                best_cost_init = cost_try
                best_p_init = p_try
        except:
            pass

    # 2. Spectral Initialization
    def spectral_init():
        """Use spectral decomposition to get an initial assignment."""
        try:
            w_a, v_a = np.linalg.eigh(a_sym)
            w_b, v_b = np.linalg.eigh(b_sym)
            # Use the top few eigenvectors
            k = min(10, n)
            Va = v_a[:, :k]
            Vb = v_b[:, :k]
            # Assign based on correlation
            C_spec = Va @ Vb.T
            row_ind, col_ind = linear_sum_assignment(-C_spec)
            p_spec = np.empty(n, dtype=np.int64)
            p_spec[row_ind] = col_ind
            return p_spec
        except:
            return None

    p_spec = spectral_init()
    if p_spec is not None:
        cost_spec = compute_cost_fast(p_spec)
        if cost_spec < best_cost_init:
            best_cost_init = cost_spec
            best_p_init = p_spec

    # 3. Hungarian on the linearized cost
    C = a_f @ b_f.T
    row_ind, col_ind = linear_sum_assignment(-C)
    p_hung = np.empty(n, dtype=np.int64)
    p_hung[row_ind] = col_ind
    cost_hung = compute_cost_fast(p_hung)
    
    if cost_hung < best_cost_init:
        best_cost_init = cost_hung
        best_p_init = p_hung

    # 4. Rank Matching Initialization
    a_row_sums = a_f.sum(axis=1)
    b_row_sums = b_f.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    cost_rank = compute_cost_fast(p_rank)
    
    if cost_rank < best_cost_init:
        best_cost_init = cost_rank
        best_p_init = p_rank

    # 5. Random Initialization
    rng_init = np.random.RandomState(42)
    p_rand = rng_init.permutation(n)
    cost_rand = compute_cost_fast(p_rand)
    
    if cost_rand < best_cost_init:
        best_cost_init = cost_rand
        best_p_init = p_rand

    if best_p_init is None:
        # Fallback
        best_p_init = np.arange(n)
        best_cost_init = compute_cost_fast(best_p_init)

    best_p = best_p_init.copy()
    best_cost = best_cost_init

    # 6. Polish Best Initial Solution with Local Search
    end_time_lk = deadline - 20.0
    p_lk, cost_lk = efficient_local_search(best_p, best_cost, end_time_lk)
    if cost_lk < best_cost:
        best_p = p_lk
        best_cost = cost_lk
        
    # 7. Try Random Initialization + Local Search
    if time.time() < deadline - 15.0:
        rng_init2 = np.random.RandomState(123)
        p_rand2 = rng_init2.permutation(n)
        cost_rand2 = compute_cost_fast(p_rand2)
        
        end_time_lk2 = deadline - 10.0
        p_lk2, cost_lk2 = efficient_local_search(p_rand2, cost_rand2, end_time_lk2, seed=999)
        if cost_lk2 < best_cost:
            best_p = p_lk2
            best_cost = cost_lk2

    # 8. Perturb Best and Local Search
    if time.time() < deadline - 5.0:
        rng_perturb = np.random.RandomState(456)
        p_pert = best_p.copy()
        # Swap 5 random pairs to perturb
        for _ in range(5):
            i, j = rng_perturb.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
        cost_pert = compute_cost_fast(p_pert)
        end_time_lk3 = deadline - 2.0
        p_lk3, cost_lk3 = efficient_local_search(p_pert, cost_pert, end_time_lk3, seed=789)
        if cost_lk3 < best_cost:
            best_p = p_lk3
            best_cost = cost_lk3

    # 9. Final SA Polish if time remains
    if time.time() < deadline - 1.0:
        p_sa, cost_sa = simulated_annealing(best_p, best_cost, deadline - 0.5, seed=111)
        if cost_sa < best_cost:
            best_p = p_sa
            best_cost = cost_sa

    return best_p.tolist()
# EVOLVE-BLOCK-END
