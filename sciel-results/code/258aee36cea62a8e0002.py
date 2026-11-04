"""QAP solver: fast local search with incremental precomputation + Hungarian refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Rank-matching + Hungarian initial permutation.
    2. Fast local search with incremental precomputed dot products.
    3. Multiple Hungarian linearization refinement rounds.
    4. Final systematic neighborhood polish.
    """
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s limit
    np.random.seed(42)
    
    # Precompute row/col dot products for current permutation
    def init_precomp(p):
        """Initialize precomputed arrays for permutation p."""
        dot_row = np.empty((n, n), dtype=np.float64)
        dot_col = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            loc = p[i]
            dot_row[i] = a[i] @ b[loc, :]
            dot_col[i] = a[:, i] @ b[:, loc]
        return dot_row, dot_col
    
    def update_precomp_swap(i, j, p, dot_row, dot_col):
        """Update precomputed arrays after swapping positions i and j in p."""
        u = p[i]
        v = p[j]
        if u == v:
            return
        # For each facility k, its row dot product changes if p[k] is u or v
        # dot_row[k] = a[k] @ b[p[k], :]
        # After swap: facilities at positions i,j swap locations u,v
        # Only facilities k where p[k] in {u, v} are affected, which are i and j
        # dot_row[i] was a[i] @ b[u, :], becomes a[i] @ b[v, :]
        # dot_row[j] was a[j] @ b[v, :], becomes a[j] @ b[u, :]
        dot_row[i] = a[i] @ b[v, :]
        dot_row[j] = a[j] @ b[u, :]
        # dot_col[k] = a[:, k] @ b[:, p[k]]
        # dot_col[i] was a[:, i] @ b[:, u], becomes a[:, i] @ b[:, v]
        # dot_col[j] was a[:, j] @ b[:, v], becomes a[:, j] @ b[:, u]
        dot_col[i] = a[:, i] @ b[:, v]
        dot_col[j] = a[:, j] @ b[:, u]
    
    def swap_delta_fast(i, j, p, dot_row, dot_col):
        """Compute cost change for swapping facilities i and j using precomputed dots."""
        u = p[i]
        v = p[j]
        if u == v:
            return 0.0
        
        # Using precomputed dot products:
        # Current contribution involving i and j:
        #   a[i,i]*b[u,u] + a[i,j]*b[u,v] + a[j,i]*b[v,u] + a[j,j]*b[v,v]
        #   + sum_{k!=i,j} a[i,k]*b[u,p[k]] + sum_{k!=i,j} a[k,i]*b[p[k],u]
        #   + sum_{k!=i,j} a[j,k]*b[v,p[k]] + sum_{k!=i,j} a[k,j]*b[p[k],v]
        # 
        # After swap (i gets v, j gets u):
        #   a[i,i]*b[v,v] + a[i,j]*b[v,u] + a[j,i]*b[u,v] + a[j,j]*b[u,u]
        #   + sum_{k!=i,j} a[i,k]*b[v,p[k]] + sum_{k!=i,j} a[k,i]*b[p[k],v]
        #   + sum_{k!=i,j} a[j,k]*b[u,p[k]] + sum_{k!=i,j} a[k,j]*b[p[k],u]
        #
        # delta = new - old
        # Corner terms:
        delta = (a[i,i]*(b[v,v]-b[u,u]) + a[i,j]*(b[v,u]-b[u,v]) +
                 a[j,i]*(b[u,v]-b[v,u]) + a[j,j]*(b[u,u]-b[v,v]))
        
        # Row i terms: sum_{k!=i,j} a[i,k]*(b[v,p[k]] - b[u,p[k]])
        # = sum_k a[i,k]*(b[v,p[k]] - b[u,p[k]]) - a[i,i]*(b[v,u]-b[u,u]) - a[i,j]*(b[v,v]-b[u,v])
        # Note: sum_k a[i,k]*b[v,p[k]] = dot_row[i] with p[i] replaced... 
        # Actually dot_row[i] = a[i] @ b[u,:] currently. We need a[i] @ b[v,p] for all k.
        # That's a[i] @ (b[v,p[0]], b[v,p[1]], ...) which isn't directly precomputed.
        # 
        # Let me use the difference approach:
        # br_diff[k] = b[v, p[k]] - b[u, p[k]]
        # We need sum_k a[i,k] * br_diff[k]
        # = a[i] @ (b[v,p] - b[u,p])
        # = a[i] @ b[v,p] - a[i] @ b[u,p]
        # = a[i] @ b[v,p] - dot_row[i]
        # But a[i] @ b[v,p] = sum_k a[i,k]*b[v,p[k]], which we don't have precomputed.
        
        # Alternative: use the formula from the original but with precomputed dots where possible
        # Actually the original 4-dot approach is already O(n). Let me keep it but optimize.
        
        br_diff = b[v] - b[u]  # shape (n,)
        bc_diff = b[:, v] - b[:, u]  # shape (n,)
        
        s_ri = np.dot(a[i], br_diff[p])
        delta += s_ri - a[i, i] * br_diff[u] - a[i, j] * br_diff[v]
        
        s_rj = np.dot(a[j], br_diff[p])
        delta -= s_rj - a[j, i] * br_diff[u] - a[j, j] * br_diff[v]
        
        s_ci = np.dot(a[:, i], bc_diff[p])
        delta += s_ci - a[i, i] * bc_diff[u] - a[j, i] * bc_diff[v]
        
        s_cj = np.dot(a[:, j], bc_diff[p])
        delta -= s_cj - a[i, j] * bc_diff[u] - a[j, j] * bc_diff[v]
        
        return delta
    
    def compute_cost(p):
        """Compute exact QAP cost for permutation p."""
        total = 0.0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total
    
    def local_search(p, cost, time_limit, batch_size=400):
        """Hill climbing with batch-evaluated 2-swap moves."""
        dot_row, dot_col = init_precomp(p)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for _ in range(batch_size):
                if time.time() > time_limit:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                d = swap_delta_fast(i, j, p, dot_row, dot_col)
                if d < -1e-9:
                    p[i], p[j] = p[j], p[i]
                    cost += d
                    update_precomp_swap(i, j, p, dot_row, dot_col)
                    improved = True
        return p, cost
    
    def systematic_polish(p, cost, time_limit):
        """Systematic 1-sweep neighborhood search: try all swaps for each i."""
        dot_row, dot_col = init_precomp(p)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n):
                if time.time() > time_limit:
                    return p, cost
                best_d = 0.0
                best_j = -1
                for j in range(i + 1, n):
                    d = swap_delta_fast(i, j, p, dot_row, dot_col)
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j >= 0 and best_d < -1e-9:
                    p[i], p[best_j] = p[best_j], p[i]
                    cost += best_d
                    update_precomp_swap(i, best_j, p, dot_row, dot_col)
                    improved = True
        return p, cost
    
    def best_first_local_search(p, cost, time_limit, num_candidates=300):
        """Evaluate many candidates, apply the best improving move."""
        dot_row, dot_col = init_precomp(p)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            candidates = []
            for _ in range(num_candidates):
                if time.time() > time_limit:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                d = swap_delta_fast(i, j, p, dot_row, dot_col)
                if d < -1e-9:
                    candidates.append((d, i, j))
            
            if not candidates:
                break
            
            candidates.sort(key=lambda x: x[0])
            d, i, j = candidates[0]
            p[i], p[j] = p[j], p[i]
            cost += d
            update_precomp_swap(i, j, p, dot_row, dot_col)
            improved = True
        return p, cost
    
    # --- Initial solution: rank-matching + Hungarian ---
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Refine with Hungarian
    b_col_p = b[:, p]
    b_row_p = b[p, :]
    C = a @ b_col_p.T + a.T @ b_row_p
    row_ind, col_ind = linear_sum_assignment(C)
    p_hung = np.empty(n, dtype=np.int64)
    p_hung[row_ind] = col_ind
    
    cost_hung = compute_cost(p_hung)
    cost_rank = compute_cost(p)
    
    best_p = p_hung.copy() if cost_hung < cost_rank else p.copy()
    best_cost = min(cost_hung, cost_rank)
    
    # Phase 1: Initial local search
    phase1_deadline = deadline - 45.0
    best_p, best_cost = local_search(best_p, best_cost, phase1_deadline, batch_size=400)
    
    # Phase 2: Hungarian refinement (2 rounds)
    for round_num in range(2):
        if time.time() > deadline - 25.0:
            break
        
        b_col_p = b[:, best_p]
        b_row_p = b[best_p, :]
        C = a @ b_col_p.T + a.T @ b_row_p
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
        
        if time.time() < deadline - 20.0:
            best_p, best_cost = local_search(best_p, best_cost, deadline - 20.0, batch_size=400)
    
    # Phase 3: Systematic polish (exhaustive 1-sweep neighborhood)
    if time.time() < deadline - 5.0:
        best_p, best_cost = systematic_polish(best_p, best_cost, deadline - 3.0)
    
    # Phase 4: Final best-first polish
    if time.time() < deadline - 2.0:
        best_p, best_cost = best_first_local_search(best_p, best_cost, deadline - 1.0, num_candidates=200)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
