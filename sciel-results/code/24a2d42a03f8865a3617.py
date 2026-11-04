import time
import numpy as np
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    """
    Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Rank-matching + Hungarian initial permutation.
    2. Fast local search with O(n) swap delta and best-of-batch move selection.
    3. Multiple Hungarian linearization refinement rounds to escape local minima.
    4. 3-cycle moves for additional escape from local minima.
    5. Final high-precision polish.
    """
    n = len(a)
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    # Precompute useful arrays
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    def swap_delta_fast(i, j, p):
        """Compute cost change for swapping facilities i and j in permutation p."""
        u = p[i]
        v = p[j]
        if u == v:
            return 0.0
        
        # Extract rows/cols of b for u and v
        b_u_row = b[u, :]
        b_v_row = b[v, :]
        b_u_col = b[:, u]
        b_v_col = b[:, v]
        
        # Diagonal terms
        buu = b[u, u]
        bvv = b[v, v]
        buv = b[u, v]
        bvu = b[v, u]
        
        # Dot products with current permutation p
        a_i = a[i]
        a_j = a[j]
        a_ci = a[:, i]
        a_cj = a[:, j]
        
        # S1: sum_k a[i,k] * b[v, p[k]]
        S1 = np.dot(a_i, b_v_row[p])
        # S2: sum_k a[i,k] * b[u, p[k]]
        S2 = np.dot(a_i, b_u_row[p])
        # S3: sum_k a[j,k] * b[u, p[k]]
        S3 = np.dot(a_j, b_u_row[p])
        # S4: sum_k a[j,k] * b[v, p[k]]
        S4 = np.dot(a_j, b_v_row[p])
        
        # S5: sum_k a[k,i] * b[p[k], v]
        S5 = np.dot(a_ci, b_v_col[p])
        # S6: sum_k a[k,i] * b[p[k], u]
        S6 = np.dot(a_ci, b_u_col[p])
        # S7: sum_k a[k,j] * b[p[k], u]
        S7 = np.dot(a_cj, b_u_col[p])
        # S8: sum_k a[k,j] * b[p[k], v]
        S8 = np.dot(a_cj, b_v_col[p])
        
        # Total change from row/col swaps
        delta = (S1 - S2) + (S3 - S4) + (S5 - S6) + (S7 - S8)
        
        # Corrections for k=i and k=j
        corr_i = a[i,i]*(bvu - buu) + a[j,i]*(buu - bvu) + a[i,i]*(buv - buu) + a[j,i]*(buu - buv)
        corr_j = a[i,j]*(bvv - buv) + a[j,j]*(buv - bvv) + a[i,j]*(bvv - bvu) + a[j,j]*(bvu - bvv)
        
        delta -= corr_i
        delta -= corr_j
        
        # Direct interaction terms
        delta += a[i,i]*(bvv - buu)
        delta += a[j,j]*(buu - bvv)
        delta += a[i,j]*(bvu - buv)
        delta += a[j,i]*(buv - bvu)
        
        return delta

    def cycle3_delta(i, j, k, p):
        """Compute cost change for cycling facilities i->j->k->i (i gets j's loc, j gets k's, k gets i's)."""
        u, v, w = p[i], p[j], p[k]
        
        # Old contribution from the 3x3 block
        old_ijk = (a[i,i]*b[u,u] + a[i,j]*b[u,v] + a[i,k]*b[u,w] +
                   a[j,i]*b[v,u] + a[j,j]*b[v,v] + a[j,k]*b[v,w] +
                   a[k,i]*b[w,u] + a[k,j]*b[w,v] + a[k,k]*b[w,w])
        
        # New contribution from the 3x3 block
        new_ijk = (a[i,i]*b[v,v] + a[i,j]*b[v,w] + a[i,k]*b[v,u] +
                   a[j,i]*b[w,v] + a[j,j]*b[w,w] + a[j,k]*b[w,u] +
                   a[k,i]*b[u,w] + a[k,j]*b[u,v] + a[k,k]*b[u,u])
        
        # Terms with other facilities
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        mask[k] = False
        
        pm = p[mask]
        am_i = a[i, mask]
        am_j = a[j, mask]
        am_k = a[k, mask]
        am_ci = a[mask, i]
        am_cj = a[mask, j]
        am_ck = a[mask, k]
        
        # Row contributions: facility i now at v, j at w, k at u
        row_delta = (np.dot(am_i, b[v, pm]) - np.dot(am_i, b[u, pm]) +
                    np.dot(am_j, b[w, pm]) - np.dot(am_j, b[v, pm]) +
                    np.dot(am_k, b[u, pm]) - np.dot(am_k, b[w, pm]))
        
        # Col contributions: location v now has i, w has j, u has k
        col_delta = (np.dot(am_ci, b[pm, v]) - np.dot(am_ci, b[pm, u]) +
                    np.dot(am_cj, b[pm, w]) - np.dot(am_cj, b[pm, v]) +
                    np.dot(am_ck, b[pm, u]) - np.dot(am_ck, b[pm, w]))
        
        return (new_ijk - old_ijk) + row_delta + col_delta

    def compute_cost(p):
        """Compute exact QAP cost for permutation p."""
        total = 0.0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total

    def local_search_2swap(p, cost, time_limit, batch_size=200):
        """Hill climbing with best-of-batch 2-swap move selection."""
        p = p.copy()
        improved = True
        no_improve_count = 0
        max_no_improve = 3000
        
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0.0
            best_i, best_j = -1, -1
            
            for _ in range(batch_size):
                if time.time() > time_limit:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                d = swap_delta_fast(i, j, p)
                if d < best_delta:
                    best_delta = d
                    best_i, best_j = i, j
            
            if best_i != -1 and best_delta < -1e-9:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += best_delta
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break

        return p, cost

    def local_search_3cycle(p, cost, time_limit, batch_size=100):
        """Hill climbing with best-of-batch 3-cycle move selection."""
        p = p.copy()
        improved = True
        no_improve_count = 0
        max_no_improve = 1500
        
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0.0
            best_triplet = None
            
            for _ in range(batch_size):
                if time.time() > time_limit:
                    break
                i, j, k = np.random.choice(n, 3, replace=False)
                d = cycle3_delta(i, j, k, p)
                if d < best_delta:
                    best_delta = d
                    best_triplet = (i, j, k)
            
            if best_triplet is not None and best_delta < -1e-9:
                i, j, k = best_triplet
                p[i], p[j], p[k] = p[j], p[k], p[i]
                cost += best_delta
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break

        return p, cost

    # --- Initial solution: rank-matching + Hungarian ---
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
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
    phase1_deadline = deadline - 55.0
    best_p, best_cost = local_search_2swap(best_p, best_cost, phase1_deadline, batch_size=150)
    
    # Phase 2: Hungarian refinement rounds with alternating 2-swap and 3-cycle local search
    num_rounds = 5
    for round_num in range(num_rounds):
        if time.time() > deadline - 20.0:
            break
        
        # Build linear cost matrix based on current best_p
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
        
        # Local search after Hungarian: alternate between 2-swap and 3-cycle
        if time.time() < deadline - 15.0:
            if round_num % 2 == 0:
                best_p, best_cost = local_search_2swap(best_p, best_cost, deadline - 15.0, batch_size=150)
            else:
                best_p, best_cost = local_search_3cycle(best_p, best_cost, deadline - 15.0, batch_size=80)
        
        # Always try both types in later rounds
        if time.time() < deadline - 10.0 and round_num >= 2:
            other_func = local_search_3cycle if round_num % 2 == 0 else local_search_2swap
            other_bs = 80 if round_num % 2 == 0 else 150
            best_p, best_cost = other_func(best_p, best_cost, deadline - 10.0, batch_size=other_bs)
            
    # Phase 3: Final polish with both move types
    if time.time() < deadline - 3.0:
        best_p, best_cost = local_search_2swap(best_p, best_cost, deadline - 1.5, batch_size=100)
        if time.time() < deadline - 0.5:
            best_p, best_cost = local_search_3cycle(best_p, best_cost, deadline - 0.1, batch_size=50)

    return best_p.tolist()
