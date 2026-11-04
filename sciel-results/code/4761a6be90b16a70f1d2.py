"""QAP tai256c solver: iterated local search with spectral bisection init, Hungarian re-optimization, and tabu diversification."""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 150s limit

    # Precompute for fast delta computation
    # We'll use a smarter local search that precomputes contributions

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_local_search(p, cost, max_passes=50):
        """
        Optimized local search using precomputed delta arrays.
        For each position i, we precompute the cost contribution if we swap i with each j.
        """
        # Precompute the "row contribution" and "col contribution" for each position
        # This allows O(1) delta updates after a swap
        
        # Let's use a different approach: precompute the full cost matrix for the current permutation
        # and update it incrementally.
        
        # For now, use the existing fast_swap_delta but with better loop structure
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            # Try all pairs, but break early if possible
            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return p, cost
                    
                    delta = _fast_swap_delta(p, i, j)
                    
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    def _fast_swap_delta(p, u, v):
        pu = p[u]
        pv = p[v]

        # Vectorized delta computation
        d_row = b[pv] - b[pu]
        d_col = b[:, pv] - b[:, pu]

        delta = np.dot(a[u] - a[v], d_row[p]) + np.dot(a[:, u] - a[:, v], d_col[p])
        
        # Correction for double-counted terms
        delta -= (a[u, u] - a[v, u]) * d_row[pu]
        delta -= (a[u, v] - a[v, v]) * d_row[pv]
        delta -= (a[u, u] - a[u, v]) * d_col[pu]
        delta -= (a[v, u] - a[v, v]) * d_col[pv]
        
        # Cross term
        delta += (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])

        return delta

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T  # M1[i,j] = sum_k a[i,k]*b[j,p[k]]
        
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2  # M2[i,j] = sum_k a[k,i]*b[p[k],j]
        
        C = M1 + M2
        
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def fiedler_partition(matrix, size):
        """Find Fiedler vector of a graph Laplacian to bisect nodes."""
        D = np.diag(matrix.sum(axis=1))
        L = D - matrix
        try:
            vals, vecs = np.linalg.eigh(L)
            idx = np.argsort(vals)
            fiedler = vecs[:, idx[1]]
        except Exception:
            fiedler = np.random.randn(size)
        
        return fiedler > 0

    def recursive_assign(facilities, locations, depth=0):
        """Recursively bisect and assign."""
        if len(facilities) <= 1:
            if len(facilities) == 1 and len(locations) == 1:
                return {facilities[0]: locations[0]}
            return {}
        
        if len(facilities) <= 2:
            result = {}
            for i, f in enumerate(facilities):
                result[f] = locations[i]
            return result
        
        sub_a = a[np.ix_(facilities, facilities)]
        fac_partition = fiedler_partition(sub_a, len(facilities))
        
        sub_b = b[np.ix_(locations, locations)]
        loc_partition = fiedler_partition(-sub_b, len(locations))
        
        fac_group1 = [facilities[i] for i in range(len(facilities)) if fac_partition[i]]
        fac_group2 = [facilities[i] for i in range(len(facilities)) if not fac_partition[i]]
        loc_group1 = [locations[i] for i in range(len(locations)) if loc_partition[i]]
        loc_group2 = [locations[i] for i in range(len(locations)) if not loc_partition[i]]
        
        if len(fac_group1) > len(loc_group1) and len(fac_group2) > 0:
            diff = len(fac_group1) - len(loc_group1)
            for _ in range(min(diff, len(fac_group1))):
                fac_group2.append(fac_group1.pop())
        elif len(fac_group1) < len(loc_group1) and len(fac_group2) > 0:
            diff = len(loc_group1) - len(fac_group1)
            for _ in range(min(diff, len(fac_group2))):
                fac_group1.append(fac_group2.pop())
        
        if len(fac_group1) == 0:
            fac_group1 = [facilities[0]]
            fac_group2 = facilities[1:]
        if len(fac_group2) == 0:
            fac_group2 = [facilities[-1]]
            fac_group1 = facilities[:-1]
        if len(loc_group1) == 0:
            loc_group1 = [locations[0]]
            loc_group2 = locations[1:]
        if len(loc_group2) == 0:
            loc_group2 = [locations[-1]]
            loc_group1 = locations[:-1]
        
        result = {}
        result.update(recursive_assign(fac_group1, loc_group1, depth + 1))
        result.update(recursive_assign(fac_group2, loc_group2, depth + 1))
        
        return result

    def spectral_bisection_init():
        """Generate initial permutation using spectral bisection approach."""
        all_facilities = list(range(n))
        all_locations = list(range(n))
        
        assignment = recursive_assign(all_facilities, all_locations)
        
        p = np.empty(n, dtype=np.int64)
        for fac, loc in assignment.items():
            p[fac] = loc
        
        assigned_locs = set(assignment.values())
        unassigned_facs = [f for f in range(n) if f not in assignment]
        unassigned_locs = [l for l in range(n) if l not in assigned_locs]
        for f, l in zip(unassigned_facs, unassigned_locs):
            p[f] = l
        
        return p

    def rank_matching_init():
        """Fallback rank-matching initialization."""
        a_row_sums = a.sum(axis=1)
        b_col_sums = b.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_col_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def tabu_kick(p, cost, n_kicks=5):
        """Apply block-reversal kicks with tabu for stronger diversification."""
        best_p = p.copy()
        best_cost = cost
        tabu_tenure = 8
        tabu_list = {}
        iteration = 0
        
        for _ in range(n_kicks):
            if time.time() > deadline - 5:
                break
            
            i, j = np.random.choice(n, 2, replace=False)
            if i > j:
                i, j = j, i
            key = (i, j)
            
            is_tabu = key in tabu_list and tabu_list[key] > iteration
            
            p[i:j+1] = p[i:j+1][::-1]
            new_cost = compute_cost(p)
            
            if not is_tabu or (new_cost < best_cost):
                cost = new_cost
                tabu_list[key] = iteration + tabu_tenure
                
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
            else:
                p[i:j+1] = p[i:j+1][::-1]
            
            iteration += 1
        
        return best_p, best_cost

    # Initial solution 1: spectral bisection for a globally informed start
    p = spectral_bisection_init()
    
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    # First local search
    current_p, current_cost = fast_local_search(current_p, current_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Also try rank-matching as a second initial solution
    p2 = rank_matching_init()
    p2, cost2 = fast_local_search(p2.copy(), compute_cost(p2))
    if cost2 < best_cost:
        best_p = p2.copy()
        best_cost = cost2

    # Iterated Local Search with periodic Hungarian re-optimization and tabu kicks
    iteration = 0
    while time.time() < deadline - 15:
        iteration += 1

        p = best_p.copy()
        
        block_len = np.random.randint(3, 11)
        start = np.random.randint(0, n - block_len)
        p[start:start+block_len] = p[start:start+block_len][::-1]
        
        p, cost = tabu_kick(p, compute_cost(p), n_kicks=2)

        p, cost = fast_local_search(p, cost)

        if iteration % 4 == 0 and time.time() < deadline - 30:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = fast_local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

    # Final high-precision polish
    if time.time() < deadline - 3:
        final_p = best_p.copy()
        final_cost = best_cost
        
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            final_p, final_cost = fast_local_search(final_p, final_cost, max_passes=30)
            if final_cost < best_cost:
                best_p = final_p.copy()
                best_cost = final_cost
    
    return best_p.tolist()
