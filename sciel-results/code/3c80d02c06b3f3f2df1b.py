"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Symmetrize for cost computation (standard QAP practice)
    # The cost is sum_{i,j} a[i,j] * b[p[i], p[j]]
    # We'll work with the matrices as given.

    # Initial solution: rank-matching heuristic
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    def compute_cost(p):
        # sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Vectorized: (a * b[np.ix_(p, p)]).sum()
        return np.sum(a * b[np.ix_(p, p)])

    best_p = p.copy()
    best_cost = compute_cost(best_p)

    def lap_surrogate(p):
        """
        Build a linear assignment surrogate based on current assignment.
        For each facility i, the 'effective cost' of assigning it to location l is:
        sum_j a[i,j] * b[l, p[j]]
        This captures the interaction with all other facilities.
        """
        # cost[i, l] = sum_j a[i,j] * b[l, p[j]]
        # = (a @ B_p)[i, l] where B_p[l, j] = b[l, p[j]]
        B_p = b[:, p]  # shape (n, n): B_p[l, j] = b[l, p[j]]
        cost_matrix = a @ B_p  # shape (n, n): cost_matrix[i, l] = sum_j a[i,j]*b[l,p[j]]
        # Solve LAP
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def two_opt_swap(p, i, j):
        """Try swapping facilities i and j in permutation p."""
        new_p = p.copy()
        new_p[i], new_p[j] = new_p[j], new_p[i]
        return new_p

    def delta_cost_swap(p, i, j):
        """
        Compute the change in cost when swapping facilities i and j.
        cost = sum_{u,v} a[u,v] * b[p[u], p[v]]
        When we swap p[i] and p[j], the terms that change involve:
        - terms where u or v is i or j
        Let's compute the difference directly.
        """
        pi = p[i]
        pj = p[j]
        # Terms involving i: sum_v a[i,v]*b[pi, p[v]] + sum_u a[u,i]*b[p[u], pi]
        # Terms involving j: sum_v a[j,v]*b[pj, p[v]] + sum_u a[u,j]*b[p[u], pj]
        # Cross terms: a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
        
        # After swap:
        # Terms involving i: sum_v a[i,v]*b[pj, p[v]] + sum_u a[u,i]*b[p[u], pj]
        # Terms involving j: sum_v a[j,v]*b[pi, p[v]] + sum_u a[u,j]*b[p[u], pi]
        # Cross terms: a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
        
        # Delta = (new terms) - (old terms)
        # Let's compute the contribution of all pairs (u,v) where u or v is in {i,j}
        
        # Old contribution:
        old = 0
        # u=i, v != i
        old += np.sum(a[i, :] * b[pi, p])
        # u != i, v=i
        old += np.sum(a[:, i] * b[p, pi])
        # u=j, v != j
        old += np.sum(a[j, :] * b[pj, p])
        # u != j, v=j
        old += np.sum(a[:, j] * b[p, pj])
        # Note: the pair (i,j) and (j,i) are counted in both above, so we need to be careful.
        # Actually, let's just compute the full delta by considering all (u,v) where u or v is i or j.
        
        # Simpler approach: compute the cost of the submatrix involving i and j.
        # Let's just compute the full cost difference by evaluating the affected terms.
        
        # The total cost is sum_{u,v} a[u,v] * b[p[u], p[v]]
        # When we swap p[i] <-> p[j], the only terms that change are those where u or v is i or j.
        # Let's compute:
        # delta = sum_{u,v in {i,j} x [n]} ... this gets complicated.
        
        # Alternative: just compute the two costs for the affected rows/cols.
        # Actually, let's use a direct computation for small n or just do the swap and compute.
        # For n=100, computing the full cost is O(n^2) = 10000 ops, which is fast.
        # But we want to be efficient. Let's compute the delta analytically.
        
        # The cost can be written as sum_{u,v} a[u,v] * b[p[u], p[v]]
        # Let C(u,v) = a[u,v] * b[p[u], p[v]]
        # When we swap p[i] and p[j], C(u,v) changes only if u or v is i or j.
        
        # Let's compute the sum of C(u,v) over all (u,v) where u or v is in {i,j}, before and after.
        
        # Before:
        before = 0
        for u in [i, j]:
            for v in range(n):
                before += a[u, v] * b[p[u], p[v]]
        for u in range(n):
            for v in [i, j]:
                if u not in [i, j]:  # avoid double counting
                    before += a[u, v] * b[p[u], p[v]]
        
        # After swap:
        after = 0
        p_new = p.copy()
        p_new[i], p_new[j] = p_new[j], p_new[i]
        for u in [i, j]:
            for v in range(n):
                after += a[u, v] * b[p_new[u], p_new[v]]
        for u in range(n):
            for v in [i, j]:
                if u not in [i, j]:
                    after += a[u, v] * b[p_new[u], p_new[v]]
        
        return after - before

    # Main search loop
    improved = True
    while time.time() < deadline:
        # Local search with 2-opt swaps
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                delta = delta_cost_swap(best_p, i, j)
                if delta < 0:
                    best_p[i], best_p[j] = best_p[j], best_p[i]
                    best_cost += delta
                    improved = True
                    # Restart the inner loop since the permutation changed
                    break
            if improved:
                break
        
        # If no improvement from 2-opt, try LAP-guided perturbation
        if not improved:
            new_p = lap_surrogate(best_p)
            new_cost = compute_cost(new_p)
            if new_cost < best_cost:
                best_p = new_p
                best_cost = new_cost
            else:
                # Try a few random restarts with LAP
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    # Random perturbation of current best
                    perturbed = best_p.copy()
                    for _ in range(10):
                        i, j = np.random.choice(n, 2, replace=False)
                        perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                    new_p = lap_surrogate(perturbed)
                    new_cost = compute_cost(new_p)
                    if new_cost < best_cost:
                        best_p = new_p
                        best_cost = new_cost

    return best_p.tolist()
# EVOLVE-BLOCK-END
