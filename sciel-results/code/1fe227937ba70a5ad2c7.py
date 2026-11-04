"""QAP solver: systematic local search with precomputed O(1) swap deltas + Hungarian + restarts."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0

    def compute_cost(p):
        B_p = b[np.ix_(p, p)]
        return int(np.sum(a * B_p))

    # Precompute b columns for fast access
    b_cols = {l: b[:, l] for l in range(n)}

    def build_row_cost(p):
        """row_cost[i, l] = sum_k a[i,k] * b[l, p[k]]"""
        # b[:, p] gives shape (n, n) where [l, k] = b[l, p[k]]
        b_col_p = b[:, p]
        return a @ b_col_p

    def swap_delta(i, j, p, row_cost):
        """O(1) swap delta using precomputed row_cost.
        
        cost = sum_k row_cost[k, p[k]]
        After swapping p[i]=u and p[j]=v:
        - row_cost[k, l] changes by (a[k,i]-a[k,j])*(b[l,v]-b[l,u]) for all k,l
        - p[i] becomes v, p[j] becomes u
        
        New cost - Old cost:
        = [new_row_cost[i,v] + new_row_cost[j,u] + sum_{k!=i,j} new_row_cost[k,p[k]]]
          - [row_cost[i,u] + row_cost[j,v] + sum_{k!=i,j} row_cost[k,p[k]]]
        
        For k != i,j: new_row_cost[k,p[k]] = row_cost[k,p[k]] + (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
        For i: new_row_cost[i,v] = row_cost[i,v] + (a[i,i]-a[i,j])*(b[v,v]-b[v,u])
        For j: new_row_cost[j,u] = row_cost[j,u] + (a[j,i]-a[j,j])*(b[u,v]-b[u,u])
        
        Delta = (row_cost[i,v] - row_cost[i,u]) + (row_cost[j,u] - row_cost[j,v])
              + (a[i,i]-a[i,j])*(b[v,v]-b[v,u]) + (a[j,i]-a[j,j])*(b[u,v]-b[u,u])
              + sum_{k!=i,j} (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
        
        The sum term: sum_k (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u]) 
                    - (a[i,i]-a[i,j])*(b[u,v]-b[u,u]) 
                    - (a[j,i]-a[j,j])*(b[v,u]-b[v,v])
        """
        u = p[i]
        v = p[j]
        
        d = row_cost[i, v] - row_cost[i, u]
        d += row_cost[j, u] - row_cost[j, v]
        d += (a[i, i] - a[i, j]) * (b[v, v] - b[v, u])
        d += (a[j, i] - a[j, j]) * (b[u, v] - b[u, u])
        
        # Sum over all k: (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
        # = (a[:,i] - a[:,j]) @ (b[p, v] - b[p, u])
        diff_a = a[:, i] - a[:, j]
        diff_b = b[p, v] - b[p, u]
        S = float(np.dot(diff_a, diff_b))
        # Subtract k=i and k=j terms
        S -= (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
        S -= (a[j, i] - a[j, j]) * (b[v, u] - b[v, v])
        
        d += S
        return d

    def local_search_first_improvement(p, cost, time_limit, max_passes=50):
        """First-improvement local search: scan all pairs, apply first improving swap."""
        row_cost = build_row_cost(p)
        passes = 0
        while passes < max_passes and time.time() < time_limit:
            passes += 1
            improved = False
            for i in range(n):
                if time.time() > time_limit:
                    break
                for j in range(i + 1, n):
                    d = swap_delta(i, j, p, row_cost)
                    if d < -1e-9:
                        u = p[i]
                        v = p[j]
                        p[i], p[j] = v, u
                        cost += d
                        improved = True
                        # Update row_cost: rank-1 update
                        da = a[:, i] - a[:, j]
                        db = b[:, v] - b[:, u]
                        row_cost += np.outer(da, db)
                        break
                if improved:
                    break
            if not improved:
                break
        return p, cost

    def hungarian_refine(p_cur, cost_cur):
        """Hungarian linearization refinement."""
        b_col_p = b[:, p_cur]
        b_row_p = b[p_cur, :]
        C = a @ b_col_p.T + a.T @ b_row_p
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        cost_new = compute_cost(p_new)
        if cost_new < cost_cur:
            return p_new, cost_new
        return p_cur, cost_cur

    def spectral_init():
        """Spectral initialization using eigenvalues."""
        # Use the symmetric part of a and b
        a_sym = (a + a.T) / 2
        b_sym = (b + b.T) / 2
        # Get top eigenvectors
        try:
            evals_a, evecs_a = np.linalg.eigh(a_sym)
            evals_b, evecs_b = np.linalg.eigh(b_sym)
            # Sort by absolute value descending
            idx_a = np.argsort(-np.abs(evals_a))
            idx_b = np.argsort(-np.abs(evals_b))
            v_a = evecs_a[:, idx_a[:n//2]]
            v_b = evecs_b[:, idx_b[:n//2]]
            # Match by sign of dot products
            M = v_a.T @ v_b
            row_ind, col_ind = linear_sum_assignment(-M)
            p = np.empty(n, dtype=np.int64)
            # Map facilities to locations via spectral matching
            # Use the full eigenvector matrix for better matching
            V_a = evecs_a[:, idx_a]
            V_b = evecs_b[:, idx_b]
            M_full = V_a.T @ V_b
            row_ind, col_ind = linear_sum_assignment(-M_full)
            p[row_ind] = col_ind
            return p
        except:
            return None

    def random_init():
        """Random initialization."""
        return np.random.permutation(n)

    def rank_init():
        """Rank-matching initialization."""
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    # Generate multiple initial solutions
    best_p = None
    best_cost = float('inf')

    # Try spectral init
    p_spec = spectral_init()
    if p_spec is not None:
        c = compute_cost(p_spec)
        if c < best_cost:
            best_cost = c
            best_p = p_spec.copy()

    # Try rank init
    p_rank = rank_init()
    c = compute_cost(p_rank)
    if c < best_cost:
        best_cost = c
        best_p = p_rank.copy()

    # Try random inits
    for _ in range(3):
        if time.time() > deadline - 60:
            break
        p_rand = random_init()
        c = compute_cost(p_rand)
        if c < best_cost:
            best_cost = c
            best_p = p_rand.copy()

    # Hungarian refine the best
    if time.time() < deadline - 50:
        best_p, best_cost = hungarian_refine(best_p, best_cost)

    # Local search
    if time.time() < deadline - 30:
        best_p, best_cost = local_search_first_improvement(best_p, best_cost, deadline - 30)

    # Interleave Hungarian and local search
    for _ in range(8):
        if time.time() > deadline - 15:
            break
        best_p, best_cost = hungarian_refine(best_p, best_cost)
        if time.time() > deadline - 10:
            break
        best_p, best_cost = local_search_first_improvement(best_p, best_cost, deadline - 10)

    # Final polish with more passes
    if time.time() < deadline - 3:
        best_p, best_cost = local_search_first_improvement(best_p, best_cost, deadline - 3, max_passes=100)

    # Final Hungarian
    if time.time() < deadline - 1:
        best_p, best_cost = hungarian_refine(best_p, best_cost)

    return best_p.tolist()
# EVOLVE-BLOCK-END
