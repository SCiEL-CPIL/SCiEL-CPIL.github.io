"""QAP tai256c solver: iterated local search with fast O(n) swap deltas."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    # Precompute row and column sums of a for initialization
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    # Initial solution: rank matching
    fac_order = np.argsort(-a_row_sums, kind="stable")
    loc_order = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order

    best_p = p.copy()
    best_cost = compute_cost(best_p)

    # Fast 2-opt local search with O(n) delta computation
    def local_search(p, cost):
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(n):
                if time.time() > deadline - 1.0:
                    break
                pi = p[i]
                for j in range(i + 1, n):
                    if time.time() > deadline - 1.0:
                        break
                    pj = p[j]
                    
                    # Compute delta for swapping p[i] and p[j]
                    # delta = sum_{k != i,j} a[i,k]*(b[pj,p[k]] - b[pi,p[k]])
                    #       + sum_{k != i,j} a[j,k]*(b[pi,p[k]] - b[pj,p[k]])
                    #       + sum_{k != i,j} a[k,i]*(b[p[k],pj] - b[p[k],pi])
                    #       + sum_{k != i,j} a[k,j]*(b[p[k],pi] - b[p[k],pj])
                    #       + a[i,j]*(b[pj,pi] - b[pi,pj])
                    #       + a[j,i]*(b[pi,pj] - b[pj,pi])
                    #       + a[i,i]*(b[pj,pj] - b[pi,pi])
                    #       + a[j,j]*(b[pi,pi] - b[pj,pj])
                    
                    # Vectorized computation
                    # Row i contribution: a[i,:] @ (b[pj,:] - b[pi,:]) indexed by p
                    # But we need to exclude i and j
                    diff_row_i = b[pj] - b[pi]  # shape (n,)
                    diff_row_j = b[pi] - b[pj]  # shape (n,)
                    
                    # For row i: sum_k a[i,k] * (b[pj, p[k]] - b[pi, p[k]])
                    # = a[i,:] @ diff_row_i[p]
                    row_i_delta = np.dot(a[i], diff_row_i[p])
                    # Subtract contributions from k=i and k=j
                    row_i_delta -= a[i, i] * (b[pj, pi] - b[pi, pi])
                    row_i_delta -= a[i, j] * (b[pj, pj] - b[pi, pj])
                    
                    # For row j: sum_k a[j,k] * (b[pi, p[k]] - b[pj, p[k]])
                    row_j_delta = np.dot(a[j], diff_row_j[p])
                    row_j_delta -= a[j, i] * (b[pi, pi] - b[pj, pi])
                    row_j_delta -= a[j, j] * (b[pi, pj] - b[pj, pj])
                    
                    # For column i: sum_k a[k,i] * (b[p[k], pj] - b[p[k], pi])
                    # = sum_k a[k,i] * (b[p[k], pj] - b[p[k], pi])
                    # = (a[:,i] * (b[:, pj] - b[:, pi]))[p] summed? No.
                    # b[p[k], pj] - b[p[k], pi] for each k
                    col_diff_i = b[:, pj] - b[:, pi]  # shape (n,)
                    col_i_delta = np.dot(a[:, i], col_diff_i[p])
                    col_i_delta -= a[i, i] * (b[pi, pj] - b[pi, pi])
                    col_i_delta -= a[j, i] * (b[pj, pj] - b[pj, pi])
                    
                    # For column j: sum_k a[k,j] * (b[p[k], pi] - b[p[k], pj])
                    col_diff_j = b[:, pi] - b[:, pj]  # shape (n,)
                    col_j_delta = np.dot(a[:, j], col_diff_j[p])
                    col_j_delta -= a[i, j] * (b[pi, pi] - b[pi, pj])
                    col_j_delta -= a[j, j] * (b[pj, pi] - b[pj, pj])
                    
                    # Cross terms
                    cross_ij = a[i, j] * (b[pj, pi] - b[pi, pj])
                    cross_ji = a[j, i] * (b[pi, pj] - b[pj, pi])
                    diag_ii = a[i, i] * (b[pj, pj] - b[pi, pi])
                    diag_jj = a[j, j] * (b[pi, pi] - b[pj, pj])
                    
                    delta = row_i_delta + row_j_delta + col_i_delta + col_j_delta + cross_ij + cross_ji + diag_ii + diag_jj
                    
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
        
        return p, cost

    # Initial local search
    p, cost = local_search(p.copy(), best_cost)
    if cost < best_cost:
        best_p = p.copy()
        best_cost = cost

    # Iterated local search
    np.random.seed(42)
    while time.time() < deadline - 2.0:
        # Perturb with random swaps
        p = best_p.copy()
        n_perturb = np.random.randint(2, 8)
        for _ in range(n_perturb):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]

        # Local search from perturbed solution
        p, cost = local_search(p, compute_cost(p))

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

    return best_p.tolist()
# EVOLVE-BLOCK-END
