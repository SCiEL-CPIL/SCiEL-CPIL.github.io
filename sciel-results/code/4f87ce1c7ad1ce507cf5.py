"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s kill

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def local_search(p, time_limit):
        """Perform 2-opt local search with vectorized delta computation."""
        current_cost = compute_cost(p)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1

            # Precompute b indexed by p for faster access
            b_p = b[np.ix_(p, p)]  # b_p[i,j] = b[p[i], p[j]]

            for i in range(n):
                if time.time() > time_limit:
                    break
                u = p[i]
                # For this i, compute delta for all j > i
                # Precompute vectors that don't depend on j
                a_i = a[i, :]
                a_col_i = a[:, i]

                for j in range(i + 1, n):
                    v = p[j]
                    a_j = a[j, :]
                    a_col_j = a[:, j]

                    # Compute delta using the block formula
                    # Old terms involving i or j:
                    # a[i,i]*b[u,u] + a[i,j]*b[u,v] + a[j,i]*b[v,u] + a[j,j]*b[v,v]
                    # + sum_{k!=i,j} [a[i,k]*b[u,p[k]] + a[k,i]*b[p[k],u] + a[j,k]*b[v,p[k]] + a[k,j]*b[p[k],v]]
                    #
                    # New terms:
                    # a[i,i]*b[v,v] + a[i,j]*b[v,u] + a[j,i]*b[u,v] + a[j,j]*b[u,u]
                    # + sum_{k!=i,j} [a[i,k]*b[v,p[k]] + a[k,i]*b[p[k],v] + a[j,k]*b[u,p[k]] + a[k,j]*b[p[k],u]]

                    # Use precomputed b_p where b_p[k,l] = b[p[k],p[l]]
                    # b[u, p[k]] = b_p[np.where(p==u)[0][0], k] -- this is slow
                    # Better: b[u, p] and b[p, u]

                    bu_p = b[u, p]
                    bv_p = b[v, p]
                    bp_u = b[p, u]
                    bp_v = b[p, v]

                    # Sum over all k, then subtract k=i and k=j
                    s_i_u = a_i @ bu_p
                    s_i_v = a_i @ bv_p
                    s_j_u = a_j @ bu_p
                    s_j_v = a_j @ bv_p
                    s_u_i = a_col_i @ bp_u
                    s_v_i = a_col_i @ bp_v
                    s_u_j = a_col_j @ bp_u
                    s_v_j = a_col_j @ bp_v

                    # Subtract k=i and k=j
                    aii = a[i, i]
                    aij = a[i, j]
                    aji = a[j, i]
                    ajj = a[j, j]
                    buu = b[u, u]
                    buv = b[u, v]
                    bvu = b[v, u]
                    bvv = b[v, v]

                    s_i_u_c = s_i_u - aii * buu - aij * buv
                    s_i_v_c = s_i_v - aii * bvu - aij * bvv
                    s_j_u_c = s_j_u - aji * buu - ajj * buv
                    s_j_v_c = s_j_v - aji * bvu - ajj * bvv
                    s_u_i_c = s_u_i - aii * buu - aji * bvu
                    s_v_i_c = s_v_i - aii * buv - aji * bvv
                    s_u_j_c = s_u_j - aij * buu - ajj * bvu
                    s_v_j_c = s_v_j - aij * buv - ajj * bvv

                    delta = (s_i_v_c + s_v_i_c + s_j_u_c + s_u_j_c) - \
                            (s_i_u_c + s_u_i_c + s_j_v_c + s_v_j_c) + \
                            (aii * bvv + aij * bvu + aji * buv + ajj * buu) - \
                            (aii * buu + aij * buv + aji * bvu + ajj * bvv)

                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j

            if best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
            else:
                break

        return p, current_cost

    # Multiple restarts with different initializations
    best_p = None
    best_cost = np.inf
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts

    for restart in range(num_restarts):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        t_limit = time.time() + min(time_per_restart, remaining - 2.0)

        if restart == 0:
            # Rank-matching initialization
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        else:
            # Random initialization with some structure
            p = np.random.permutation(n).astype(np.int64)
            # Apply a few random swaps to perturb
            for _ in range(10):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]

        p, cost = local_search(p, t_limit)

        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

    if best_p is None:
        # Fallback
        p = np.random.permutation(n).astype(np.int64)
        best_p = p

    return best_p.tolist()
# EVOLVE-BLOCK-END
