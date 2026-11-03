"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Precompute
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0

    # Initial solution: rank-matching (heavy flows to short distances)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)

    def compute_cost(p):
        # sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Efficient: build permuted b matrix
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    best_p = p.copy()
    best_cost = compute_cost(p)

    # Local search: 2-opt style swaps
    # For a swap of facilities i and j, the delta in cost can be computed efficiently
    # cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    # Swapping p[i] and p[j]:
    #   New terms involving i or j change.
    #   delta = sum_{k != i,j} a[i,k](b[p[j],p[k]] - b[p[i],p[k]])
    #         + sum_{k != i,j} a[j,k](b[p[i],p[k]] - b[p[j],p[k]])
    #         + sum_{k != i,j} a[k,i](b[p[k],p[j]] - b[p[k],p[i]])
    #         + sum_{k != i,j} a[k,j](b[p[k],p[i]] - b[p[k],p[j]])
    #         + a[i,j](b[p[j],p[i]] - b[p[i],p[j]])
    #         + a[j,i](b[p[i],p[j]] - b[p[j],p[i]])
    # Simplify using symmetry of b:
    #   Let u = p[i], v = p[j]
    #   delta = sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
    #         + sum_{k!=i,j} (a[k,i] - a[k,j]) * (b[p[k], v] - b[p[k], u])
    #         + (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
    # Since b is symmetric (or we use b_sym), b[v,u] = b[u,v], so last term is 0 if b symmetric.
    # But b might not be perfectly symmetric. Let's just compute directly for correctness.

    # Actually, let's use a simpler delta computation:
    # delta = sum_k a[i,k]*(b[v,p[k]]-b[u,p[k]]) + sum_k a[k,i]*(b[p[k],v]-b[p[k],u])
    #       + sum_k a[j,k]*(b[u,p[k]]-b[v,p[k]]) + sum_k a[k,j]*(b[p[k],u]-b[p[k],v])
    # where k runs over all, but for k=i and k=j we need to be careful.
    # Let's just do: for k != i, j:
    #   terms from a[i,k] and a[k,i] and a[j,k] and a[k,j]
    # and the diagonal-ish terms a[i,j], a[j,i]

    # Simpler approach: precompute row/col sums for fast delta
    # Let's define for current permutation p:
    #   row_i = sum_k a[i,k] * b[u, p[k]]  for facility i at location u
    #   col_i = sum_k a[k,i] * b[p[k], u]
    # Then delta for swapping i (at u) and j (at v):
    #   Remove contributions of i and j, add new ones.

    # Let's implement a clean 2-opt local search with O(n) delta per swap.

    def compute_delta_swap(p, i, j):
        u = p[i]
        v = p[j]
        # delta = sum_{k!=i,j} a[i,k]*(b[v,p[k]] - b[u,p[k]])
        #       + sum_{k!=i,j} a[k,i]*(b[p[k],v] - b[p[k],u])
        #       + sum_{k!=i,j} a[j,k]*(b[u,p[k]] - b[v,p[k]])
        #       + sum_{k!=i,j} a[k,j]*(b[p[k],u] - b[p[k],v])
        #       + a[i,j]*(b[v,u] - b[u,v]) + a[j,i]*(b[u,v] - b[v,u])
        # = sum_{k!=i,j} (a[i,k] - a[j,k])*(b[v,p[k]] - b[u,p[k]])
        #   + sum_{k!=i,j} (a[k,i] - a[k,j])*(b[p[k],v] - b[p[k],u])
        #   + (a[i,j] - a[j,i])*(b[v,u] - b[u,v])

        # For efficiency, precompute:
        #   diff_row[i,j] = a[i,:] - a[j,:]
        #   diff_col[i,j] = a[:,i] - a[:,j]
        # But that's O(n^2) memory. For n=100, that's 10000 entries, fine.

        # Actually let's just compute directly with numpy vectorization.
        # Exclude i and j from sums
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False

        diff_a_row = a[i, mask] - a[j, mask]  # (n-2,)
        diff_b_row = b[v, p[mask]] - b[u, p[mask]]  # (n-2,)
        term1 = np.dot(diff_a_row, diff_b_row)

        diff_a_col = a[mask, i] - a[mask, j]  # (n-2,)
        diff_b_col = b[p[mask], v] - b[p[mask], u]  # (n-2,)
        term2 = np.dot(diff_a_col, diff_b_col)

        term3 = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])

        return term1 + term2 + term3

    # Hill climbing with random restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            p = best_p.copy()
        else:
            # Perturb best_p
            p = best_p.copy()
            # Randomly swap a few pairs
            num_swaps = 5 + restart // 10
            for _ in range(num_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]

        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all pairs (or a random subset for speed)
            # For n=100, 4950 pairs. Let's try all in each pass.
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    delta = compute_delta_swap(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        improved = True
                        if not improved:
                            break

            # Re-evaluate
            cur_cost = compute_cost(p)
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_p = p.copy()

        # If we improved, we might want to do another pass
        if improved:
            # Continue hill climbing from new position
            continue

    # Final polish: try all single swaps one more time
    if time.time() < deadline:
        p = best_p.copy()
        for i in range(n):
            for j in range(i + 1, n):
                delta = compute_delta_swap(p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
        cur_cost = compute_cost(p)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
