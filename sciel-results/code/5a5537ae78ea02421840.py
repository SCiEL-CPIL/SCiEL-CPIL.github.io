"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 145.0  # safety margin before 150s kill

    # Symmetrize a (flows) to exploit structure; b (distances) keep as-is
    # a[i,j] * b[p[i],p[j]] — if a is symmetric, cost = sum a[i,j]*b[p[i],p[j]]
    # We'll work with the given matrices directly.

    # Step 1: Get an initial assignment via rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    # Compute initial cost
    cost = _compute_cost(a, b, p)

    # Step 2: Hungarian-assisted re-optimization
    # Build a cost matrix for Hungarian: for facility i at location j,
    # approximate marginal cost. Use row-sum matching as a proxy.
    # Actually, let's try a different approach: use the structure of QAP.
    # The cost is sum_{i,j} a[i,j] * b[p[i], p[j]].
    # We can think of this as: for each pair (i,j), the contribution is a[i,j]*b[p[i],p[j]].
    # A good initial solution can come from trying to align high a[i,j] with low b[*,*].

    # Let's try Hungarian on a derived matrix:
    # For facility i, define its "demand vector" d_i[j] = sum_k a[i,k] (row sum)
    # For location j, define its "supply" s_j = sum_k b[j,k] (row sum)
    # But that's just rank matching again.

    # Better: use the fact that cost = trace(A^T P B P^T) where P is permutation matrix.
    # Approximate by minimizing sum_i (A @ P B P^T)[i,i] = sum_i sum_j a[i,j] b[p[i],p[j]]
    # which is exactly our cost.

    # Let's try a simple local search with 2-opt swaps.
    # For swapping p[i] and p[j], the delta in cost can be computed efficiently.

    # Precompute for fast delta computation:
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # If we swap p[i] <-> p[j], only terms involving i or j change.
    # Old terms involving i: sum_j a[i,j]*b[p[i],p[j]] + sum_j a[j,i]*b[p[j],p[i]]
    # New terms: sum_j a[i,j]*b[p[j],p[j]] ... wait, need to be careful.

    # Let's define:
    # Let x = p[i], y = p[j] (the two locations being swapped)
    # Old contribution from pairs (i,*) and (*,i) and (j,*) and (*,j):
    #   sum_k a[i,k]*b[x, p[k]] + sum_k a[k,i]*b[p[k], x] + sum_k a[j,k]*b[y, p[k]] + sum_k a[k,j]*b[p[k], y]
    #   minus a[i,j]*b[x,y] and a[j,i]*b[y,x] which are counted twice? No, let's think again.

    # Actually, let's just compute the full delta by iterating over all k != i, j:
    # delta = sum_{k != i, j} [a[i,k]*b[y,p[k]] + a[k,i]*b[p[k],y] + a[j,k]*b[x,p[k]] + a[k,j]*b[p[k],x]
    #          - a[i,k]*b[x,p[k]] - a[k,i]*b[p[k],x] - a[j,k]*b[y,p[k]] - a[k,j]*b[p[k],y]]
    #         + a[i,j]*b[y,x] + a[j,i]*b[x,y] - a[i,j]*b[x,y] - a[j,i]*b[y,x]

    # This is O(n) per swap, and we try O(n^2) swaps per pass, so O(n^3) per pass.
    # For n=150, that's ~3.4M operations per pass, which is fast.

    # Let's implement a simple 2-opt local search.

    # First, let's try to improve the initial solution with a few passes of 2-opt.
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Precompute for efficiency
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, n):
                # Compute delta for swapping p[i] and p[j]
                x, y = p[i], p[j]
                delta = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    pk = p[k]
                    delta += a[i, k] * (b[y, pk] - b[x, pk])
                    delta += a[k, i] * (b[pk, y] - b[pk, x])
                    delta += a[j, k] * (b[x, pk] - b[y, pk])
                    delta += a[k, j] * (b[pk, x] - b[pk, y])
                # Terms involving i and j directly
                delta += a[i, j] * (b[y, x] - b[x, y])
                delta += a[j, i] * (b[x, y] - b[y, x])

                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
        if time.time() > deadline - 5.0:
            break

    return p.tolist()


def _compute_cost(a, b, p):
    n = len(a)
    cost = 0
    for i in range(n):
        for j in range(n):
            cost += a[i, j] * b[p[i], p[j]]
    return cost
# EVOLVE-BLOCK-END
