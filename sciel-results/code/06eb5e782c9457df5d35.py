"""Rewritten QAP solver for wil100 – spectral init + vectorised 2-opt local search with restarts."""
import numpy as np
import time
from scipy.linalg import eigh


def _qap_cost(a, b, p):
    """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _spectral_init(a, b):
    """
    Spectral initialization: align high-flow facilities with high-distance locations
    using eigenvectors of the flow and distance matrices.
    """
    n = len(a)
    # Use the largest-magnitude eigenvectors of a and b to define 'directions'
    # We want to pair facilities with large 'flow weight' to locations with large 'distance weight'
    # Take the eigenvector corresponding to the largest eigenvalue (or smallest if negative)
    try:
        w_a, v_a = eigh(a, subset_by_index=[n - 3, n - 1], eigvals_only=False)
        w_b, v_b = eigh(b, subset_by_index=[n - 3, n - 1], eigvals_only=False)
        # Use the last eigenvector (largest eigenvalue)
        va = v_a[:, -1]
        vb = v_b[:, -1]
    except Exception:
        va = a.sum(axis=1)
        vb = b.sum(axis=1)

    # Score facilities by their projection onto the dominant eigenvector
    fac_scores = np.abs(va) * np.sign(va)  # signed to preserve orientation
    loc_scores = np.abs(vb) * np.sign(vb)

    # Sort facilities by descending |flow weight|, locations by ascending |distance weight|
    fac_order = np.argsort(-np.abs(fac_scores))
    loc_order = np.argsort(np.abs(loc_scores))

    p = np.empty(n, dtype=np.int64)
    for rank in range(n):
        p[fac_order[rank]] = loc_order[rank]
    return p


def _swap_delta_vectorized(a, b, p, i, j):
    """
    Compute cost change for swapping positions i and j in permutation p.
    Uses vectorized operations over all k != i, j.
    """
    pi = p[i]
    pj = p[j]
    n = len(p)

    # Build mask for k != i, j
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    ks = np.nonzero(mask)[0]

    delta = 0.0

    # Diagonal terms
    delta += (a[i, i] - a[j, j]) * (b[pj, pj] - b[pi, pi])
    # Off-diagonal i-j terms
    delta += (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])

    if len(ks) > 0:
        pk = p[ks]
        # S1+S2: sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[pj,pk] - b[pi,pk])
        diff_a_row = a[i, ks] - a[j, ks]
        diff_b_col = b[pj, pk] - b[pi, pk]
        delta += float(np.dot(diff_a_row, diff_b_col))

        # S3+S4: sum_{k!=i,j} (a[k,i] - a[k,j]) * (b[pk,pj] - b[pk,pi])
        diff_a_col = a[ks, i] - a[ks, j]
        diff_b_row = b[pk, pj] - b[pk, pi]
        delta += float(np.dot(diff_a_col, diff_b_row))

    return delta


def _local_search(a, b, p, max_time=50.0):
    """
    First-improvement 2-opt local search with time limit.
    Returns (permutation, cost).
    """
    n = len(p)
    p = p.copy()
    start = time.time()
    cost = _qap_cost(a, b, p)

    improved = True
    while improved:
        improved = False
        if time.time() - start > max_time:
            break

        # Scan for first improving swap
        for i in range(n):
            if time.time() - start > max_time:
                break
            for j in range(i + 1, n):
                d = _swap_delta_vectorized(a, b, p, i, j)
                if d < -1e-9:
                    p[i], p[j] = p[j], p[i]
                    cost += d
                    improved = True
                    break
            if improved:
                break  # restart scan from beginning after a move

    return p, cost


def _perturb(p, num_swaps=3):
    """Apply random swaps to escape local optimum."""
    n = len(p)
    p2 = p.copy()
    for _ in range(num_swaps):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i != j:
            p2[i], p2[j] = p2[j], p2[i]
    return p2


def solve(a, b):
    n = len(a)
    total_deadline = 145.0  # leave 5s buffer under 150s
    t0 = time.time()

    # Initial spectral solution
    p = _spectral_init(a, b)
    p, cost = _local_search(a, b, p, max_time=min(30.0, total_deadline - 20.0))

    best_p = p.copy()
    best_cost = cost

    # Iterated Local Search: perturb + local search
    elapsed = time.time() - t0
    while elapsed < total_deadline - 10.0:
        # Choose number of perturbations adaptively
        num_swaps = 2 + np.random.randint(0, 3)
        p_pert = _perturb(best_p, num_swaps=num_swaps)

        # Local search on perturbed solution, with remaining time budget
        remaining = total_deadline - (time.time() - t0) - 5.0
        if remaining < 5.0:
            break
        ls_time = min(remaining * 0.8, 40.0)
        p_ls, cost_ls = _local_search(a, b, p_pert, max_time=ls_time)

        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()

        elapsed = time.time() - t0

    # Final polish: one more local search pass on best with tight time
    remaining = total_deadline - (time.time() - t0)
    if remaining > 5.0:
        best_p, best_cost = _local_search(a, b, best_p, max_time=max(1.0, remaining - 2.0))

    return best_p.tolist()
