"""Benders-decomposition QAP solver for wil100.

Master problem: linear assignment minimizing surrogate cost.
Subproblem: evaluate exact QAP cost, generate a cutting plane
that discourages the current assignment and guides the master
toward lower true cost.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def _cost(p, a, b):
    """Exact QAP cost for permutation p (facility i -> location p[i])."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _delta_swap(a, b, p, i, j):
    """Compute change in QAP cost when swapping facilities i and j in assignment p."""
    pi, pj = p[i], p[j]
    if pi == pj:
        return 0
    # Delta = sum_k [a[i,k]*(b[pj,k]-b[pi,k]) + a[j,k]*(b[pi,k]-b[pj,k])
    #         + a[k,i]*(b[pj,k]-b[pi,k]) + a[k,j]*(b[pi,k]-b[pj,k])]
    # Simplify: (a[i,:] - a[j,:]) @ (b[pj,:] - b[pi,:])
    #          + (a[:,i] - a[:,j]) @ (b[pj,:] - b[pi,:])
    diff_a = a[i] - a[j]
    diff_b = b[pj] - b[pi]
    diff_a_col = a[:, i] - a[:, j]
    delta = np.dot(diff_a, diff_b) + np.dot(diff_a_col, diff_b)
    return int(delta)


def _local_search(p, a, b, max_passes=10, deadline=None):
    """First-improvement local search using delta-based swap evaluation."""
    n = len(p)
    cost = _cost(p, a, b)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try all pairs in random order for diversity
        order = np.random.permutation(n * (n - 1) // 2)
        for idx in order:
            if deadline is not None and time.time() > deadline:
                break
            # Convert linear index to (i, j) pair
            i = int((np.sqrt(8 * idx + 1) - 1) // 2)
            j = int(idx - i * (i + 1) // 2)
            if i >= n or j >= n or i == j:
                continue
            d = _delta_swap(a, b, p, i, j)
            if d < 0:
                p[i], p[j] = p[j], p[i]
                cost += d
                improved = True
        if not improved:
            break
    return p, cost


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0   # safety margin before 120 s wall-clock

    # Precompute symmetric part of a (flow is typically symmetric) and b
    A = 0.5 * (a + a.T)
    B = 0.5 * (b + b.T)

    # Precompute base surrogate cost matrix: S[i,j] = sum_k a[i,k] * B[j,k]
    # This is the linearized flow-cost if facility i goes to location j
    surrogate_base = A @ B  # shape (n, n)

    # Row-sums for initial rank-matching heuristic
    a_row = a.sum(axis=1)
    b_row = b.sum(axis=1)

    # Initial assignment: rank-match (heavy flow rows -> low distance rows)
    heavy = np.argsort(-a_row, kind="stable")
    central = np.argsort(b_row, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    # Local search to polish initial solution
    p, best_cost = _local_search(p, a, b, max_passes=5, deadline=deadline)
    best_p = p.copy()

    # Benders cutting-plane: penalty matrix C[i,j] discourages placing facility i at location j
    C = np.zeros((n, n), dtype=np.float64)
    step = 1.0
    min_step = 1e-6

    # Track lower bound estimate
    lb_estimate = best_cost

    iteration = 0
    no_improve_count = 0
    while time.time() < deadline:
        iteration += 1

        # --- Master problem: linear assignment with surrogate cost ---
        # Surrogate cost: S[i,j] + C[i,j]
        surrogate_cost = surrogate_base + C

        p_new, _ = linear_sum_assignment(surrogate_cost)
        p_new = p_new.astype(np.int64)

        # --- Subproblem: evaluate exact cost ---
        exact_cost = _cost(p_new, a, b)

        # --- Generate cutting plane ---
        # The gap between exact cost and surrogate suggests the surrogate
        # underestimates the true cost for this assignment.
        # Add penalty to specific (facility, location) pairs that were used.
        surrogate_val = float(np.sum(surrogate_base[np.arange(n), p_new]))
        gap = exact_cost - surrogate_val

        if gap > 0 and exact_cost >= best_cost:
            # Penalize the specific pairs used in p_new
            # Scale penalty by gap to make the cut stronger when gap is large
            penalty = step * gap / max(best_cost, 1.0)
            for i in range(n):
                C[i, p_new[i]] += penalty
            step *= 0.97
            if step < min_step:
                step = min_step
        else:
            # Accept or near-accept: decay penalties to allow exploration
            if exact_cost < best_cost:
                best_cost = exact_cost
                best_p = p_new.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            C *= 0.85
            step = max(step * 1.05, min_step)

        # --- Local improvement on best_p ---
        if time.time() < deadline - 10:
            p_ls, cost_ls = _local_search(best_p.copy(), a, b, max_passes=3, deadline=deadline)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls

        # --- LAP-guided perturbation: re-solve master with perturbed costs ---
        if time.time() < deadline - 15:
            # Perturb with noise scaled to problem magnitude
            noise_scale = best_cost / (n * n) * 0.05
            noise = np.random.randn(n, n) * noise_scale
            surrogate_perturbed = surrogate_base + C + noise
            p_pert, _ = linear_sum_assignment(surrogate_perturbed)
            p_pert = p_pert.astype(np.int64)

            # Polish perturbed solution
            p_pert, pert_cost = _local_search(p_pert, a, b, max_passes=3, deadline=deadline)
            if pert_cost < best_cost:
                best_cost = pert_cost
                best_p = p_pert.copy()
                no_improve_count = 0

        # Check deadline
        if time.time() > deadline - 5.0:
            break

        # If stuck, do a larger perturbation
        if no_improve_count > 20:
            C *= 0.5
            no_improve_count = 0

    # Final polish with more iterations
    if time.time() < deadline - 2:
        best_p, best_cost = _local_search(best_p.copy(), a, b, max_passes=20, deadline=deadline)

    return best_p.tolist()
