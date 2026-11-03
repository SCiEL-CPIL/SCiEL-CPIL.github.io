"""Benders-decomposition QAP solver for wil100.

Master problem: linear assignment minimizing surrogate cost.
Subproblem: evaluate exact QAP cost, generate a cutting plane
(per-facility penalty) that discourages the current assignment
and guides the master toward lower true cost.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def _cost(p, a, b):
    """Exact QAP cost for permutation p (facility i -> location p[i])."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _surrogate(p, a, b, c):
    """Surrogate cost: sum_i sum_j a[i,j]*b[p[i],p[j]] + sum_i c[i]*p[i]."""
    return int(np.sum(a * b[np.ix_(p, p)])) + int(np.dot(c, p))


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0   # safety margin before 120 s wall-clock

    # Precompute symmetric part of a (flow is typically symmetric) and b
    A = 0.5 * (a + a.T)
    B = 0.5 * (b + b.T)

    # Row-sums for initial rank-matching heuristic
    a_row = a.sum(axis=1)
    b_row = b.sum(axis=1)

    # Initial assignment: rank-match (heavy flow rows -> low distance rows)
    heavy = np.argsort(-a_row, kind="stable")
    central = np.argsort(b_row, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    best_p = p.copy()
    best_cost = _cost(best_p, a, b)

    # Benders cutting-plane vector
    c = np.zeros(n, dtype=np.float64)
    step = 1.0
    min_step = 1e-6

    iteration = 0
    while time.time() < deadline:
        iteration += 1

        # --- Master problem: linear assignment with surrogate cost ---
        # Surrogate cost matrix: cost[i, j] = sum_k a[i,k] * b[j, k] + c[i] * j
        # (facility i at location j, total linearized flow cost)
        surrogate_cost = B @ (a.T)  # shape (n, n): surrogate_cost[i,j] = sum_k a[i,k]*b[j,k]
        # Add cutting-plane penalty
        surrogate_cost += np.outer(c, np.arange(n, dtype=np.float64))

        p_new, _ = linear_sum_assignment(surrogate_cost)
        p_new = p_new.astype(np.int64)

        # --- Subproblem: evaluate exact cost ---
        exact_cost = _cost(p_new, a, b)

        # --- Generate cutting plane ---
        # If exact cost > surrogate + best_cost, add penalty to discourage
        # this assignment. The cut: for the current assignment, add a
        # per-facility penalty that makes this exact assignment less attractive.
        # Simple Benders cut: c[i] += step * (exact_cost - best_cost) / n
        # applied to the locations used in p_new
        gap = exact_cost - best_cost
        if gap > 0:
            # Penalize the specific (facility, location) pairs in this solution
            # by adding to c[i] for each facility i placed at p_new[i]
            # Use a smoothed version: penalize proportional to gap
            c[np.arange(n)] += step * gap / (n * (1.0 + gap / max(best_cost, 1)))
            # Decay step to converge
            step *= 0.95
            if step < min_step:
                step = min_step
        else:
            # Exact cost is at or below best: accept
            if exact_cost < best_cost:
                best_cost = exact_cost
                best_p = p_new.copy()
            # Reset or reduce penalty to allow exploration
            c *= 0.5
            step = max(step * 1.1, min_step)

        # --- Local improvement: 2-opt swaps on best_p ---
        # Try to improve best_p with pairwise swaps
        improved = True
        max_swaps = 200
        swap_count = 0
        while improved and swap_count < max_swaps and time.time() < deadline:
            improved = False
            swap_count += 1
            # Sample random pairs to try (not all n^2/2 for speed)
            npairs = min(50, n * 5)
            idx = np.random.choice(n, size=(npairs, 2), replace=False)
            for i, j in idx:
                if i == j:
                    continue
                # Compute delta for swapping p[i] and p[j]
                pi, pj = best_p[i], best_p[j]
                # delta = sum_k a[i,k]*(b[pj,k]-b[pi,k]) + sum_k a[j,k]*(b[pi,k]-b[pj,k])
                #       + sum_k a[k,i]*(b[pj,k]-b[pi,k]) + sum_k a[k,j]*(b[pi,k]-b[pj,k])
                # Simplified: recompute cost change
                old = best_cost
                best_p[i], best_p[j] = pj, pi
                new = _cost(best_p, a, b)
                if new < old:
                    best_cost = new
                    improved = True
                else:
                    best_p[i], best_p[j] = pi, pj

        # --- LAP-guided perturbation: re-solve master with perturbed costs ---
        if time.time() < deadline:
            # Perturb surrogate cost with small random noise to escape local minima
            noise = np.random.randn(n, n) * (best_cost / n) * 0.01
            surrogate_perturbed = B @ (a.T) + np.outer(c, np.arange(n, dtype=np.float64)) + noise
            p_pert, _ = linear_sum_assignment(surrogate_perturbed)
            p_pert = p_pert.astype(np.int64)
            pert_cost = _cost(p_pert, a, b)
            if pert_cost < best_cost:
                best_cost = pert_cost
                best_p = p_pert.copy()

        # Check deadline
        if time.time() > deadline - 5.0:
            break

    return best_p.tolist()
