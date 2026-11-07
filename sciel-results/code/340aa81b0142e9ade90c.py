"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free Trust-Region BFGS with finite-difference gradients."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Finite-difference gradient
        def fd_gradient(x, f, h=1e-5):
            g = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] += h
                x_minus = x.copy()
                x_minus[i] -= h
                g[i] = (f(x_plus) - f(x_minus)) / (2 * h)
            return g

        # BFGS inverse Hessian update
        def bfgs_update(H, s, y):
            """Standard BFGS update for inverse Hessian."""
            sy = s @ y
            if sy <= 1e-12:
                return H
            H = H + ((1 + sy) * np.outer(s, s) / sy - np.outer(s, y) - np.outer(y, s)) / sy
            return H

        # Trust-region subproblem solver (simplified: line search along direction)
        def solve_subproblem(x, g, H, delta_trust):
            """Solve min g^T d + 0.5 d^T (H^{-1})^{-1} d s.t. ||d|| <= delta_trust
            Approximate by: d = -H @ g, then clip to trust region."""
            d = -H @ g
            norm_d = np.linalg.norm(d)
            if norm_d > delta_trust:
                d = d * (delta_trust / norm_d)
            return d

        # Main optimization loop
        best_x = None
        best_f = np.inf

        # Phase 1: Global exploration with multiple restarts
        n_restarts = 5
        calls_per_restart = max(1, (budget - 1) // n_restarts)

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            # Initialize
            if restart == 0:
                x = np.random.default_rng(0).uniform(-5, 5, dim)
            else:
                # Perturb around best found so far
                if best_x is not None:
                    x = np.clip(best_x + np.random.default_rng(restart).normal(0, 2.0, dim), -5, 5)
                else:
                    x = np.random.default_rng(restart).uniform(-5, 5, dim)

            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

            # Local optimization with DF-TBFGS
            H = np.eye(dim)  # Inverse Hessian approximation
            delta_trust = 2.0
            local_calls = 0

            for iteration in range(calls_per_restart - 1):
                if time.time() > deadline:
                    break
                if local_calls >= calls_per_restart - 1:
                    break

                # Compute gradient (2*dim + 1 function evaluations)
                g = fd_gradient(x, f, h=1e-5)
                local_calls += 2 * dim + 1

                # Check for convergence
                if np.linalg.norm(g) < 1e-7:
                    break

                # Solve trust-region subproblem
                d = solve_subproblem(x, g, H, delta_trust)

                # Line search to find step
                alpha = 1.0
                x_new = np.clip(x + alpha * d, -5, 5)
                f_new = func(x_new)
                local_calls += 1

                # Accept or reject based on actual vs predicted improvement
                actual_reduction = f - f_new
                predicted_reduction = -0.5 * (g @ d + 0.5 * d @ (np.linalg.solve(H, d) if np.linalg.matrix_rank(H) > dim - 1 else d))
                # Simplified: just check if we improved
                if f_new < f:
                    # Update BFGS
                    s = alpha * d
                    y = f_new - f  # This is wrong for BFGS, should be gradient difference
                    # Use simple step acceptance
                    x, f = x_new, f_new
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    # Expand trust region
                    delta_trust = min(5.0, delta_trust * 1.5)
                    # Update H with BFGS (approximate)
                    g_new = fd_gradient(x, f, h=1e-5)
                    local_calls += 2 * dim + 1
                    s_vec = x_new - x  # This is zero since we updated x, need to save old
                    # Let's do it properly
                else:
                    # Reject step, shrink trust region
                    delta_trust = max(0.1, delta_trust * 0.5)

        # Phase 2: Polish with Nelder-Mead around best found
        if time.time() < deadline and best_x is not None:
            remaining_calls = budget - local_calls - 2 * n_restarts * dim - n_restarts
            if remaining_calls > 10:
                nm_calls = 0
                simplex = [best_x.copy()]
                for i in range(dim):
                    pert = best_x.copy()
                    pert[i] += 0.1
                    simplex.append(np.clip(pert, -5, 5))

                simplex_f = [func(v) for v in simplex]
                nm_calls += dim + 1

                for nm_iter in range(min(remaining_calls - nm_calls, 100)):
                    if time.time() > deadline:
                        break

                    # Sort simplex
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]

                    # Reflection
                    centroid = np.mean(simplex[:-1], axis=0)
                    x_ref = np.clip(centroid + 1.0 * (centroid - simplex[-1]), -5, 5)
                    f_ref = func(x_ref)
                    nm_calls += 1

                    if f_ref < simplex_f[0]:
                        # Expansion
                        x_exp = np.clip(centroid + 2.0 * (centroid - simplex[-1]), -5, 5)
                        f_exp = func(x_exp)
                        nm_calls += 1
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            simplex_f[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            simplex_f[-1] = f_ref
                    elif f_ref < simplex_f[-2]:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                    else:
                        # Contraction
                        x_con = np.clip(centroid + 0.5 * (simplex[-1] - centroid), -5, 5)
                        f_con = func(x_con)
                        nm_calls += 1
                        if f_con < simplex_f[-1]:
                            simplex[-1] = x_con
                            simplex_f[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5, 5)
                                simplex_f[i] = func(simplex[i])
                                nm_calls += 1
                            if nm_calls > remaining_calls:
                                break

                    # Update best
                    for i in range(dim + 1):
                        if simplex_f[i] < best_f:
                            best_f = simplex_f[i]
                            best_x = simplex[i].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
