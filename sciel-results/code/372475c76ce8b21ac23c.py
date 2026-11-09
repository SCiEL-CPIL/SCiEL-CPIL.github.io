"""Optimized program for Bent Cigar (f12) using Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead optimizer with box constraints and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit

        # Parameters for Nelder-Mead
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage

        # Initial simplex setup
        # We start at a random point and create a simplex with small perturbations
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-5, 5, self.dim)
        
        # Initialize simplex vertices
        # v0 is x0, v1...vd are x0 + delta * e_i
        delta = 0.5
        simplex = np.zeros((self.dim + 1, self.dim))
        simplex[0] = x0
        for i in range(1, self.dim + 1):
            simplex[i] = x0.copy()
            simplex[i][i-1] += delta

        # Evaluate initial simplex
        fvals = np.array([func(x) for x in simplex])
        best_x = simplex[0]
        best_f = fvals[0]
        
        # Update best if not at 0
        idx = np.argmin(fvals)
        if fvals[idx] < best_f:
            best_f = fvals[idx]
            best_x = simplex[idx]

        calls = self.dim + 1
        budget_remaining = self.budget - calls

        # Main Nelder-Mead loop
        while budget_remaining > 0:
            # Check deadline
            if time.time() > deadline:
                break

            # Sort simplex by function value
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]

            # Update global best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

            # Check convergence (optional, but good for stopping early if stuck)
            # If the spread of function values is very small, we might be converged
            if np.max(fvals) - np.min(fvals) < 1e-10 * (1 + np.abs(np.min(fvals))):
                break

            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)

            # 1. Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            budget_remaining -= 1

            if fr < fvals[0]:
                # 2. Expansion
                if time.time() > deadline or budget_remaining <= 0:
                    break
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                budget_remaining -= 1

                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Simple reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # 3. Contraction
                if time.time() > deadline or budget_remaining <= 0:
                    break
                # Inside contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                budget_remaining -= 1

                if fc < fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # 4. Shrinkage
                    if time.time() > deadline or budget_remaining < self.dim:
                        break
                    for i in range(1, self.dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        budget_remaining -= 1
                        if time.time() > deadline or budget_remaining <= 0:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
