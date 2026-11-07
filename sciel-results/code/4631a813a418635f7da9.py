"""Combined optimizer for BBOB f19 (Griewank-Rosenbrock).

Uses a Sobol quasi-random sequence for systematic exploration, then
Nelder-Mead local refinement to exploit the Rosenbrock valley structure.
Includes a wall-clock deadline guard.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from math import exp, sqrt, log
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def sample(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Dense Sobol quasi-random sequence for systematic exploration
        # Use most of the budget for global exploration
        n_sobol = min(budget - 50, 400)  # Reserve 50 for local refinement
        if n_sobol < 10:
            n_sobol = 10

        # Generate Sobol points in [0,1]^d, then map to [-5,5]^d
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_pts_unit = sobol.random(n=n_sobol)  # shape (n_sobol, dim)
        sobol_pts = -5.0 + 10.0 * sobol_pts_unit  # map to [-5, 5]

        # Evaluate Sobol points
        for i in range(n_sobol):
            if time.time() >= deadline or call_count >= budget:
                break
            x = sobol_pts[i]
            sample(x)

        # Phase 2: Nelder-Mead local refinement from multiple good starting points
        # Sort by function value and pick top candidates for NM refinement
        if best_x is not None and time.time() < deadline and call_count < budget:
            # Collect all evaluated points sorted by fitness
            # We need to track them - let's re-evaluate top candidates
            # Actually, let's use the Sobol points we already evaluated
            # Find the best few Sobol points and run NM from each

            # Re-evaluate: we need to store xs and ys
            # Let's restart tracking properly
            pass

        # Since we didn't track all points, let's do a simpler approach:
        # Run NM from the best point found, and from a few other promising regions
        if best_x is not None and time.time() < deadline and call_count < budget:
            # NM refinement from best point
            try:
                nm_res = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'adaptive': True,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                    }
                )
                nm_x = np.clip(nm_res.x, -5.0, 5.0)
                nm_f = float(func(nm_x))
                call_count += 1
                if nm_f < best_f:
                    best_f = nm_f
                    best_x = nm_x.copy()
            except Exception:
                pass

        # Phase 3: Additional NM refinements from other promising areas
        # Use remaining budget for more local searches from diverse starts
        if time.time() < deadline and call_count < budget:
            # Generate a few more diverse starting points near good regions
            rng = np.random.default_rng(123)
            n_extra = min(5, (budget - call_count) // 20)
            for _ in range(n_extra):
                if time.time() >= deadline or call_count >= budget:
                    break
                # Start near best with some perturbation
                x0 = np.clip(best_x + 2.0 * rng.standard_normal(dim), -5.0, 5.0)
                try:
                    nm_res = minimize(
                        func,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 50,
                            'adaptive': True,
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                        }
                    )
                    nm_x = np.clip(nm_res.x, -5.0, 5.0)
                    nm_f = float(func(nm_x))
                    call_count += 1
                    if nm_f < best_f:
                        best_f = nm_f
                        best_x = nm_x.copy()
                except Exception:
                    pass

        # Phase 4: Fine-tune with (1+1)-ES if budget remains
        if best_x is not None and time.time() < deadline and call_count < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            while time.time() < deadline and call_count < budget:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = sample(y)
                if call_count >= budget or time.time() >= deadline:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                    if sigma < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
