"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid optimizer: structured initialization + multi-restart Powell + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        calls_made = 0
        dim = self.dim
        budget = self.budget

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        best_x = np.zeros(dim)
        best_f = np.inf
        # Keep a small archive of top distinct points to enable diverse restarts
        archive = []  # list of (f, x)

        def counted_func(x):
            nonlocal calls_made, best_x, best_f, archive
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                # Add to archive if sufficiently distinct from existing entries
                if len(archive) == 0:
                    archive.append((fx, x.copy()))
                else:
                    is_distinct = True
                    for f_prev, x_prev in archive:
                        if np.linalg.norm(x - x_prev) < 0.5:
                            is_distinct = False
                            break
                    if is_distinct:
                        archive.append((fx, x.copy()))
                        archive.sort(key=lambda t: t[0])
                        if len(archive) > 10:
                            archive.pop()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Structured initialization to find good basins
        # Try origin first (strong prior for unshifted Rastrigin), then sample in [-4, 4]^5
        n_init = 120
        for i in range(n_init):
            if calls_made >= budget or time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            elif i < 5:
                # Try a few structured points near the origin
                x = rng.uniform(-1, 1, dim)
            else:
                x = rng.uniform(-4, 4, dim)
            counted_func(x)

        # Phase 2: Multi-restart Powell local refinement
        # Powell is derivative-free and robust for multimodal functions
        # Use diverse starting points from the archive when available
        max_powell_runs = 8
        used_starts = [best_x.copy()]
        
        for run_idx in range(max_powell_runs):
            if calls_made >= budget or time.time() > deadline - 15:
                break

            # Choose starting point
            if run_idx == 0:
                x_start = best_x.copy()
            elif run_idx < len(archive):
                # Use a point from the archive
                f_a, x_a = archive[run_idx]
                # Check if we've already used this point
                if any(np.linalg.norm(x_a - u) < 0.3 for u in used_starts):
                    # Perturb it
                    x_start = x_a + rng.normal(0, 0.5, dim)
                else:
                    x_start = x_a.copy()
            else:
                # Perturb the best point
                sigma = 0.8 / (run_idx + 1)
                x_start = best_x + rng.normal(0, sigma, dim)
            
            x_start = clip(x_start)
            used_starts.append(x_start.copy())

            try:
                minimize(
                    counted_func,
                    x_start,
                    method='Powell',
                    options={'maxiter': 80, 'maxfev': 80}
                )
            except Exception:
                pass

        # Phase 3: Final polish with Nelder-Mead for high precision
        if calls_made < budget and time.time() < deadline - 5:
            try:
                minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
            except Exception:
                pass
        
        # Phase 4: One more very tight polish if budget allows
        if calls_made < budget and time.time() < deadline - 2:
            try:
                minimize(
                    counted_func,
                    best_x,
                    method='Powell',
                    options={'maxiter': 50, 'maxfev': 50}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
