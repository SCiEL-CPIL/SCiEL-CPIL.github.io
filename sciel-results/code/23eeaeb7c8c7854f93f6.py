"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid (1+1)-ES with Nelder-Mead local refinement and deadline guard for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 30s safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Diversified initialization
        n_init = min(12, budget)
        candidates = []
        # Origin
        candidates.append(np.zeros(dim))
        # Axis-aligned points
        for i in range(dim):
            c = np.zeros(dim)
            c[i] = rng.choice([-3.0, 3.0])
            candidates.append(c)
        # Random points
        while len(candidates) < n_init:
            candidates.append(rng.uniform(-4.0, 4.0, dim))

        best_x = None
        best_f = np.inf
        calls_made = 0
        for c in candidates:
            if calls_made >= budget or time.time() > deadline:
                break
            c = np.clip(c, -5, 5)
            fc = func(c)
            calls_made += 1
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: Alternating (1+1)-ES and Nelder-Mead refinement
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        success_rate = 0.3
        nm_interval = 30  # Run NM every 30 ES steps
        evals_in_phase = 0
        nm_fail_count = 0
        max_nm_fails = 3

        while calls_made < budget and time.time() < deadline:
            # Check for deadline
            if time.time() > deadline:
                break

            # Periodically run Nelder-Mead for local refinement
            if evals_in_phase % nm_interval == 0 and evals_in_phase > 0 and nm_fail_count < max_nm_fails:
                # Only run NM if we've made some progress (not stuck)
                if success_rate > 0.15 or evals_in_phase < 50:
                    try:
                        # Create a wrapper that counts calls and respects deadline
                        nm_calls = [0]
                        
                        def nm_func(x_vec):
                            nm_calls[0] += 1
                            if time.time() > deadline or calls_made + nm_calls[0] >= budget:
                                return np.inf
                            return func(np.clip(x_vec, -5, 5))
                        
                        # Run Nelder-Mead with tight tolerances
                        result = minimize(
                            nm_func, 
                            x, 
                            method='Nelder-Mead',
                            options={
                                'maxiter': 15,
                                'xatol': 1e-6,
                                'fatol': 1e-8,
                                'adaptive': True
                            }
                        )
                        
                        calls_made += nm_calls[0]
                        
                        if result.fun < fx and np.isfinite(result.fun):
                            x = np.clip(result.x, -5, 5)
                            fx = result.fun
                            nm_fail_count = 0
                        else:
                            nm_fail_count += 1
                    
                    except Exception:
                        nm_fail_count += 1
                    evals_in_phase += 1
                    continue

            # (1+1)-ES step
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1
            evals_in_phase += 1

            if fy < fx:
                success_rate = 0.7 * success_rate + 0.3 * 1.0
                x = y
                fx = fy
                # Increase sigma if success rate is high
                if success_rate > 0.5:
                    sigma *= 1.15
                else:
                    sigma *= 1.03
            else:
                success_rate = 0.7 * success_rate + 0.3 * 0.0
                # Decrease sigma if success rate is low
                if success_rate < 0.3:
                    sigma *= 0.88
                else:
                    sigma *= 0.96

            # Prevent sigma from getting too small or too large
            sigma = np.clip(sigma, 1e-7, 8.0)

            # Check if stuck — try a larger jump
            if evals_in_phase % 80 == 0 and success_rate < 0.08:
                # Try a random restart from current best
                jump = x + sigma * 2.5 * rng.standard_normal(dim)
                jump = np.clip(jump, -5, 5)
                if time.time() > deadline or calls_made >= budget:
                    break
                fj = func(jump)
                calls_made += 1
                evals_in_phase += 1
                if fj < fx:
                    x = jump
                    fx = fj
                    success_rate = 0.4
                    nm_fail_count = 0

        return x, fx
# EVOLVE-BLOCK-END
