"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-start LHS + (1+1)-ES with adaptive sigma, multi-candidate refinement,
    and final scipy polish for BBOB f21 (Gallagher 101 peaks).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            try:
                val = float(func(x))
            except Exception:
                return None
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # Phase 1: LHS global sampling for diverse starting points
        # Use ~15% of budget for exploration (75 points max, min 20)
        n_starts = max(20, min(75, budget // 7))
        
        # Latin Hypercube Sampling
        lhs_points = []
        for i in range(n_starts):
            x = np.empty(dim)
            for d in range(dim):
                perm = rng.permutation(n_starts)
                x[d] = -5.0 + 10.0 * (perm[i] + rng.random()) / n_starts
            x = np.clip(x, -5, 5)
            lhs_points.append(x)

        # Evaluate LHS points, keep track of top candidates
        candidates = []  # list of (f, x)
        for x in lhs_points:
            if not check_time() or calls_made >= budget:
                break
            fx = safe_call(x)
            if fx is not None:
                candidates.append((fx, x.copy()))

        if not candidates:
            # Fallback
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is not None:
                best_f = fx
                best_x = x.copy()
                candidates.append((fx, x.copy()))

        if best_x is None:
            return best_x, best_f

        # Sort candidates by fitness, keep top K for multi-start refinement
        candidates.sort(key=lambda t: t[0])
        K = min(5, len(candidates))  # Keep top 5 candidates
        top_candidates = [(f, x.copy()) for f, x in candidates[:K]]

        # Phase 2: Multi-candidate (1+1)-ES refinement (round-robin)
        # Each candidate gets its own sigma and state
        cand_states = []
        for f, x in top_candidates:
            cand_states.append({
                'x': x,
                'f': f,
                'sigma': 1.0,
                'no_improve': 0,
                'success_count': 0,
                'total_steps': 0
            })

        # Round-robin refinement
        max_no_improve = 150
        window = 10

        while check_time() and calls_made < budget:
            # Try each candidate in round-robin
            for ci, state in enumerate(cand_states):
                if not check_time() or calls_made >= budget:
                    break

                x = state['x']
                fx = state['f']
                sigma = state['sigma']

                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break

                state['total_steps'] += 1

                if fy < fx:
                    state['x'] = y.copy()
                    state['f'] = fy
                    state['success_count'] += 1
                    state['no_improve'] = 0
                    sigma *= 1.1
                else:
                    sigma *= 0.95
                    state['no_improve'] += 1

                # Adaptive sigma adjustment based on success rate over a window
                if state['total_steps'] >= window:
                    success_rate = state['success_count'] / state['total_steps']
                    if success_rate > 0.25:
                        sigma *= 1.15
                    elif success_rate < 0.15:
                        sigma *= 0.85
                    state['success_count'] = 0
                    state['total_steps'] = 0

                # Clamp sigma
                sigma = max(sigma, 1e-6)
                sigma = min(sigma, 5.0)
                state['sigma'] = sigma

                # Stagnation handling: if no improvement for a long time, restart locally
                if state['no_improve'] > max_no_improve:
                    jump_size = max(sigma * 2.0, 0.5)
                    new_x = np.clip(x + jump_size * rng.standard_normal(dim), -5, 5)
                    new_fy = safe_call(new_x)
                    if new_fy is None:
                        break
                    state['x'] = new_x.copy()
                    state['f'] = new_fy
                    state['sigma'] = 1.0
                    state['no_improve'] = 0
                    state['total_steps'] = 0
                    state['success_count'] = 0

        # Phase 3: Final polish using scipy.optimize on the best point
        if best_x is not None and check_time() and calls_made < budget:
            try:
                def wrapper(x):
                    if calls_made >= budget:
                        return best_f  # Return best known to stop
                    val = float(func(np.clip(x, -5, 5)))
                    calls_made += 1
                    return val

                # Use Nelder-Mead for final polish
                res = minimize(
                    wrapper,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
