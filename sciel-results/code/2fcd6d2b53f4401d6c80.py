import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = None
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        def wrapper(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            return func(x)

        rng = np.random.default_rng(42)

        # Phase 1: Initial global search with Halton sequence for better coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    points[i, j] = halton(i + 1, j + 2)
            return points

        global_calls = 15
        halton_points = halton_sequence(global_calls, dim)
        halton_points = halton_points * 10 - 5

        for x in halton_points:
            if calls_made >= total_budget or time.time() > deadline:
                break
            x = np.clip(x, -5, 5)
            try:
                fx = wrapper(x)
            except TimeoutError:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = wrapper(best_x)
            except TimeoutError:
                return np.zeros(dim), float('inf')

        # Phase 2: Coordinate-wise Adaptive Step Size (CMA-ES-lite) with periodic gradient polish
        x = best_x.copy()
        fx = best_f
        # Independent step size per dimension
        sigma = np.ones(dim) * 1.5
        
        # Track success rate per dimension for adaptive step sizes
        # Use a sliding window of recent successes/failures per coordinate
        window_size = 20
        success_counts = np.zeros(dim)  # number of successes in window
        trial_counts = np.zeros(dim)    # number of trials in window

        nm_interval = 40  # Do gradient polish every N ES iterations
        es_count = 0
        polish_maxfev = 60  # Max function evaluations for polish

        while calls_made < total_budget and time.time() < deadline:
            # Check if we should do gradient polish
            if es_count > 0 and es_count % nm_interval == 0 and (total_budget - calls_made) > polish_maxfev + 5:
                # Finite-difference gradient estimation and local polish
                try:
                    # Estimate gradient at best_x using central differences
                    h = 1e-5
                    grad = np.zeros(dim)
                    for i in range(dim):
                        if calls_made >= total_budget or time.time() > deadline:
                            break
                        xp = best_x.copy()
                        xm = best_x.copy()
                        xp[i] += h
                        xm[i] -= h
                        xp = np.clip(xp, -5, 5)
                        xm = np.clip(xm, -5, 5)
                        try:
                            fxp = wrapper(xp)
                            fxm = wrapper(xm)
                            grad[i] = (fxp - fxm) / (2 * h)
                        except TimeoutError:
                            raise
                        except Exception:
                            grad[i] = 0.0
                    
                    # Use Nelder-Mead for final polish, but start from best_x
                    # with a small initial simplex
                    step_size = 0.1
                    simplex = np.zeros((dim + 1, dim))
                    simplex[0] = best_x
                    for i in range(dim):
                        simplex[i + 1] = best_x
                        simplex[i + 1][i] += step_size
                    
                    res = minimize(wrapper, best_x, method='Nelder-Mead',
                                   options={
                                       'maxfev': polish_maxfev,
                                       'xatol': 1e-8,
                                       'fatol': 1e-8,
                                       'adaptive': True,
                                       'initial_simplex': simplex
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                        x = best_x.copy()
                        fx = best_f
                        # Reset sigma after polish (it found a better point)
                        sigma = np.ones(dim) * 0.3
                        # Reset success tracking
                        success_counts = np.zeros(dim)
                        trial_counts = np.zeros(dim)
                except TimeoutError:
                    break
                except Exception:
                    pass

            if time.time() > deadline or calls_made >= total_budget:
                break

            # 1+1-ES step with coordinate-wise adaptive step sizes
            # For each coordinate, decide whether to move based on adaptive sigma
            y = x.copy()
            moved = False
            improvement = 0.0
            
            for i in range(dim):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                # Try moving in coordinate i
                delta = sigma[i] * rng.standard_normal()
                y_trial = x.copy()
                y_trial[i] += delta
                y_trial = np.clip(y_trial, -5, 5)
                
                try:
                    fy_trial = wrapper(y_trial)
                except TimeoutError:
                    break
                
                trial_counts[i] += 1
                
                if fy_trial < fx:
                    # Success: move and expand
                    x[i] = y_trial[i]
                    fx = fy_trial
                    sigma[i] *= 1.1
                    success_counts[i] += 1
                    moved = True
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Failure: shrink
                    sigma[i] *= 0.9

            # Update adaptive step sizes based on success rate
            for i in range(dim):
                if trial_counts[i] >= window_size:
                    success_rate = success_counts[i] / trial_counts[i]
                    # If success rate is high, expand; if low, shrink
                    if success_rate > 0.6:
                        sigma[i] *= 1.05
                    elif success_rate < 0.4:
                        sigma[i] *= 0.95
                    # Reset counters
                    success_counts[i] = 0
                    trial_counts[i] = 0

            # Clamp sigma
            sigma = np.clip(sigma, 0.01, 5.0)
            es_count += 1

        return best_x, best_f
