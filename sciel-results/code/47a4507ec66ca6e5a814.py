# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Ridge-following optimizer for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        budget = self.budget

        state = {
            'calls': 0,
            'best_f': np.inf,
            'best_x': np.zeros(n)
        }

        def wrapped_func(x):
            if state['calls'] >= budget or time.time() > deadline:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            state['calls'] += 1
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)

        # =========================================================
        # Phase 1: Coarse multi-start to locate ridge region
        # =========================================================
        # Try several starting points with different strategies
        candidates = []
        
        # Origin
        candidates.append(np.zeros(n))
        
        # Random points
        for _ in range(8):
            candidates.append(rng.uniform(-4, 4, size=n))
        
        # Axis-aligned points
        for i in range(n):
            for sign in [1, -1]:
                pt = np.zeros(n)
                pt[i] = sign * 3.0
                candidates.append(pt)

        # Evaluate all candidates
        for pt in candidates:
            if state['calls'] >= budget - 50 or time.time() > deadline:
                break
            wrapped_func(pt)

        if state['calls'] >= budget - 50 or time.time() > deadline:
            return state['best_x'], state['best_f']

        # =========================================================
        # Phase 2: Ridge following via adaptive line search
        # =========================================================
        # For each of the top few candidates, do local ridge optimization
        # Strategy: coordinate descent with line search along each axis,
        # then try combined directions
        
        # Get top candidates by function value
        evaluated = []
        for pt in candidates:
            if state['calls'] >= budget - 40:
                break
            # Re-evaluate to get current best (already in state)
        
        # Use the best point as seed for local refinement
        best_x = state['best_x'].copy()
        best_f = state['best_f']

        # Adaptive line search along each coordinate
        def line_search_1d(x0, direction, f0, func, state, deadline, budget, max_iter=20):
            """Golden section / bisection line search along a direction."""
            n = len(x0)
            # Find bracket
            step = 1.0
            # Try to find a better point
            x_plus = np.clip(x0 + step * direction, -5, 5)
            f_plus = func(x_plus)
            if f_plus >= f0:
                # Try negative direction
                x_minus = np.clip(x0 - step * direction, -5, 5)
                f_minus = func(x_minus)
                if f_minus >= f0:
                    return x0, f0, 0
                else:
                    # Negative direction is better
                    a, b = -step, 0.0
                    fa, fb = f_minus, f0
                    direction = -direction
                    x0 = x0  # keep original, we'll search in negative
                    # Actually, let's just do a simple bisection
                    lo, hi = -step, 0.0
                    flo, fhi = f_minus, f0
                    for _ in range(max_iter):
                        if state['calls'] >= budget - 5 or time.time() > deadline:
                            break
                        mid = 0.5 * (lo + hi)
                        x_mid = np.clip(x0 + mid * direction, -5, 5)
                        f_mid = func(x_mid)
                        if f_mid < flo:
                            hi = mid
                            fhi = f_mid
                        else:
                            lo = mid
                            flo = f_mid
                    # Return the best found
                    if fhi < flo:
                        x_best = np.clip(x0 + hi * direction, -5, 5)
                        return x_best, fhi, 1
                    else:
                        x_best = np.clip(x0 + lo * direction, -5, 5)
                        return x_best, flo, 1
            else:
                # Positive direction is better
                lo, hi = 0.0, step
                flo, fhi = f0, f_plus
                for _ in range(max_iter):
                    if state['calls'] >= budget - 5 or time.time() > deadline:
                        break
                    mid = 0.5 * (lo + hi)
                    x_mid = np.clip(x0 + mid * direction, -5, 5)
                    f_mid = func(x_mid)
                    if f_mid < flo:
                        hi = mid
                        fhi = f_mid
                    else:
                        lo = mid
                        flo = f_mid
                if fhi < flo:
                    x_best = np.clip(x0 + hi * direction, -5, 5)
                    return x_best, fhi, 1
                else:
                    x_best = np.clip(x0 + lo * direction, -5, 5)
                    return x_best, flo, 1

        # Coordinate descent with line search
        for iteration in range(30):
            if state['calls'] >= budget - 10 or time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if state['calls'] >= budget - 10 or time.time() > deadline:
                    break
                direction = np.zeros(n)
                direction[i] = 1.0
                x_new, f_new, _ = line_search_1d(
                    best_x, direction, best_f, wrapped_func, state, deadline, budget
                )
                if f_new < best_f - 1e-10:
                    best_x = x_new
                    best_f = f_new
                    improved = True
                    if f_new < state['best_f']:
                        state['best_f'] = f_new
                        state['best_x'] = x_new.copy()
            if not improved:
                # Try smaller steps
                break

        # =========================================================
        # Phase 3: Multi-directional local search from best point
        # =========================================================
        if state['calls'] < budget - 10:
            # Try random directions from the best point
            for _ in range(20):
                if state['calls'] >= budget - 5 or time.time() > deadline:
                    break
                direction = rng.standard_normal(n)
                direction /= np.linalg.norm(direction)
                x_new, f_new, _ = line_search_1d(
                    state['best_x'], direction, state['best_f'], 
                    wrapped_func, state, deadline, budget, max_iter=15
                )
                if f_new < state['best_f'] - 1e-10:
                    state['best_f'] = f_new
                    state['best_x'] = x_new.copy()

        # =========================================================
        # Phase 4: Fine-tuning with very small steps
        # =========================================================
        if state['calls'] < budget - 5:
            for _ in range(10):
                if state['calls'] >= budget - 2 or time.time() > deadline:
                    break
                for i in range(n):
                    if state['calls'] >= budget - 2 or time.time() > deadline:
                        break
                    direction = np.zeros(n)
                    direction[i] = 1.0
                    # Very small step
                    for step in [0.01, 0.001, 0.0001]:
                        if state['calls'] >= budget - 2 or time.time() > deadline:
                            break
                        x_try = np.clip(state['best_x'] + step * direction, -5, 5)
                        f_try = wrapped_func(x_try)
                        if f_try < state['best_f'] - 1e-12:
                            state['best_f'] = f_try
                            state['best_x'] = x_try.copy()
                            break
                        x_try = np.clip(state['best_x'] - step * direction, -5, 5)
                        f_try = wrapped_func(x_try)
                        if f_try < state['best_f'] - 1e-12:
                            state['best_f'] = f_try
                            state['best_x'] = x_try.copy()
                            break

        return state['best_x'], state['best_f']
# EVOLVE-BLOCK-END
