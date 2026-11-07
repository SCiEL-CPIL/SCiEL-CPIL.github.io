import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        max_calls = self.budget
        calls_made = 0
        dim = self.dim
        best_x = None
        best_f = np.inf

        # Wrapper that counts calls, clips, and tracks best
        def eval_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= max_calls:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        starts = [np.zeros(dim)]
        # Random points spread across the box
        for _ in range(10):
            starts.append(rng.uniform(-4.5, 4.5, size=dim))
        # Structured points: all-ones, all-negatives, axis-aligned
        starts.append(np.ones(dim) * 1.0)
        starts.append(np.ones(dim) * -1.0)
        for i in range(dim):
            pt = np.zeros(dim)
            pt[i] = 2.0
            starts.append(pt)
            pt2 = np.zeros(dim)
            pt2[i] = -2.0
            starts.append(pt2)

        # Shuffle to get diversity, but keep origin first for quick initial best
        rng.shuffle(starts[1:])
        starts = [starts[0]] + starts[1:]

        # Phase 1: Multi-start L-BFGS-B with small per-start budgets
        # L-BFGS-B in 5D typically needs 20-50 evals to converge in the valley
        # Use ~20-30 evals per start to cover many basins
        num_starts = min(len(starts), 15)
        
        for idx in range(num_starts):
            if time.time() > deadline:
                break
            if calls_made >= max_calls:
                break
            
            start = starts[idx]
            
            # Small budget per start for broad coverage
            remaining = max_calls - calls_made
            if remaining < 15:
                break
            
            alloc = min(30, remaining)
            
            nm_calls = 0
            nm_limit = alloc

            def lbfgs_func(z):
                nonlocal nm_calls
                if nm_calls >= nm_limit:
                    return np.inf
                nm_calls += 1
                return func(np.clip(z, -5.0, 5.0))

            try:
                res = minimize(lbfgs_func, start, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0)] * dim,
                               options={'maxiter': max(1, nm_limit // 2),
                                        'maxfun': nm_limit,
                                        'ftol': 1e-10,
                                        'gtol': 1e-6})
                # Update best from all evaluations made
                if nm_calls > 0:
                    # The eval_func wasn't used here, so manually check
                    # Actually we need to track best from these calls
                    # Let's re-evaluate: lbfgs_func doesn't update best_f
                    # We need to fix this
                    pass
            except Exception:
                pass

        # The above approach has a bug: lbfgs_func doesn't update best_f/best_x
        # Let me redo this properly using eval_func which tracks best

        # Reset and do it properly
        calls_made = 0
        best_x = None
        best_f = np.inf

        for idx in range(num_starts):
            if time.time() > deadline:
                break
            if calls_made >= max_calls:
                break
            
            start = starts[idx]
            
            remaining = max_calls - calls_made
            if remaining < 15:
                break
            
            alloc = min(30, remaining)
            
            nm_calls = 0
            nm_limit = alloc

            def lbfgs_func(z):
                nonlocal nm_calls
                if nm_calls >= nm_limit:
                    return np.inf
                nm_calls += 1
                x_clip = np.clip(z, -5.0, 5.0)
                val = func(x_clip)
                # Track best
                if val < best_f:
                    best_f = val
                    best_x = x_clip.copy()
                return val

            try:
                res = minimize(lbfgs_func, start, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0)] * dim,
                               options={'maxiter': max(1, nm_limit // 2),
                                        'maxfun': nm_limit,
                                        'ftol': 1e-10,
                                        'gtol': 1e-6})
            except Exception:
                pass

        # Phase 2: Polish the best point found with a larger budget
        if best_x is not None and calls_made < max_calls and time.time() < deadline:
            remaining = max_calls - calls_made
            if remaining > 20:
                nm_calls = 0
                nm_limit = remaining

                def polish_func(z):
                    nonlocal nm_calls
                    if nm_calls >= nm_limit:
                        return np.inf
                    nm_calls += 1
                    x_clip = np.clip(z, -5.0, 5.0)
                    val = func(x_clip)
                    if val < best_f:
                        best_f = val
                        best_x = x_clip.copy()
                    return val

                try:
                    res = minimize(polish_func, best_x, method='L-BFGS-B',
                                   bounds=[(-5.0, 5.0)] * dim,
                                   options={'maxiter': max(1, nm_limit // 2),
                                            'maxfun': nm_limit,
                                            'ftol': 1e-15,
                                            'gtol': 1e-10})
                except Exception:
                    pass

        # Phase 3: If still have budget, try a few more random restarts from best
        if best_x is not None and calls_made < max_calls and time.time() < deadline:
            remaining = max_calls - calls_made
            if remaining > 30:
                # Perturb best point and try L-BFGS-B
                for attempt in range(3):
                    if time.time() > deadline or calls_made >= max_calls:
                        break
                    
                    perturbation = rng.normal(0, 0.5, size=dim)
                    start = np.clip(best_x + perturbation, -5.0, 5.0)
                    
                    alloc = min(remaining - calls_made, 40)
                    if alloc < 10:
                        break
                    
                    nm_calls = 0
                    nm_limit = alloc

                    def restart_func(z):
                        nonlocal nm_calls
                        if nm_calls >= nm_limit:
                            return np.inf
                        nm_calls += 1
                        x_clip = np.clip(z, -5.0, 5.0)
                        val = func(x_clip)
                        if val < best_f:
                            best_f = val
                            best_x = x_clip.copy()
                        return val

                    try:
                        res = minimize(restart_func, start, method='L-BFGS-B',
                                       bounds=[(-5.0, 5.0)] * dim,
                                       options={'maxiter': max(1, nm_limit // 2),
                                                'maxfun': nm_limit,
                                                'ftol': 1e-12,
                                                'gtol': 1e-8})
                    except Exception:
                        pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = eval_func(best_x)

        return best_x, best_f
