import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for the Griewank-Rosenbrock function (f19).
    Uses a multi-start strategy with L-BFGS-B and Nelder-Mead refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        # Leave a safety margin of 20 seconds before the 300s hard kill
        deadline = start_time + 280.0

        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.shape[0] != dim:
                x = np.resize(x, dim)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = np.inf
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def time_ok():
            return time.time() < deadline and calls_made < budget

        def nelder_mead(x0, f0, max_iter=50, initial_step=0.5):
            """
            Custom Nelder-Mead implementation to avoid scipy overhead 
            and allow tighter control over calls.
            """
            nonlocal calls_made, best_x, best_f
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0.copy()
            fvals = np.zeros(dim + 1)
            fvals[0] = f0

            # Initialize simplex vertices
            for i in range(1, dim + 1):
                if not time_ok():
                    return x0, f0
                simplex[i] = x0.copy()
                simplex[i][i - 1] += initial_step
                fvals[i] = safe_call(simplex[i])

            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrink

            for _ in range(max_iter):
                if not time_ok():
                    break

                # Sort vertices by fitness
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Check convergence
                if np.std(fvals) < 1e-10 * (1.0 + abs(fvals[0])):
                    break

                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = safe_call(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = safe_call(xe)
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1], fvals[-1] = xr, fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (simplex[-1] - centroid)
                    fc = safe_call(xc)

                    if fc < min(fr, fvals[-1]):
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if not time_ok():
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = safe_call(simplex[i])

            order = np.argsort(fvals)
            return simplex[order[0]], fvals[order[0]]

        # Phase 1: Diverse Initial Sampling
        # We use a mix of structured and random points to cover the space.
        rng = np.random.default_rng(42)
        
        initial_points = []
        
        # Origin
        initial_points.append(np.zeros(dim))
        
        # Points along axes (Griewank/Rosenbrock often have structure along axes)
        for d in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                p = np.zeros(dim)
                p[d] = val
                initial_points.append(p)
        
        # Random points to fill gaps
        # Target ~25 initial points total
        num_random = 25 - len(initial_points)
        if num_random > 0:
            random_pts = rng.uniform(-4.5, 4.5, size=(num_random, dim))
            initial_points.extend(random_pts)

        # Evaluate initial points
        for p in initial_points:
            if not time_ok():
                break
            safe_call(p)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        # Phase 2: Multi-start L-BFGS-B
        # Select the best few points from Phase 1 as starting points for local optimization.
        # We need to track which points were good. Since safe_call updates best_x/best_f globally,
        # we should have stored the points. Let's re-evaluate or just use the current best 
        # and a few random restarts if budget allows.
        
        # Actually, let's keep a list of evaluated points and their scores to pick top K.
        # The current safe_call doesn't return a history. 
        # Let's modify strategy: We will run L-BFGS-B from the current best, 
        # and also from a few other random points to escape local minima.
        
        # To be more efficient, let's just run L-BFGS-B from the current best_x.
        # If it gets stuck, we might try a few more.
        
        start_points = [best_x.copy()]
        
        # Add a few random restarts to ensure we don't miss a better basin
        # We have used ~25 calls. We have ~475 left.
        # L-BFGS-B is fast. Let's try 5 restarts.
        
        # Generate 4 additional diverse starting points
        for _ in range(4):
            if not time_ok():
                break
            p = rng.uniform(-4.5, 4.5, dim)
            # Ensure it's not too close to the current best
            if np.linalg.norm(p - best_x) > 1.0:
                start_points.append(p)

        for sp in start_points:
            if not time_ok():
                break
            
            # Wrap func for scipy
            def wrapped_func(x):
                return safe_call(x)
            
            try:
                # L-BFGS-B is good for bounded optimization
                # maxfun limits the number of function evaluations per run
                # We want to leave budget for other restarts and polishing
                res = minimize(wrapped_func, sp, method='L-BFGS-B', 
                               bounds=[(-5.0, 5.0)] * dim,
                               options={'maxiter': 100, 'maxfun': 50, 'gtol': 1e-6})
                
                # safe_call already updates best_x and best_f if res.fun is better
                # But we should explicitly check if res.x is valid and better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Nelder-Mead Refinement
        # Use Nelder-Mead to polish the best result found by L-BFGS-B.
        # Nelder-Mead is derivative-free and can be more robust in tight regions.
        if time_ok() and best_x is not None:
            # Use a smaller initial step for polishing
            x_nm, f_nm = nelder_mead(best_x, best_f, max_iter=30, initial_step=0.2)
            if f_nm < best_f:
                best_x, best_f = x_nm, f_nm

        # Phase 4: Final Coordinate Descent Polish
        # Very fine-tuned coordinate descent to squeeze out the last bits of accuracy.
        if time_ok() and best_x is not None:
            x_cd, f_cd = best_x, best_f
            step_sizes = [0.1, 0.01, 0.001]
            for step in step_sizes:
                if not time_ok():
                    break
                improved = True
                while improved and time_ok():
                    improved = False
                    for d in range(dim):
                        for sign in (1.0, -1.0):
                            cand = x_cd.copy()
                            cand[d] = np.clip(cand[d] + sign * step, -5.0, 5.0)
                            fc = safe_call(cand)
                            if fc < f_cd - 1e-12:
                                x_cd = cand
                                f_cd = fc
                                improved = True
                                if f_cd < best_f:
                                    best_x = x_cd.copy()
                                    best_f = f_cd
                                break # Move to next dimension if improved

        if best_x is None:
            best_x = np.zeros(dim)
            
        return best_x, best_f
