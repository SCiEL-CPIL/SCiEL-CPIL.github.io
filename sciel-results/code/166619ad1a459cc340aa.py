"""Portfolio optimizer for BBOB f19 (Griewank-Rosenbrock) using Multi-Start Nelder-Mead."""
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer with global initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            fx = float(fx)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return x, fx

        def nelder_mead(x0, max_evals, initial_step=0.5):
            """
            Nelder-Mead local search.
            Returns the best point found within max_evals.
            """
            nonlocal calls_made, best_x, best_f, deadline

            # Initial simplex
            n = dim
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(1, n + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += initial_step

            # Evaluate initial simplex
            f_values = []
            for i in range(n + 1):
                if calls_made >= budget or time.time() >= deadline:
                    return best_x, best_f
                _, fi = safe_call(simplex[i])
                f_values.append(fi)
            
            if calls_made >= budget or time.time() >= deadline:
                return best_x, best_f

            f_values = np.array(f_values)
            
            # Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Tolerance for convergence
            tol = 1e-8
            
            while calls_made < budget and time.time() < deadline:
                # Sort simplex by function value
                idx = np.argsort(f_values)
                simplex = simplex[idx]
                f_values = f_values[idx]
                
                # Check convergence: difference between best and worst is small
                if f_values[-1] - f_values[0] < tol:
                    break
                    
                # Centroid of all points except the worst
                x_c = np.mean(simplex[:n], axis=0)
                
                # Reflection
                x_r = x_c + alpha * (x_c - simplex[n])
                x_r = np.clip(x_r, -5.0, 5.0)
                _, f_r = safe_call(x_r)
                
                if f_r < f_values[0]:
                    # Expansion
                    x_e = x_c + gamma * (x_c - simplex[n])
                    x_e = np.clip(x_e, -5.0, 5.0)
                    _, f_e = safe_call(x_e)
                    if f_e < f_r:
                        simplex[n] = x_e
                        f_values[n] = f_e
                    else:
                        simplex[n] = x_r
                        f_values[n] = f_r
                elif f_r < f_values[n-1]:
                    # Accept reflection
                    simplex[n] = x_r
                    f_values[n] = f_r
                else:
                    # Contraction
                    if f_r < f_values[n]:
                        # Outside contraction
                        x_c = x_c + 0.5 * (x_r - x_c)
                    else:
                        # Inside contraction
                        x_c = x_c - 0.5 * (x_r - x_c)
                    
                    x_cc = x_c
                    x_cc = np.clip(x_cc, -5.0, 5.0)
                    _, f_cc = safe_call(x_cc)
                    
                    if f_cc < f_values[n]:
                        simplex[n] = x_cc
                        f_values[n] = f_cc
                    else:
                        # Shrinkage
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            x_s = np.clip(simplex[i], -5.0, 5.0)
                            _, f_s = safe_call(x_s)
                            simplex[i] = x_s
                            f_values[i] = f_s
                            
                # Update global best
                if best_f is None or f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = simplex[0].copy()

                # Check budget
                if calls_made >= budget:
                    break

            return best_x, best_f

        # Phase 1: Global Exploration
        # Use a mix of random and structured points
        rng = np.random.default_rng(42)
        
        # 50 initial points
        # 10 random, 10 uniform grid-like, 30 random
        init_points = []
        
        # Random points
        init_points.extend([rng.uniform(-5, 5, size=dim) for _ in range(10)])
        
        # Structured points (midpoints of edges, corners, center)
        # Center
        init_points.append(np.zeros(dim))
        
        # Corners (sample some)
        corners = []
        for _ in range(20):
            c = rng.choice([-5.0, 5.0], size=dim)
            corners.append(c)
        init_points.extend(corners)
        
        # Midpoints of axes
        for i in range(dim):
            for val in [-2.5, 2.5]:
                p = np.zeros(dim)
                p[i] = val
                init_points.append(p)
                
        # Ensure we have at least 50 points
        while len(init_points) < 50:
            init_points.append(rng.uniform(-5, 5, size=dim))

        # Evaluate initial points
        for p in init_points:
            if calls_made >= budget or time.time() >= deadline:
                break
            _, _ = safe_call(p)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float('inf')

        # Phase 2: Local Refinement with Multi-Start Nelder-Mead
        # Get top K points from the initial evaluation
        # We need to retrieve the function values of the initial points.
        # Since safe_call doesn't store history, we re-evaluate the best few?
        # No, that wastes budget. 
        # Instead, we can just run NM from the global best and a few random restarts.
        # Or, we can keep a list of (x, f) during Phase 1.
        
        # Let's modify Phase 1 to store points
        # Re-implementing Phase 1 with storage is better.
        
        # Reset and do it properly
        calls_made = 0
        best_x = None
        best_f = float('inf')
        history = []

        def safe_call_stored(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            fx = float(fx)
            calls_made += 1
            history.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return x, fx

        # Re-run Phase 1 with storage
        for p in init_points:
            if calls_made >= budget or time.time() >= deadline:
                break
            safe_call_stored(p)

        # Sort history by function value
        history.sort(key=lambda item: item[1])
        
        # Select top K points for NM
        K = 10
        top_points = [h[0] for h in history[:K]]
        
        # Allocate remaining budget to NM
        remaining_budget = budget - calls_made
        if remaining_budget > 0 and time.time() < deadline:
            # Divide budget among top points
            # Give more budget to better points
            # Simple strategy: equal split
            budget_per_start = remaining_budget // K
            
            for i, p in enumerate(top_points):
                if calls_made >= budget or time.time() >= deadline:
                    break
                # Run NM
                # Use a smaller initial step for refined points
                nelder_mead(p, budget_per_start, initial_step=0.5)

        # Phase 3: Final Polish
        if calls_made < budget and time.time() < deadline and best_x is not None:
            # Run one more NM on the global best with remaining budget
            nelder_mead(best_x, budget - calls_made, initial_step=0.1)

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
