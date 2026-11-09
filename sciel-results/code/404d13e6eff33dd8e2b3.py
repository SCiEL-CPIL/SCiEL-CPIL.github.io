"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead Simplex Method with Global Scatter and Time Guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.time_limit = 320.0  # Safety margin before 330s kill

    def __call__(self, func):
        start_time = time.time()
        calls = 0
        dim = self.dim
        budget = self.budget
        
        # State tracking
        best_x = None
        best_f = np.inf
        history = []
        
        # 1. Global Scatter Phase: Evaluate a deterministic grid + random points
        # This ensures early coverage for AOCC
        # We use a small set of points to get a rough landscape view
        n_global = min(50, budget // 5)
        
        # Deterministic corners and center
        points = []
        # Center
        points.append(np.zeros(dim))
        # Corners (subset if dim is high, but dim=5 is small)
        for i in range(2**dim):
            pt = np.array([(i >> j) & 1 for j in range(dim)], dtype=float)
            pt = pt * 2 - 1  # Map 0..1 to -1..1
            points.append(pt)
        
        # Add some random points
        rng = np.random.default_rng(42)
        for _ in range(max(0, n_global - len(points))):
            points.append(rng.uniform(-5, 5, dim))
        
        # Evaluate global points
        for p in points:
            if calls >= budget:
                break
            if time.time() - start_time > self.time_limit:
                break
            p_clipped = np.clip(p, -5, 5)
            f = func(p_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = p_clipped.copy()
            history.append(f)
            
        # If we ran out of budget or time during global phase, return best
        if calls >= budget or time.time() - start_time > self.time_limit:
            return best_x, best_f

        # 2. Local Refinement Phase: Nelder-Mead
        # Initialize simplex around the best point found
        x0 = best_x
        f0 = best_f
        
        # Simplex size
        delta = 0.5
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += delta
            simplex.append(xi)
        
        # Evaluate initial simplex
        f_vals = [f0]
        for i in range(1, dim + 1):
            if calls >= budget or time.time() - start_time > self.time_limit:
                break
            f_val = func(simplex[i])
            calls += 1
            f_vals.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = simplex[i].copy()
            history.append(f_val)
        
        # Nelder-Mead Loop
        # We maintain a list of (f, x) sorted
        def sort_simplex(simplex, f_vals):
            # Sort by f_vals
            indices = np.argsort(f_vals)
            return [simplex[i] for i in indices], [f_vals[i] for i in indices]

        # Standard Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while calls < budget:
            if time.time() - start_time > self.time_limit:
                break
                
            simplex, f_vals = sort_simplex(simplex, f_vals)
            
            # Check for convergence (optional, but good for robustness)
            if len(f_vals) > 0 and np.std(f_vals) < 1e-12:
                break
            
            # Centroid of all but worst
            worst_x = simplex[-1]
            worst_f = f_vals[-1]
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - worst_x)
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            history.append(fr)
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()
            
            if fr < f_vals[0]:  # Better than best
                # Expansion
                xe = centroid + gamma * (centroid - worst_x)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                history.append(fe)
                if fe < best_f:
                    best_f = fe
                    best_x = xe.copy()
                
                if fe < fr:
                    simplex[-1] = xe
                    f_vals[-1] = fe
                else:
                    simplex[-1] = xr
                    f_vals[-1] = fr
                    
            elif fr < f_vals[-2]:  # Better than second worst
                simplex[-1] = xr
                f_vals[-1] = fr
            else:
                # Contraction
                xc = centroid + rho * (worst_x - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                history.append(fc)
                if fc < best_f:
                    best_f = fc
                    best_x = xc.copy()
                
                if fc < fr:
                    simplex[-1] = xc
                    f_vals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        if calls >= budget or time.time() - start_time > self.time_limit:
                            break
                        f_val = func(simplex[i])
                        calls += 1
                        f_vals[i] = f_val
                        history.append(f_val)
                        if f_val < best_f:
                            best_f = f_val
                            best_x = simplex[i].copy()
                    if calls >= budget:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
