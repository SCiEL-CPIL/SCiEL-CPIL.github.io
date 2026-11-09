"""Portfolio-based continuous optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-strategy optimizer combining:
    1. PSO for global exploration
    2. Adaptive (1+1)-ES with multiple restarts for local refinement
    
    Uses time guard to stay within wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s safety margin
        budget = self.budget
        dim = self.dim

        # Track best
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            try:
                x = np.asarray(x, dtype=float)
                x = np.clip(x, -5.0, 5.0)
                f = func(x)
                f = float(f)
            except Exception:
                return np.inf
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # --- Strategy 1: PSO for global exploration ---
        n_particles = 20
        max_pso_iter = 30
        
        # Initialize particles
        particles = rng.uniform(-4, 4, (n_particles, dim))
        velocities = rng.uniform(-2, 2, (n_particles, dim))
        pbest_x = particles.copy()
        pbest_f = np.full(n_particles, np.inf)
        gbest_x = None
        gbest_f = np.inf
        
        # Evaluate initial positions
        for i in range(n_particles):
            if calls_made >= budget or time.time() > deadline:
                break
            f = safe_call(particles[i])
            pbest_f[i] = f
            if f < gbest_f:
                gbest_f = f
                gbest_x = particles[i].copy()
        
        # PSO iteration
        w = 0.9  # inertia
        c1 = 1.5  # cognitive
        c2 = 1.5  # social
        
        for it in range(max_pso_iter):
            if calls_made >= budget or time.time() > deadline:
                break
            
            for i in range(n_particles):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Update velocity
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = w * velocities[i] + c1 * r1 * (pbest_x[i] - particles[i]) + c2 * r2 * (gbest_x - particles[i])
                velocities[i] = np.clip(velocities[i], -3.0, 3.0)
                
                # Update position
                particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)
                
                # Evaluate
                f = safe_call(particles[i])
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest_x[i] = particles[i].copy()
                if f < gbest_f:
                    gbest_f = f
                    gbest_x = particles[i].copy()
            
            # Adapt parameters
            w = max(0.4, w * 0.95)
            c1 = max(0.5, c1 * 0.97)
            c2 = min(2.5, c2 * 1.02)
        
        # Update global best from PSO
        if gbest_x is not None and gbest_f < best_f:
            best_f = gbest_f
            best_x = gbest_x.copy()

        # --- Strategy 2: Multi-restart (1+1)-ES with adaptive sigma ---
        n_starts = 6
        for s in range(n_starts):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Different starting points
            if s == 0:
                x = np.zeros(dim)
            elif s == 1:
                x = best_x if best_x is not None else rng.uniform(-4, 4, dim)
            elif s == 2:
                x = rng.uniform(-4, 4, dim)
            elif s == 3:
                x = best_x + rng.normal(0, 0.5, dim) if best_x is not None else rng.uniform(-3, 3, dim)
            elif s == 4:
                x = rng.uniform(-5, 5, dim)
            else:
                x = best_x + rng.normal(0, 1.0, dim) if best_x is not None else rng.uniform(-4, 4, dim)
            
            x = np.clip(x, -5.0, 5.0)
            fx = safe_call(x)
            if not np.isfinite(fx):
                continue
            
            # Adaptive (1+1)-ES
            sigma = 1.0
            for step in range(budget - calls_made):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = safe_call(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                
                if sigma < 1e-4:
                    break
            
            # Local refinement with coordinate descent
            if calls_made < budget and time.time() < deadline:
                x = self._coordinate_descent(func, safe_call, x, fx, budget, deadline, dim)
                fx = safe_call(x)

        # --- Strategy 3: Final local refinement around best point ---
        if best_x is not None and calls_made < budget and time.time() < deadline:
            best_x = self._coordinate_descent(func, safe_call, best_x, best_f, budget, deadline, dim)
            best_f = safe_call(best_x)

        # Ensure we've made at least some calls
        if calls_made == 0:
            x0 = np.zeros(dim)
            safe_call(x0)

        return best_x, best_f

    def _coordinate_descent(self, func, safe_call, x, fx, budget, deadline, dim):
        """Simple coordinate descent to refine a point."""
        x = x.copy()
        fx = safe_call(x)
        step_size = 0.5
        
        for _ in range(200):  # Max iterations
            if time.time() > deadline:
                break
            improved = False
            for d in range(dim):
                for direction in [1, -1]:
                    y = x.copy()
                    y[d] += direction * step_size
                    y = np.clip(y, -5.0, 5.0)
                    fy = safe_call(y)
                    if fy < fx:
                        x, fx = y, fy
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                step_size *= 0.5
                if step_size < 1e-4:
                    break
        
        return x
# EVOLVE-BLOCK-END
