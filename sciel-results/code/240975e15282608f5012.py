import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320
        dim = self.dim
        budget = self.budget
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)

        best_x = np.zeros(dim)
        best_f = float('inf')
        eval_count = 0

        def check_deadline():
            return time.time() < deadline - 2.0

        def evaluate(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget or not check_deadline():
                return best_f
            x_clipped = np.clip(x, lower, upper)
            fx = func(x_clipped)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # --- Phase 1: Latin Hypercube Sampling for initial global coverage ---
        n_lhs = 60
        rng = np.random.default_rng(42)
        # Generate LHS points
        indices = np.array([rng.permutation(n_lhs) for _ in range(dim)]).T
        x_lhs = (indices + 0.5) / n_lhs * (upper - lower) + lower
        for x in x_lhs:
            if eval_count >= budget or not check_deadline():
                break
            evaluate(x)

        if eval_count >= budget or not check_deadline():
            return best_x, best_f

        # --- Phase 2: Particle Swarm Optimization ---
        n_particles = 20
        # Initialize particles around the best point found so far with some spread
        # Use a mix: half around best, half random
        particles = []
        velocities = []
        
        # Seed some particles near the best
        for i in range(n_particles):
            if i < n_particles // 2:
                # Around best
                x_p = best_x + rng.normal(0, 1.0, dim)
            else:
                # Random in box
                x_p = rng.uniform(lower, upper)
            
            x_p = np.clip(x_p, lower, upper)
            particles.append(x_p)
            velocities.append(rng.uniform(-1.0, 1.0, dim))
            
            fx = evaluate(x_p)
            if fx < best_f:
                best_f = fx
                best_x = x_p.copy()

        if eval_count >= budget or not check_deadline():
            return best_x, best_f

        # PSO Parameters
        w = 0.7  # Inertia
        c1 = 1.5 # Cognitive
        c2 = 1.5 # Social
        
        pbests = [p.copy() for p in particles]
        pbest_fs = [evaluate(p) for p in pbests] # Re-eval to ensure pbest is tracked correctly? No, we just set them to current values.
        # Actually, we should just use the values we just computed or re-eval if necessary. 
        # To save budget, let's assume the last evaluation of each particle is its pbest.
        # But we just evaluated them. Let's store those values.
        # Wait, the loop above evaluated them. Let's capture the values properly.
        
        # Redo initialization properly to track pbest
        pbests = []
        pbest_fs = []
        for i in range(n_particles):
            if i < n_particles // 2:
                x_p = best_x + rng.normal(0, 1.0, dim)
            else:
                x_p = rng.uniform(lower, upper)
            x_p = np.clip(x_p, lower, upper)
            
            # If we already evaluated a similar point, we might waste budget. 
            # But for simplicity and robustness, let's just evaluate.
            # We need to be careful with budget.
            if eval_count >= budget or not check_deadline():
                break
                
            fx = evaluate(x_p)
            pbests.append(x_p)
            pbest_fs.append(fx)
            
            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x_p.copy()

        # If we ran out of budget during init, return
        if eval_count >= budget or not check_deadline():
            return best_x, best_f

        # Ensure pbests list is full
        while len(pbests) < n_particles:
            # Fallback: duplicate best
            pbests.append(best_x.copy())
            pbest_fs.append(best_f)
            
        # Re-initialize velocities if not done properly or just use existing
        # We initialized velocities in the first loop, but then re-initialized particles.
        # Let's re-initialize velocities for the new particles.
        velocities = [rng.uniform(-1.0, 1.0, dim) for _ in range(n_particles)]

        max_iters = 500 # Will be cut by budget/time
        for it in range(max_iters):
            if eval_count >= budget or not check_deadline():
                break

            # Adaptive parameters: decrease w, increase c2 over time?
            # Simple decay for w
            w = 0.7 * (1.0 - it / max_iters) + 0.4 * (it / max_iters)
            
            for i in range(n_particles):
                if eval_count >= budget or not check_deadline():
                    break
                
                r1 = rng.random()
                r2 = rng.random()
                
                v = w * velocities[i] + c1 * r1 * (pbests[i] - particles[i]) + c2 * r2 * (best_x - particles[i])
                velocities[i] = v
                
                new_pos = particles[i] + v
                new_pos = np.clip(new_pos, lower, upper)
                
                fx = evaluate(new_pos)
                particles[i] = new_pos
                
                if fx < pbest_fs[i]:
                    pbest_fs[i] = fx
                    pbests[i] = new_pos.copy()

        # --- Phase 3: Final Local Refinement with Nelder-Mead ---
        # Use remaining budget to polish the best solution
        if eval_count < budget and check_deadline():
            try:
                from scipy.optimize import minimize
                
                # Wrapper for Nelder-Mead
                nm_eval_count = [0]
                def nm_func(x):
                    if eval_count >= budget or not check_deadline():
                        return best_f
                    x_clipped = np.clip(x, lower, upper)
                    fx = func(x_clipped)
                    eval_count += 1
                    nm_eval_count[0] += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x_clipped.copy()
                    return fx

                # Run Nelder-Mead
                # maxiter is a rough guide. We rely on the wrapper to stop.
                minimize(nm_func, best_x, method='Nelder-Mead', 
                         options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
            except Exception:
                pass

        return best_x, best_f
