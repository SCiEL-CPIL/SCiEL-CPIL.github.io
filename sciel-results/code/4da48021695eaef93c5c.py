import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300.0  # Safety margin for 330s limit
        
        # State tracking
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return best_f
            # Time check
            if time.time() - start_time > time_limit:
                return best_f
            
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Particle Swarm Optimization (PSO) to find the basin
        # Budget allocation: 20% for PSO, 80% for Nelder-Mead
        pso_budget = int(self.budget * 0.2)
        if pso_budget < 10:
            pso_budget = 10
            
        n_particles = 10
        # Initialize particles
        particles = np.random.uniform(-5, 5, (n_particles, self.dim))
        v_particles = np.random.uniform(-1, 1, (n_particles, self.dim))
        
        p_personal = particles.copy()
        f_personal = np.array([safe_eval(p) for p in particles])
        
        global_idx = np.argmin(f_personal)
        p_global = p_personal[global_idx].copy()
        f_global = f_personal[global_idx]
        
        # PSO Parameters
        w = 0.7
        c1 = 1.5
        c2 = 1.5
        
        # Run PSO
        while calls_made < pso_budget and time.time() - start_time < time_limit:
            for i in range(n_particles):
                if calls_made >= self.budget:
                    break
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                v_particles[i] = w * v_particles[i] + c1 * r1 * (p_personal[i] - particles[i]) + c2 * r2 * (p_global - particles[i])
                particles[i] = np.clip(particles[i] + v_particles[i], -5, 5)
                
                f_i = safe_eval(particles[i])
                if f_i < f_personal[i]:
                    f_personal[i] = f_i
                    p_personal[i] = particles[i].copy()
                    if f_i < f_global:
                        f_global = f_i
                        p_global = p_personal[i].copy()
                        
        # Phase 2: Nelder-Mead
        # Initialize simplex around p_global
        # If p_global is None (shouldn't happen), use origin
        if p_global is None:
            p_global = np.zeros(self.dim)
            
        # Create initial simplex
        # One point at p_global, others offset slightly
        simplex = [p_global]
        for i in range(self.dim):
            offset = np.zeros(self.dim)
            # Heuristic step size
            step = 0.1
            offset[i] = step
            simplex.append(np.clip(p_global + offset, -5, 5))
            
        # Evaluate initial simplex
        f_simplex = []
        for i in range(self.dim + 1):
            if calls_made >= self.budget:
                break
            f_simplex.append(safe_eval(simplex[i]))
            
        # If we ran out of budget in init, return best
        if len(f_simplex) < self.dim + 1:
            return best_x, best_f

        # Nelder-Mead Loop
        # Standard parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Sort simplex
        # We need to keep the simplex points and their values sorted
        # We will use a list of (f, x) and sort it
        
        sm = list(zip(f_simplex, simplex))
        sm.sort(key=lambda t: t[0])
        
        # Convert to lists for easy manipulation
        # sm is list of (f, x)
        
        while calls_made < self.budget and time.time() - start_time < time_limit:
            # Get best and worst
            f_best, x_best = sm[0]
            f_worst, x_worst = sm[-1]
            
            # Calculate centroid of all points except the worst
            centroid = np.mean([x for f, x in sm[:-1]], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - x_worst)
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = safe_eval(x_ref)
            
            if f_ref < f_best:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = safe_eval(x_exp)
                
                if f_exp < f_ref:
                    # Replace worst with expansion
                    sm[-1] = (f_exp, x_exp)
                else:
                    # Replace worst with reflection
                    sm[-1] = (f_ref, x_ref)
            elif f_ref < sm[-2][0]:
                # Replace worst with reflection
                sm[-1] = (f_ref, x_ref)
            else:
                # Contraction
                if f_ref < f_worst:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = safe_eval(x_con)
                    
                    if f_con < f_ref:
                        sm[-1] = (f_con, x_con)
                    else:
                        # Shrink
                        for i in range(1, len(sm)):
                            sm[i] = (sm[i][0], x_best + sigma * (sm[i][1] - x_best))
                else:
                    # Inside contraction
                    x_con = centroid + rho * (x_worst - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = safe_eval(x_con)
                    
                    if f_con < f_worst:
                        sm[-1] = (f_con, x_con)
                    else:
                        # Shrink
                        for i in range(1, len(sm)):
                            sm[i] = (sm[i][0], x_best + sigma * (sm[i][1] - x_best))
            
            # Re-sort simplex
            sm.sort(key=lambda t: t[0])
            
            # Check for convergence (optional, but good for stopping early if stuck)
            # If the spread is very small, we might be done
            # But we have budget, so keep going unless we hit limit
            
        return best_x, best_f
