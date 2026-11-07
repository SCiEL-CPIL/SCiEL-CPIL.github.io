"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer for BBOB f15 (Rotated Rastrigin).
    
    Strategy:
    1. Particle Swarm Optimization (PSO) for robust global exploration.
       Uses a small swarm with well-spread initialization to navigate the multimodal landscape.
       Particles are placed at diverse symmetric and random locations to cover the search space.
       
    2. Adaptive (1+1)-ES refinement with smart restarts.
       Focuses remaining budget on polishing the best candidate found.
       When stagnation occurs, performs larger jumps to escape local minima.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin for 300s limit
        
        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            if calls >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Particle Swarm Optimization ---
        # PSO with a small swarm for efficient exploration of the 5D space.
        
        n_particles = 15
        # Allocate ~65% of budget to PSO to ensure thorough exploration
        pso_budget = int(budget * 0.65)
        
        # Initialize particles with diverse placement
        particles = np.zeros((n_particles, dim))
        
        # Deterministic symmetric points for good initial coverage
        # Spread across the domain to cover potential rotated minimum locations
        particles[0] = np.zeros(dim)
        particles[1] = np.full(dim, 2.0)
        particles[2] = np.full(dim, -2.0)
        particles[3] = np.array([2.0, -2.0, 2.0, -2.0, 2.0])[:dim]
        particles[4] = np.array([-2.0, 2.0, -2.0, 2.0, -2.0])[:dim]
        particles[5] = np.array([0.0, 2.0, -2.0, 0.0, 2.0])[:dim]
        particles[6] = np.array([2.0, 0.0, -2.0, 2.0, 0.0])[:dim]
        particles[7] = np.array([1.0, 1.0, 1.0, 1.0, 1.0])[:dim]
        particles[8] = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])[:dim]
        
        # Random points for additional diversity
        for i in range(9, n_particles):
            particles[i] = rng.uniform(-3.5, 3.5, dim)
        
        # Evaluate initial positions
        p_fits = np.zeros(n_particles)
        for i in range(n_particles):
            if calls >= pso_budget or time.time() > deadline:
                break
            p_fits[i] = eval_func(particles[i])
        
        # Initialize velocities with small random values
        velocities = np.zeros((n_particles, dim))
        for i in range(n_particles):
            velocities[i] = rng.uniform(-0.8, 0.8, dim)
        
        # Personal bests
        p_best = particles.copy()
        p_best_f = p_fits.copy()
        
        # Global best
        g_best_idx = np.argmin(p_best_f)
        g_best = p_best[g_best_idx].copy()
        g_best_f = p_best_f[g_best_idx]
        
        # PSO parameters
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        
        n_pso_calls = pso_budget - calls
        n_iterations = max(1, n_pso_calls // n_particles)
        
        # Stagnation counter for early termination
        stagnation_counter = 0
        max_stagnation = 40
        
        for it in range(n_iterations):
            if calls >= pso_budget or time.time() > deadline:
                break
            
            # Inertia weight: decays from 0.9 to 0.3
            w = 0.9 - 0.6 * (it / max(1, n_iterations))
            
            improved_this_iter = False
            
            for i in range(n_particles):
                if calls >= pso_budget or time.time() > deadline:
                    break
                
                # Update velocity
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (p_best[i] - particles[i]) + 
                               c2 * r2 * (g_best - particles[i]))
                
                # Clip velocity to prevent overshooting
                velocities[i] = np.clip(velocities[i], -2.0, 2.0)
                
                # Update position
                new_pos = particles[i] + velocities[i]
                new_pos = np.clip(new_pos, -5.0, 5.0)
                
                # Evaluate
                f_new = eval_func(new_pos)
                
                # Update personal best
                if f_new < p_best_f[i]:
                    p_best[i] = new_pos.copy()
                    p_best_f[i] = f_new
                    improved_this_iter = True
                    
                    # Update global best
                    if f_new < g_best_f:
                        g_best = new_pos.copy()
                        g_best_f = f_new
                
                particles[i] = new_pos
            
            if improved_this_iter:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            if stagnation_counter > max_stagnation:
                break

        # --- Phase 2: Adaptive (1+1)-ES Refinement ---
        # Refine the best point found by PSO using (1+1)-ES with smart restarts.
        
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            # Start with a moderate sigma for local search
            sigma = 0.4
            window = 12
            recent = [False] * window
            idx = 0
            stagnation = 0
            max_stagnation_es = 80
            
            # Track best sigma for recovery
            best_sigma = sigma
            
            # Run until budget or time is up
            while calls < budget and time.time() < deadline:
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = eval_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    recent[idx % window] = True
                    stagnation = 0
                    # If improving well, remember this sigma
                    if sigma > best_sigma * 0.5:
                        best_sigma = sigma
                else:
                    recent[idx % window] = False
                    stagnation += 1
                idx += 1
                
                # Adapt sigma based on success rate
                if idx >= window:
                    sr = sum(recent) / window
                    if sr > 0.45:
                        sigma *= 1.25
                    elif sr < 0.15:
                        sigma *= 0.75
                    else:
                        sigma *= 1.0
                sigma = np.clip(sigma, 0.005, 3.0)
                
                # Smart restart when stuck
                if stagnation > max_stagnation_es:
                    # Try several jump strategies
                    jumped = False
                    
                    # Strategy 1: Jump to a random point near the best
                    for _ in range(3):
                        jump_scale = 1.5 * rng.uniform(0.5, 2.0)
                        y = np.clip(x + jump_scale * rng.standard_normal(dim), -5.0, 5.0)
                        fy = eval_func(y)
                        if fy < fx:
                            x, fx = y, fy
                            jumped = True
                            break
                    
                    # Strategy 2: If still stuck, try a larger random restart
                    if not jumped and calls + 5 < budget:
                        y = rng.uniform(-4.0, 4.0, dim)
                        fy = eval_func(y)
                        if fy < fx:
                            x, fx = y, fy
                            jumped = True
                    
                    # Reset parameters after restart
                    if jumped:
                        sigma = max(0.3, best_sigma * 0.5)
                        stagnation = 0
                        max_stagnation_es = 50
                    else:
                        # Reduce sigma to try finer search
                        sigma *= 0.6
                        max_stagnation_es = 30
                        
                        # If very little budget left, break
                        if budget - calls < 15:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
