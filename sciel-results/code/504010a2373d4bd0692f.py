import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, use 280s to be safe
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        
        # PSO Parameters
        # For dim=5 and budget=500, a small swarm is efficient.
        # Standard PSO often uses 20-50 particles. 
        # With 500 budget, if swarm=20, we get 25 generations.
        # If swarm=10, we get 50 generations.
        # Let's try swarm=10 to allow more global exploration iterations.
        # However, PSO benefits from larger populations for diversity.
        # Let's stick with a moderate size. 20 particles is standard.
        # 500 / 20 = 25 iterations.
        # Let's try 15 particles -> 33 iterations.
        n_particles = 15
        
        # Inertia weight decay: from 0.9 to 0.4
        w_start = 0.9
        w_end = 0.4
        
        # Cognitive and Social coefficients
        # Constriction factor approach is often more stable.
        # phi1 = 2.0, phi2 = 2.0 -> chi = 2 / (2 - phi + sqrt(phi^2 - 4*phi))
        phi1 = 2.0
        phi2 = 2.0
        chi = 2.0 / (2.0 - (phi1 + phi2) + np.sqrt((phi1 + phi2)**2 - 4.0 * (phi1 + phi2)))
        
        # If using standard velocity update with w, c1, c2:
        # v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        # Let's use the standard formulation with inertia decay.
        c1 = 1.5
        c2 = 1.5
        
        # Bounds
        lower_bound = -5.0
        upper_bound = 5.0
        
        # Initialize particles
        # Use a deterministic seed for reproducibility of the search process itself
        rng = np.random.default_rng(42)
        
        # Initialize positions uniformly in the box
        X = rng.uniform(lower_bound, upper_bound, size=(n_particles, dim))
        
        # Initialize velocities to zero
        V = np.zeros((n_particles, dim))
        
        # Evaluate initial population
        F = np.array([func(x) for x in X])
        calls_made = n_particles
        
        # Initialize personal bests
        Pbest = X.copy()
        Fbest = F.copy()
        
        # Initialize global best
        gbest_idx = np.argmin(Fbest)
        Gbest = Pbest[gbest_idx].copy()
        Gbest_val = Fbest[gbest_idx]
        
        best_overall_val = Gbest_val
        best_overall_x = Gbest.copy()
        
        # Main loop
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough budget for another full iteration
            if calls_made + n_particles > budget:
                # If we can't afford a full swarm update, we might just stop
                # or do a partial update. Standard PSO requires full swarm.
                # Given the strict budget, stopping is safer to avoid weird states.
                break
                
            # Update inertia weight
            # Linear decay based on iteration count
            # Total expected iterations = budget // n_particles
            total_iters = budget // n_particles
            current_iter = calls_made // n_particles
            if total_iters > 0:
                progress = current_iter / total_iters
            else:
                progress = 1.0
            w = w_start + (w_end - w_start) * progress
            
            # Update velocities and positions
            for i in range(n_particles):
                # r1, r2 are random numbers in [0, 1]
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                
                # Velocity update
                V[i] = w * V[i] + c1 * r1 * (Pbest[i] - X[i]) + c2 * r2 * (Gbest - X[i])
                
                # Position update
                X[i] = X[i] + V[i]
                
                # Clip positions to bounds
                X[i] = np.clip(X[i], lower_bound, upper_bound)
            
            # Evaluate new positions
            F_new = np.array([func(x) for x in X])
            calls_made += n_particles
            
            # Update personal bests
            for i in range(n_particles):
                if F_new[i] < Fbest[i]:
                    Fbest[i] = F_new[i]
                    Pbest[i] = X[i].copy()
                    
                    # Update global best
                    if Fbest[i] < Gbest_val:
                        Gbest_val = Fbest[i]
                        Gbest = Pbest[i].copy()
                        
                        # Update overall best
                        if Gbest_val < best_overall_val:
                            best_overall_val = Gbest_val
                            best_overall_x = Gbest.copy()
            
            # Optional: Stagnation check? 
            # If no improvement for many iterations, we could randomize some particles.
            # But with only ~30 iterations, it's hard to define stagnation robustly without overhead.
            # We'll rely on the inertia decay to slow down and refine.
            
        return best_overall_x, best_overall_val
