import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # PSO Parameters
        N = 20  # Swarm size
        w_start = 0.9
        w_end = 0.4
        c1 = 1.5
        c2 = 1.5
        
        # Initialize particles
        # Uniform distribution in [-5, 5]
        particles = rng.uniform(-5, 5, size=(N, self.dim))
        velocities = np.zeros((N, self.dim))
        
        # Initial evaluation
        fitnesses = np.array([func(p) for p in particles])
        evals = N
        
        # Personal bests
        pbests = particles.copy()
        pbest_fitnesses = fitnesses.copy()
        
        # Global best
        global_best_idx = np.argmin(pbest_fitnesses)
        global_best = pbests[global_best_idx].copy()
        global_best_fitness = pbest_fitnesses[global_best_idx]
        
        # Main loop
        while evals < self.budget:
            # Calculate inertia weight decay
            # We estimate progress based on evals
            # Total expected evals is budget
            # Progress t = (evals - N) / (budget - N)
            if self.budget > N:
                t = (evals - N) / (self.budget - N)
            else:
                t = 1.0
            t = np.clip(t, 0, 1)
            w = w_start + (w_end - w_start) * t
            
            # Update velocities and positions
            new_particles = np.zeros_like(particles)
            new_fitnesses = np.zeros(N)
            
            for i in range(N):
                if evals >= self.budget:
                    break
                    
                # Update velocity
                r1 = rng.random(self.dim)
                r2 = rng.random(self.dim)
                velocities[i] = w * velocities[i] + c1 * r1 * (pbests[i] - particles[i]) + c2 * r2 * (global_best - particles[i])
                
                # Clamp velocity to prevent large jumps
                # A common heuristic is to clamp to a fraction of the box size
                v_max = 1.0
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                
                # Update position
                new_particles[i] = particles[i] + velocities[i]
                new_particles[i] = np.clip(new_particles[i], -5, 5)
                
                # Evaluate
                f = func(new_particles[i])
                new_fitnesses[i] = f
                evals += 1
                
                # Update personal best
                if f < pbest_fitnesses[i]:
                    pbest_fitnesses[i] = f
                    pbests[i] = new_particles[i].copy()
                    
                    # Update global best
                    if f < global_best_fitness:
                        global_best_fitness = f
                        global_best = new_particles[i].copy()
                        
            # Update particles for next iteration
            particles = new_particles
            
        return global_best, global_best_fitness
