import numpy as np
import time

class Optimizer:
    """
    Particle Swarm Optimization (PSO) for minimizing a black-box function.
    Uses inertia weight decay and constriction factors for stability.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s limit
        
        # PSO Parameters
        N = 10  # Swarm size
        # We have 500 budget. Initial evals: 10. Remaining: 490.
        # Iterations: 49.
        # Total calls: 10 + 10*49 = 500.
        
        w_init = 0.9
        w_final = 0.4
        c1 = 2.0  # Cognitive
        c2 = 2.0  # Social
        # Constriction factor for standard PSO
        # phi = c1 + c2
        # chi = 2 / (2 - phi - sqrt(phi^2 - 4*phi)) if phi < 4 else 1
        phi = c1 + c2
        if phi < 4:
            chi = 2.0 / (2.0 - phi - np.sqrt(phi * phi - 4.0 * phi))
        else:
            chi = 1.0

        rng = np.random.default_rng(42)
        
        # Initialize positions in [-5, 5]^dim
        x = rng.uniform(-5, 5, (N, self.dim))
        # Initialize velocities to 0
        v = np.zeros((N, self.dim))
        
        # Evaluate initial population
        fx = np.array([func(xi) for xi in x])
        
        # Initialize local and global bests
        pbest_x = x.copy()
        pbest_f = fx.copy()
        global_idx = np.argmin(fx)
        gbest_x = x[global_idx].copy()
        gbest_f = fx[global_idx]
        
        calls_made = N
        budget_remaining = self.budget - calls_made
        
        # If budget is too small for even one iteration, just return best
        if budget_remaining <= 0:
            return gbest_x, gbest_f
            
        n_iters = budget_remaining // N
        # If there's a remainder, we can do a few single-particle steps at the end?
        # For simplicity and strict budget adherence, we'll just do full iterations.
        # 490 // 10 = 49 iterations. 49 * 10 = 490 calls. Total 500. Perfect.
        
        for i in range(n_iters):
            # Check time
            if time.time() > deadline:
                break
                
            # Update inertia weight
            w = w_init + (w_final - w_init) * (i / max(1, n_iters - 1))
            
            for j in range(N):
                if time.time() > deadline:
                    break
                    
                # Update velocity
                r1 = rng.random(self.dim)
                r2 = rng.random(self.dim)
                
                v[j] = chi * (w * v[j] + c1 * r1 * (pbest_x[j] - x[j]) + c2 * r2 * (gbest_x - x[j]))
                
                # Clamp velocity to prevent overshooting too far
                # Max velocity should be comparable to box size
                v[j] = np.clip(v[j], -5.0, 5.0)
                
                # Update position
                x[j] = x[j] + v[j]
                
                # Clip position to box [-5, 5]
                x[j] = np.clip(x[j], -5.0, 5.0)
                
                # Evaluate
                fx_j = func(x[j])
                calls_made += 1
                
                # Update local best
                if fx_j < pbest_f[j]:
                    pbest_f[j] = fx_j
                    pbest_x[j] = x[j].copy()
                    
                    # Update global best
                    if fx_j < gbest_f:
                        gbest_f = fx_j
                        gbest_x = x[j].copy()
                        
        return gbest_x, gbest_f
