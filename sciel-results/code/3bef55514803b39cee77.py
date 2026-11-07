"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Particle Swarm Optimization for continuous black-box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # PSO Parameters
        swarm_size = 10
        # 500 calls / 10 particles = 50 generations
        
        # Initialize particles
        # Positions
        X = np.random.uniform(-5, 5, (swarm_size, dim))
        # Velocities
        V = np.random.uniform(-0.5, 0.5, (swarm_size, dim))
        
        # Personal Bests
        P_pos = X.copy()
        P_val = np.full(swarm_size, np.inf)
        
        # Global Best
        G_pos = np.zeros(dim)
        G_val = np.inf
        
        # Inertia weight decay
        w_start = 0.9
        w_end = 0.4
        c1 = 1.5
        c2 = 1.5
        v_max = 1.0
        
        calls = 0
        total_generations = budget // swarm_size
        
        for gen in range(total_generations):
            # Calculate inertia weight for this generation
            w = w_start - (w_start - w_end) * (gen / total_generations)
            
            # Evaluate current positions and update bests
            for i in range(swarm_size):
                if calls >= budget:
                    break
                
                f_val = func(X[i])
                calls += 1
                
                # Update personal best
                if f_val < P_val[i]:
                    P_val[i] = f_val
                    P_pos[i] = X[i].copy()
                    
                # Update global best
                if f_val < G_val:
                    G_val = f_val
                    G_pos = X[i].copy()
            
            if calls >= budget:
                break
            
            # Update velocities and positions
            for i in range(swarm_size):
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                
                # Velocity update
                V[i] = w * V[i] + c1 * r1 * (P_pos[i] - X[i]) + c2 * r2 * (G_pos - X[i])
                
                # Clip velocity
                V[i] = np.clip(V[i], -v_max, v_max)
                
                # Position update
                X[i] = X[i] + V[i]
                
                # Clip position to bounds
                X[i] = np.clip(X[i], -5, 5)
                
        return G_pos, G_val
# EVOLVE-BLOCK-END
