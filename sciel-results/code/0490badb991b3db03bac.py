import numpy as np

class Optimizer:
    """Particle Swarm Optimizer for BBOB f5 (Linear Slope)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # PSO Parameters
        # Population size: small to medium to maximize generations within budget
        # With budget 500 and d=5, we can afford a population of ~20-50.
        # Let's use N=20. This allows 25 generations.
        N = 20
        
        # Initialize positions uniformly in [-5, 5]
        X = np.random.uniform(-5, 5, (N, d))
        
        # Initialize velocities to zero
        V = np.zeros((N, d))
        
        # Evaluate initial population
        F = np.zeros(N)
        calls_used = 0
        for i in range(N):
            F[i] = func(X[i])
            calls_used += 1
            
        # Track global best
        best_idx = np.argmin(F)
        P_global = X[best_idx].copy()
        F_global = F[best_idx]
        
        # Track personal bests
        P_personal = X.copy()
        F_personal = F.copy()
        
        # PSO Parameters
        w = 0.9  # Inertia weight, will decay
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        V_max = 2.0  # Max velocity magnitude per dimension
        
        # Decay factor for w
        w_min = 0.4
        # Total expected generations
        # We have budget - calls_used remaining.
        # Each generation costs N calls.
        max_generations = (budget - calls_used) // N
        
        for gen in range(max_generations):
            if calls_used >= budget:
                break
                
            # Decay inertia weight
            if max_generations > 0:
                w = w_min + (w - w_min) * (1 - gen / max_generations)
            
            # Update velocities and positions
            for i in range(N):
                if calls_used >= budget:
                    break
                
                # Random factors
                r1 = np.random.rand(d)
                r2 = np.random.rand(d)
                
                # Velocity update
                V[i] = w * V[i] + c1 * r1 * (P_personal[i] - X[i]) + c2 * r2 * (P_global - X[i])
                
                # Clamp velocity
                V[i] = np.clip(V[i], -V_max, V_max)
                
                # Position update
                X[i] = X[i] + V[i]
                
                # Clip position to bounds [-5, 5]
                X[i] = np.clip(X[i], -5, 5)
                
                # Evaluate new position
                F[i] = func(X[i])
                calls_used += 1
                
                # Update personal best
                if F[i] < F_personal[i]:
                    F_personal[i] = F[i]
                    P_personal[i] = X[i].copy()
                    
                    # Update global best
                    if F[i] < F_global:
                        F_global = F[i]
                        P_global = X[i].copy()
                        
        return P_global, F_global
