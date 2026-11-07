"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Particle Swarm Optimization based optimizer for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self._call_count = 0

    def __call__(self, func):
        self._call_count = 0
        dim = self.dim
        budget = self.budget
        
        # PSO Parameters
        pop_size = 20
        w = 0.7
        c1 = 1.5
        c2 = 1.5
        v_max = 2.0
        
        # Initialize
        positions = np.random.uniform(-5, 5, (pop_size, dim))
        velocities = np.random.uniform(-1, 1, (pop_size, dim))
        
        personal_best_pos = positions.copy()
        personal_best_val = np.full(pop_size, np.inf)
        
        global_best_pos = np.full(dim, np.nan)
        global_best_val = np.inf
        
        # Evaluate initial population
        for i in range(pop_size):
            if self._call_count >= budget:
                break
            val = func(positions[i])
            self._call_count += 1
            personal_best_val[i] = val
            personal_best_pos[i] = positions[i].copy()
            if val < global_best_val:
                global_best_val = val
                global_best_pos = positions[i].copy()
                
        # Main loop
        while self._call_count < budget:
            for i in range(pop_size):
                if self._call_count >= budget:
                    break
                
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                
                # Update velocity
                velocities[i] = w * velocities[i] + c1 * r1 * (personal_best_pos[i] - positions[i]) + c2 * r2 * (global_best_pos - positions[i])
                
                # Clip velocity
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                
                # Update position
                new_pos = positions[i] + velocities[i]
                new_pos = np.clip(new_pos, -5, 5)
                
                val = func(new_pos)
                self._call_count += 1
                
                if val < personal_best_val[i]:
                    personal_best_val[i] = val
                    personal_best_pos[i] = new_pos.copy()
                    if val < global_best_val:
                        global_best_val = val
                        global_best_pos = new_pos.copy()
                        
                positions[i] = new_pos
                
        return global_best_pos, global_best_val
# EVOLVE-BLOCK-END
