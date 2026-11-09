"""Simulated Annealing with adaptive step size for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Simulated Annealing with adaptive step size and cooling, optimized for plateau structures."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial broad uniform scout
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive parameters
        # Start with a relatively high temperature to allow exploration
        T = 10.0
        T_min = 1e-8
        # Step size for the random walk
        step_size = 1.0
        
        # Tracking for adaptation
        stall_counter = 0
        # We adapt step size based on whether we found a new global best
        # If no new best, we are likely on a plateau or in a local trap, so we grow step size.
        # If new best, we are making progress, so we shrink step size to refine.

        for i in range(budget - 1):
            # Generate candidate
            # Use normal distribution for direction
            direction = rng.standard_normal(dim)
            y = np.clip(x + step_size * direction, -5, 5)
            
            fy = func(y)
            
            # Acceptance logic
            if fy < fx:
                # Improvement in current state
                x = y
                fx = fy
                
                # Check for global best improvement
                if fy < best_fx:
                    best_fx = fy
                    best_x = x.copy()
                    stall_counter = 0
                    # Shrink step size to refine around the new best
                    step_size *= 0.95
                    # Cool down slightly faster on improvement
                    T *= 0.98
                else:
                    # Improved local, but not global. 
                    # This suggests we are moving within a basin or plateau.
                    # Keep step size roughly same or shrink slightly less
                    step_size *= 0.99
                    T *= 0.995
            else:
                # Worse or equal
                if fy == fx:
                    # Plateau move. Accept to explore the plateau, but we need to jump out soon.
                    # We accept it, but we treat it as a "stall" for the purpose of step size growth
                    # because we didn't find a lower value.
                    x = y
                    fx = fy
                    stall_counter += 1
                    # Grow step size to escape the plateau
                    step_size *= 1.05
                    T *= 0.99
                else:
                    # Worse value
                    delta_f = fy - fx
                    # Acceptance probability
                    if delta_f < T * 10: # Threshold to avoid exp(0) or exp(-large)
                        prob = np.exp(-delta_f / T)
                        if rng.random() < prob:
                            x = y
                            fx = fy
                            stall_counter += 1
                            # We accepted a worse move, so we are exploring.
                            # Don't change step size drastically, maybe grow slightly to keep exploring
                            step_size *= 1.01
                            T *= 0.99
                        else:
                            # Rejected
                            stall_counter += 1
                            # Grow step size to try a different direction/magnitude next time
                            step_size *= 1.02
                            T *= 0.995

            # Clamp step size
            step_size = np.clip(step_size, 0.01, 10.0)
            
            # If we are stuck for too long (stall_counter > 50), force a large jump
            if stall_counter > 50:
                # Random restart from best point with large step
                x = best_x.copy()
                fx = best_fx
                step_size = 2.0 # Reset to a moderate large step
                T = max(T, 5.0) # Reheat
                stall_counter = 0

            if T < T_min:
                T = T_min

        return best_x, best_fx
# EVOLVE-BLOCK-END
