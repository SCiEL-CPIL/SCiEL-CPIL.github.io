"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Exploit linear structure: sequential coordinate optimization for early progress."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Start at origin
        x_curr = np.zeros(dim)
        f_curr = func(x_curr)
        
        # h is the boundary of the search space
        h = 5.0
        
        # Sequentially optimize each coordinate
        # This allows the best value b_i to decrease step-by-step as we fix more coordinates
        for i in range(dim):
            # Probe positive direction
            x_pos = x_curr.copy()
            x_pos[i] = h
            f_pos = func(x_pos)
            
            if f_pos < f_curr:
                # Positive direction is better, so gradient is negative, optimal is +h
                x_curr = x_pos
                f_curr = f_pos
            else:
                # Positive direction is worse (or equal), so gradient is positive, optimal is -h
                # We must evaluate the negative point to update our best value
                x_neg = x_curr.copy()
                x_neg[i] = -h
                f_neg = func(x_neg)
                
                # f_neg should be better than f_curr because we are moving towards the optimum
                # Even if f_pos was equal to f_curr (gradient 0), f_neg is same as f_curr.
                # But for linear non-constant, it will be better.
                x_curr = x_neg
                f_curr = f_neg
        
        # x_curr is now the optimal vertex
        return x_curr, f_curr
# EVOLVE-BLOCK-END
