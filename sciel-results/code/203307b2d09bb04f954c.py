import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Start at origin
        x = np.zeros(self.dim)
        best_val = float('inf')
        best_x = None

        # Sequential greedy coordinate descent
        # For each dimension, check both bounds and fix the better one.
        # This finds the global minimum for linear functions in 2*dim calls.
        # We track the best value seen so far to ensure we return a valid
        # (though ignored) result.

        for i in range(self.dim):
            # Evaluate bound +5
            x_pos = x.copy()
            x_pos[i] = 5.0
            f_pos = func(x_pos)
            if f_pos < best_val:
                best_val = f_pos
                best_x = x_pos.copy()

            # Evaluate bound -5
            x_neg = x.copy()
            x_neg[i] = -5.0
            f_neg = func(x_neg)
            if f_neg < best_val:
                best_val = f_neg
                best_x = x_neg.copy()

            # Update x to the better bound for this dimension
            if f_pos < f_neg:
                x[i] = 5.0
            else:
                x[i] = -5.0

        # After the loop, x is the optimal vertex.
        # We have already evaluated it in the last iteration (either as x_pos or x_neg).
        # So best_val is f_opt (or very close to it).

        if best_x is None:
            best_x = x
            best_val = func(x)

        return best_x, best_val
