"""Model-and-optimize approach for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Model-based optimizer for BBOB-style f5 (Linear slope).
    
    Strategy:
    1. Estimate the linear model parameters (gradient 'a' and offset 'b') 
       using finite differences from the origin with a large step (h=5).
       This uses 1 + dim = 6 calls to get a very accurate gradient estimate
       for a linear function, as the points are at the box boundaries.
    2. Construct the optimal vertex x* directly from the sign of the gradient.
       For a linear function f(x) = a^T x + b on [-5, 5]^d, the minimum is
       at x_i = -5 * sign(a_i).
    3. Evaluate x* at call 7. This should be the global minimum.
    4. Evaluate the other 31 vertices in order of predicted value as a 
       safeguard (and to consume budget if needed, though for AOCC we 
       prefer to stop early once the min is found).
    5. If the first vertex is not the minimum (due to estimation error),
       the sorted order ensures we find the true minimum very quickly.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin
        calls_made = 0
        dim = self.dim
        budget = self.budget

        # 1. Start at origin to estimate gradient cleanly
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_made += 1
        best_x = x0.copy()
        best_fx = f0

        # 2. Estimate gradient using forward differences with large step
        # f(x) = a^T x + b. f(0) = b.
        # f(h e_i) = a_i h + b.
        # a_i = (f(h e_i) - f(0)) / h
        # Using h=5 puts the points at the box boundary, which is ideal for
        # linear functions and maximizes the signal-to-noise ratio if there
        # were any nonlinearity (though there shouldn't be).
        h = 5.0
        a_est = np.zeros(dim)
        
        for i in range(dim):
            if time.time() > deadline or calls_made >= budget:
                break
            
            xi = x0.copy()
            xi[i] += h
            fi = func(xi)
            calls_made += 1
            
            a_est[i] = (fi - f0) / h
            
            if fi < best_fx:
                best_fx = fi
                best_x = xi.copy()

        # 3. Construct the optimal vertex directly
        # For f(x) = a^T x + b, min is at x_i = -5 * sign(a_i)
        # If a_i is 0, the sign is 0, but for a linear slope function,
        # a_i should be non-zero. We use -5 if a_i > 0, 5 if a_i < 0.
        # np.sign(0) is 0, so we handle that case.
        x_star = np.array([-5.0 if a_est[i] > 0 else 5.0 if a_est[i] < 0 else 0.0 for i in range(dim)])
        
        # 4. Generate all vertices and sort by predicted value
        vertices = []
        for mask in range(2 ** dim):
            v = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(dim)])
            vertices.append(v)

        predicted_values = []
        for v in vertices:
            pred_val = np.dot(a_est, v) + f0
            predicted_values.append((pred_val, v))
        
        # Sort by predicted value (ascending)
        predicted_values.sort(key=lambda x: x[0])

        # 5. Evaluate vertices in sorted order
        # The first vertex should be x_star (or very close in prediction).
        # We evaluate all 32 to be safe, but the first one should be the min.
        for pred_val, v in predicted_values:
            if time.time() > deadline or calls_made >= budget:
                break
            
            fv = func(v)
            calls_made += 1
            
            if fv < best_fx:
                best_fx = fv
                best_x = v.copy()

        # 6. No local search needed for a linear function.
        # The minimum is exactly at a vertex.
        # If we have budget left, we can just return.
        
        return best_x, best_fx

# EVOLVE-BLOCK-END
