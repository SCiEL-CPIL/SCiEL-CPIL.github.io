"""Black-box continuous optimizer on the BBOB-style function f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for a linear (or near-linear) function on [-5, 5]^d.
    Strategy:
      1. Use d+1 simplex probes to estimate the gradient direction and intercept.
      2. For a strictly linear function, the minimum on the box is at a vertex.
         We can compute the optimal vertex analytically from the gradient.
      3. Evaluate that vertex, then use remaining budget to do a local
         refinement (e.g., a few random perturbations around the vertex) in
         case the function is not perfectly linear or there's numerical noise.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        call_count = 0

        def _call(x):
            nonlocal call_count
            call_count += 1
            return func(x)

        # Step 1: Simplex gradient estimation
        # Use a small step size for the probes
        base = np.zeros(d)
        f0 = _call(base)

        step = 0.5  # probe step size
        probes = [base]
        fvals = [f0]

        for i in range(d):
            p = base.copy()
            p[i] += step
            fp = _call(p)
            probes.append(p)
            fvals.append(fp)

        # Estimate gradient: (f(p_i) - f0) / step for each coordinate
        grad = np.zeros(d)
        for i in range(d):
            grad[i] = (fvals[i + 1] - f0) / step

        # For a linear function f(x) = w^T x + b, the minimum on [-5,5]^d is
        # at x_i = -5 if w_i > 0, x_i = 5 if w_i < 0, x_i = 0 if w_i == 0.
        vertex = np.sign(-grad) * 5
        # Handle zero gradient components: leave at 0 (or any value)
        vertex[grad == 0] = 0

        f_vertex = _call(vertex)
        best_x = vertex
        best_f = f_vertex

        # Step 2: If we still have budget, do a few local searches to handle
        # non-linearities or noise. Use a small step around the vertex.
        remaining = budget - call_count
        if remaining > 0:
            # Try a few random perturbations
            local_step = 0.1
            for _ in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                perturbation = rng.uniform(-local_step, local_step, d)
                candidate = np.clip(best_x + perturbation, -5, 5)
                fc = _call(candidate)
                if fc < best_f:
                    best_f = fc
                    best_x = candidate

        return best_x, best_f
# EVOLVE-BLOCK-END
