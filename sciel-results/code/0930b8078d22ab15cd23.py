import numpy as np
import time


class Optimizer:
    """Linear-slope (BBOB f5) optimizer: vertex search then (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0
        dim = self.dim

        # Phase 1: Determine optimal vertex using coordinate-wise bound probes.
        # For a linear function f(x) = c^T x + d, the minimum on [-5,5]^dim is at a vertex.
        # We determine the sign of each component c_i by comparing f at x with x_i=5 vs x_i=-5
        # (all other coordinates at 0). This takes 2*dim + 1 calls (including origin).
        
        x_base = np.zeros(dim)
        f_base = func(x_base)
        best_x = x_base.copy()
        best_f = f_base
        calls = 1

        # Determine vertex signs
        s = np.zeros(dim)
        for i in range(dim):
            if time.time() > deadline:
                break
            x_pos = x_base.copy()
            x_pos[i] = 5.0
            f_pos = func(x_pos)
            calls += 1
            if f_pos < best_f:
                best_f = f_pos
                best_x = x_pos.copy()

            x_neg = x_base.copy()
            x_neg[i] = -5.0
            f_neg = func(x_neg)
            calls += 1
            if f_neg < best_f:
                best_f = f_neg
                best_x = x_neg.copy()

            # Decide direction: if f_pos < f_neg, c_i * 5 < c_i * (-5) => c_i > 0, so min at -5
            # Wait: f(x) = c^T x + d. f(x_pos) = 5*c_i + d (others 0). f(x_neg) = -5*c_i + d.
            # If f_pos < f_neg: 5*c_i < -5*c_i => 10*c_i < 0 => c_i < 0. So min at x_i = 5.
            # If f_neg < f_pos: c_i > 0. So min at x_i = -5.
            if f_pos < f_neg:
                s[i] = 5.0
            else:
                s[i] = -5.0

            # Evaluate the partial vertex to track best
            if i > 0:
                x_partial = s.copy()
                f_partial = func(x_partial)
                calls += 1
                if f_partial < best_f:
                    best_f = f_partial
                    best_x = x_partial.copy()

        # Evaluate the full vertex
        if time.time() <= deadline:
            x_vertex = s.copy()
            f_vertex = func(x_vertex)
            calls += 1
            if f_vertex < best_f:
                best_f = f_vertex
                best_x = x_vertex.copy()

        # Phase 2: Polish with (1+1)-ES starting from best_x
        # Since the function is linear, we should already be at the optimum (vertex).
        # But due to hidden shifts/rotations, the "linear" function might be slightly off
        # or the box might clip in a way that the true optimum is on the boundary.
        # A few ES steps can help if there's any non-linearity or if we're slightly off.
        
        rng = np.random.default_rng(42)
        sigma = 0.5
        x = best_x.copy()
        fx = best_f

        remaining = self.budget - calls
        for _ in range(remaining):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                best_x = x.copy()
                best_f = fx
                sigma *= 1.2
            else:
                sigma *= 0.85
            if sigma < 1e-10:
                break

        # Use remaining budget to evaluate best_x (in case of any discrepancy)
        # Actually, the return value is ignored, but we must not exceed budget.
        # The task says "Every call is recorded". We've made 'calls' evaluations.
        # We need to make sure we don't exceed budget.
        # The loop above handles this by checking remaining budget.
        
        return best_x, best_f
