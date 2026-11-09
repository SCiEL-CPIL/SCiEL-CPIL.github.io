"""Rewritten program for BBOB f5 (Linear slope) — exploit linear structure via bound sampling + vertex search."""

# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit the linear structure of f5 by sampling all 32 vertices and refining.
    
    f5 is a linear function after rotation/shift. Its minimum over [-5,5]^d
    lies at a vertex. We enumerate all 2^d = 32 vertices within budget,
    pick the best, then use a lightweight refinement to handle any
    non-linearity or numerical edge cases.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def _eval(x):
            nonlocal best_x, best_f, calls_made
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Enumerate all 2^d vertices ---
        # For a linear function f(x) = c + g^T x on [-5,5]^d, the global minimum
        # is at a vertex where each coordinate is either -5 or +5.
        # With d=5, there are 32 vertices — well within the 500-call budget.
        
        # Generate all 2^d vertices
        n_vertices = 1 << d
        vertices = np.zeros((n_vertices, d))
        for i in range(n_vertices):
            for j in range(d):
                if (i >> j) & 1:
                    vertices[i, j] = 5.0
                else:
                    vertices[i, j] = -5.0

        # Evaluate all vertices
        for i in range(n_vertices):
            if calls_made >= budget or time.time() > deadline:
                break
            _eval(vertices[i])

        # --- Phase 2: Refinement with (1+1)-ES around the best vertex ---
        # If the function is purely linear, the vertex IS the optimum.
        # But if there's any slight non-linearity or the hidden transformation
        # introduces small deviations, a local refinement can help.
        
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        rng = np.random.default_rng(42)

        # Adaptive (1+1)-ES with restarts if stuck
        no_improve = 0
        max_no_improve = 20

        while calls_made < budget and time.time() < deadline:
            y = x + sigma * rng.standard_normal(d)
            y = np.clip(y, -5.0, 5.0)
            fy = _eval(y)

            if fy < fx:
                x = y.copy()
                fx = fy
                no_improve = 0
                sigma = min(sigma * 1.05, 5.0)
            else:
                sigma *= 0.9
                no_improve += 1

            if no_improve >= max_no_improve and sigma < 1e-6:
                break
            if sigma < 1e-8:
                break

        return x, best_f
# EVOLVE-BLOCK-END
