"""Rewritten optimizer for BBOB f20 (Schwefel) using Deterministic Space-Filling Search."""
import numpy as np
import time


class Optimizer:
    """Structural change: Deterministic Halton Sequence Space-Filling Search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270  # Safety margin for 300s limit

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64).flatten()
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            if not np.isfinite(fx):
                fx = np.inf
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Generate Halton sequence points
        # We need `budget` points in dim dimensions
        # Halton sequence is defined by bases (primes)
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f = f / base
                r = r + f * (i % base)
                i = i // base
            return r

        # Primes for dimensions 1 to dim
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        # We generate indices 1 to budget
        # Note: Halton sequence is typically 1-indexed for the first point to be (0,0,...) or (1/base, ...)
        # Using 1-based indexing ensures the first point is not at the corner (0,0,..) which might be bad if the optimum is shifted.
        # Actually, standard Halton starts at 1.
        
        for i in range(1, budget + 1):
            if time.time() > deadline:
                break
            point = np.zeros(dim)
            for d in range(dim):
                base = primes[d]
                point[d] = halton(i, base)
            
            # Map [0, 1] to [-5, 5]
            x_mapped = point * 10.0 - 5.0
            
            safe_call(x_mapped)

        return best_x, best_f
