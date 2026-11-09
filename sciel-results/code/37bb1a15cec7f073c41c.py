"""Rewritten optimizer for BBOB f20 (Schwefel) with Deterministic LHS and Local Refinement."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-strategy optimizer: Structured Scan + Latin Hypercube Sampling + L-BFGS-B Refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270  # Safety margin for 300s limit

        rng = np.random.default_rng(42)
        call_count = 0

        def safe_call(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            x = np.asarray(x, dtype=np.float64).flatten()
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            if not np.isfinite(fx):
                return np.inf
            call_count += 1
            return fx

        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Structured initial scan (uses ~30 calls)
        # Try origin, axis-aligned points
        candidates = [np.zeros(dim)]
        for axis in range(dim):
            for val in [-4, -2, 0, 2, 4]:
                pt = np.zeros(dim)
                pt[axis] = val
                candidates.append(pt)
        
        for pt in candidates:
            if call_count >= budget or time.time() > deadline:
                break
            f = safe_call(pt)
            update_best(pt, f)

        # Phase 2: Deterministic Latin Hypercube Sampling (LHS)
        # We generate a fixed number of LHS points to ensure global coverage.
        # LHS ensures that each dimension is divided into N equal intervals, and one point is placed in each interval.
        # This is much more efficient for global coverage than random or DE in high dimensions if N is moderate.
        n_lhs = 150 # Use 150 calls for LHS
        if budget - call_count > n_lhs:
            # Generate LHS points
            # We use a fixed seed for determinism and consistency
            lhs_rng = np.random.default_rng(123)
            # For each dimension, we create a permutation of [0, 1, ..., n-1]
            # and map them to [0, 1) then to [-5, 5]
            # This is a simple LHS implementation.
            
            # Better: use scipy.stats.qmc or manual permutation
            # Manual permutation LHS:
            points = np.zeros((n_lhs, dim))
            for j in range(dim):
                perm = lhs_rng.permutation(n_lhs)
                # Map [0, n) to [-5, 5] with some jitter to avoid grid alignment?
                # Standard LHS: x = (perm + U) / n * (high - low) + low
                u = lhs_rng.uniform(0, 1, n_lhs)
                points[:, j] = ((perm + u) / n_lhs) * 10.0 - 5.0
            
            # Evaluate all LHS points
            lhs_results = []
            for i in range(n_lhs):
                if call_count >= budget or time.time() > deadline:
                    break
                pt = points[i]
                f = safe_call(pt)
                if np.isfinite(f):
                    lhs_results.append((f, pt))
            
            # Sort by fitness
            lhs_results.sort(key=lambda x: x[0])
            
            # Phase 3: Local Refinement from Top K LHS points
            # We take the top 5 points from LHS and the current best
            top_k = 5
            candidates_to_refine = []
            
            # Add current best
            if best_f < np.inf:
                candidates_to_refine.append(best_x)
            
            # Add top LHS points
            for f, pt in lhs_results[:top_k]:
                # Avoid duplicates (optional, but good)
                candidates_to_refine.append(pt)
            
            # Refine each candidate using L-BFGS-B
            # Budget remaining: budget - call_count
            # Split among candidates
            num_refine = len(candidates_to_refine)
            if num_refine > 0 and budget - call_count > 10:
                budget_per_refine = (budget - call_count) // num_refine
                # Ensure at least 10 calls per refinement
                budget_per_refine = max(10, budget_per_refine)
                
                for cx in candidates_to_refine:
                    if time.time() > deadline or call_count >= budget:
                        break
                    
                    # Use L-BFGS-B for local optimization
                    # It is fast and effective for smooth functions like Schwefel (after finding the basin)
                    try:
                        res = minimize(
                            lambda x: safe_call(x),
                            cx,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * dim,
                            options={'maxiter': 100, 'maxfun': budget_per_refine}
                        )
                        if res.fun < best_f:
                            update_best(res.x, res.fun)
                    except Exception:
                        pass

            # Phase 4: Final Polish with (1+1)-ES or CMA-ES if budget remains
            # L-BFGS-B might stop at a local minimum or boundary. A small step-size search can help.
            if time.time() < deadline and budget - call_count > 20 and best_f < np.inf:
                x = best_x.copy()
                fx = best_f
                sigma = 0.2 # Start with small sigma
                remaining = budget - call_count
                for _ in range(remaining):
                    if time.time() > deadline or call_count >= budget:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = safe_call(y)
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.1
                        update_best(x, fx)
                    else:
                        sigma *= 0.9
                    if sigma < 1e-8:
                        break

        return best_x, best_f
