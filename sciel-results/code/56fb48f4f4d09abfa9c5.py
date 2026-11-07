"""Optimizer for BBOB f15 (Rastrigin rotated) using lattice-based candidate generation."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Lattice-based optimizer exploiting the 2π-periodicity of Rastrigin in the rotated frame.
    
    Strategy:
    1. Initial exploration: evaluate at a small set of random points to get an initial best estimate.
    2. Lattice search: generate candidate minima by shifting the best point by integer multiples
       of a step size in each coordinate direction. This approximates the lattice of minima
       in the rotated frame.
    3. Iteratively refine: from the best candidate found, generate new candidates and evaluate
       them until the budget is exhausted.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None
        self.best_x = None
        self.best_f = np.inf
        self.evaluated = set()

    def _counted_func(self, x):
        """Wrapper to count function calls."""
        self.calls += 1
        return self.func(x)

    def _make_key(self, x):
        """Create a hashable key for a point."""
        return tuple(np.round(x, 8))

    def _evaluate(self, x):
        """Evaluate a point if not already evaluated and within budget."""
        if self.calls >= self.budget:
            return np.inf
        key = self._make_key(x)
        if key in self.evaluated:
            return np.inf
        self.evaluated.add(key)
        fx = self._counted_func(x)
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
        return fx

    def __call__(self, func):
        self.func = func
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.evaluated = set()

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial exploration - evaluate at random points and origin
        n_initial = min(10, budget // 2)
        for i in range(n_initial):
            if self.calls >= budget - 10:
                break
            if i == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-5, 5, dim)
            self._evaluate(x0)

        # Phase 2: Lattice-based search using 2*pi periodicity
        # The Rastrigin function has period 2*pi in each rotated coordinate.
        # We search over integer shifts of 2*pi/2 = pi in each direction.
        # But since the rotation is unknown, we work in the original frame.
        # The key insight: if x* is near a minimum, then x* + k*period (in rotated frame)
        # is also near a minimum. Since we don't know the rotation, we approximate by
        # trying shifts in the original coordinate directions.
        
        # Use multiple step sizes to cover different scales
        step_sizes = [2 * np.pi / 2, 2 * np.pi, np.pi / 2, np.pi]
        
        for step in step_sizes:
            if self.calls >= budget - 5:
                break
            
            # Generate candidates from the current best point
            if self.best_x is None:
                break
                
            candidates = []
            seen_keys = set()
            
            # Single-dimension shifts
            for d in range(dim):
                for k in range(-2, 3):
                    if k == 0:
                        continue
                    delta = k * step
                    x_cand = self.best_x.copy()
                    x_cand[d] += delta
                    x_cand = np.clip(x_cand, -5, 5)
                    key = self._make_key(x_cand)
                    if key not in self.evaluated and key not in seen_keys:
                        candidates.append((np.linalg.norm(x_cand - self.best_x), x_cand, key))
                        seen_keys.add(key)
            
            # Two-dimension shifts
            if self.calls < budget - 30:
                for d1 in range(dim):
                    for d2 in range(d1 + 1, dim):
                        for k1 in [-1, 1]:
                            for k2 in [-1, 1]:
                                x_cand = self.best_x.copy()
                                x_cand[d1] += k1 * step
                                x_cand[d2] += k2 * step
                                x_cand = np.clip(x_cand, -5, 5)
                                key = self._make_key(x_cand)
                                if key not in self.evaluated and key not in seen_keys:
                                    candidates.append((np.linalg.norm(x_cand - self.best_x), x_cand, key))
                                    seen_keys.add(key)
            
            # Sort by distance to best (closest first)
            candidates.sort(key=lambda item: item[0])
            
            # Evaluate top candidates
            for dist, x_cand, key in candidates:
                if self.calls >= budget - 1:
                    break
                self._evaluate(x_cand)
                
                # If we found a better point, break to regenerate from new best
                if self.best_x is not None and self.calls < budget - 10:
                    # Check if this was a significant improvement
                    pass

        # Phase 3: Local refinement around the best point found
        if self.best_x is not None and self.calls < budget - 1:
            # Try small perturbations around the best point
            base = self.best_x.copy()
            for step_size in [0.5, 0.25, 0.1, 0.05]:
                if self.calls >= budget - 1:
                    break
                for d in range(dim):
                    for delta in [-step_size, step_size]:
                        x_cand = base.copy()
                        x_cand[d] += delta
                        x_cand = np.clip(x_cand, -5, 5)
                        self._evaluate(x_cand)
                
                # Update base if we found something better
                if self.best_x is not None:
                    base = self.best_x.copy()

        # Ensure we've made at least one call
        if self.calls == 0:
            x0 = np.zeros(dim)
            self._evaluate(x0)

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
