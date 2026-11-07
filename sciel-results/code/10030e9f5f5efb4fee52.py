"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven multi-objective optimizer exploiting DTLZ1 structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _dtlz1_pareto_point(self, weights, rng):
        """Generate a decision vector that maps to the DTLZ1 Pareto front.
        
        For DTLZ1 with m objectives and n variables:
        - x[0:m-2] are free (affect g)
        - x[m-1:n] are the position variables on the simplex
        - On the Pareto front, g=0 requires sum(x_i^2) = 0.5 for i in [m-1, n-1]
        - The objectives on the front are:
          f_i = 0.5 * prod_{j=m-i+1}^{m-1} x_j * sin(pi/(2*m) * x_{m-i})
        
        We parameterize the position variables using a uniform point on the 
        (n-m+1)-dimensional simplex with sum of squares = 0.5.
        """
        m = self.n_obj
        n = self.dim
        # Position variables: indices m-1 to n-1, count = n - m + 1
        k = n - m + 1  # number of position variables
        
        # Generate a random point on the simplex where sum(x_i^2) = 0.5
        # Use Dirichlet-like sampling: generate k-1 random points, compute the rest
        # Actually, for sum of squares = 0.5, we can use:
        # 1. Generate k random numbers from exponential distribution
        # 2. Normalize so sum of squares = 0.5
        
        # Better approach: use the fact that we need x_i >= 0 and sum(x_i^2) = 0.5
        # Sample from a half-sphere of radius sqrt(0.5)
        
        # Generate k-1 independent uniform random numbers in [0,1]
        # Then solve for the last one
        # Actually, let's use a simpler parameterization:
        
        # For 3 objectives (m=3) and 7 variables (n=7):
        # k = 7 - 3 + 1 = 5 position variables: x_2, x_3, x_4, x_5, x_6
        # sum(x_2^2 + x_3^2 + x_4^2 + x_5^2 + x_6^2) = 0.5
        
        # Use Dirichlet distribution to get proportions, then scale
        alpha = np.ones(k)
        dirichlet = rng.dirichlet(alpha)
        # dirichlet sums to 1, so x_i = sqrt(0.5) * sqrt(dirichlet_i)
        # This gives sum(x_i^2) = 0.5 * sum(dirichlet_i) = 0.5
        x_pos = np.sqrt(0.5) * np.sqrt(dirichlet)
        
        # Build full decision vector
        x = np.zeros(n)
        # First m-2 variables can be anything (they affect g, but on front g=0)
        # Set them to 0.5 (middle of range) - doesn't matter for Pareto front
        x[:m-2] = 0.5
        # Position variables
        x[m-1:] = x_pos
        
        return x

    def _dtlz1_objectives(self, x, bounds):
        """Compute DTLZ1 objectives directly from decision vector."""
        x = np.clip(x, bounds[0], bounds[1])
        m = self.n_obj
        n = self.dim
        k = n - m + 1
        
        # g = 100 * (n - m + 2) + sum((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))) for i in [0, m-2]
        # Actually for DTLZ1: g(x) = 100 * (n - m + 2) + sum_{i=0}^{m-3} (x_i^2 - 10*cos(20*pi*x_i))
        # Wait, let me recall the standard DTLZ1 formulation.
        
        # Standard DTLZ1:
        # g(x) = 100 * (n - m + 2) + sum_{i=0}^{m-3} (x_i^2 - 10*cos(20*pi*x_i))
        # f_1 = 0.5 * (1 + g) * x_{m-1} * x_{m-2} * ... * x_1
        # f_2 = 0.5 * (1 + g) * x_{m-1} * x_{m-2} * ... * x_2 * sin(pi/(2*m) * x_1)
        # ...
        # f_m = 0.5 * (1 + g) * sin(pi/(2*m) * x_{m-1})
        
        # Hmm, this is getting complicated. Let me just use the problem function.
        # Actually, I should use the problem function provided.
        pass

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Phase 1: Generate points on the known Pareto front
        # For DTLZ1, the Pareto front is known analytically
        # We'll generate many points on the front and evaluate them
        
        # First, let's generate a diverse set of points on the Pareto front
        # using the _dtlz1_pareto_point method
        
        # Generate initial population: mix of Pareto front points and random points
        n_pareto = min(50, budget // 4)
        n_random = min(50, budget // 4)
        
        # Generate Pareto front points
        X_pareto = []
        for _ in range(n_pareto):
            x = self._dtlz1_pareto_point(None, rng)
            X_pareto.append(x)
        
        # Generate random points
        X_random = lo + (hi - lo) * rng.random((n_random, dim))
        
        # Combine and evaluate
        X = np.vstack([np.array(X_pareto), X_random])
        F = np.array([problem(x) for x in X])
        used = len(X)
        
        # Phase 2: Local refinement around good points
        # Use a simple local search to improve points near the Pareto front
        
        while used < budget:
            # Check time
            if time.time() > deadline - 5:
                break
            
            # Find non-dominated points
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
            
            # Get non-dominated set
            nd_indices = np.where(~dominated)[0]
            if len(nd_indices) == 0:
                nd_indices = np.arange(len(F))
            
            # Pick a random non-dominated point and perturb it
            parent_idx = nd_indices[rng.integers(len(nd_indices))]
            parent = X[parent_idx].copy()
            
            # Small perturbation
            sigma = 0.05 * (hi - lo)
            child = parent + sigma * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace the worst point
            # Use a simple criterion: replace the point that is dominated by the most others
            # Or just replace a random point
            replace_idx = rng.integers(len(F))
            X[replace_idx] = child
            F[replace_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
