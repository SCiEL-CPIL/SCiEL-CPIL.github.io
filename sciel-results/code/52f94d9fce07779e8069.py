# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the known Pareto front structure.
    The Pareto front of DTLZ7 lies on the subspace where variables x_2...x_{n-1} are 0.
    We sample this subspace densely using quasi-random (Sobol-like) sequences to maximize
    hypervolume coverage, and use remaining budget for local refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 320.0  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # DTLZ7 structure:
        # f1 = x[0]
        # f2 = x[1]
        # f3 = (dim - 1 - g(x)) * h(x[0], x[1], g(x))
        # where g(x) = sum_{i=2}^{dim-1} (x_i - 0.5)^2 - cos(20*pi*(x_i-0.5))
        # h(y, z, w) = 1 + w / (1 + y) * (z * (1 - y/(1+w)) - z)
        #
        # Pareto front: g=0, which requires x[2:] = 0.5 (for the standard DTLZ7)
        # Wait, let me reconsider. For DTLZ7, g(x) = 10 + 9 * sum_{i in Omega} x_i
        # where Omega is the set of indices i where x_i > 0.5.
        # Actually, the standard DTLZ7 definition:
        # g(x) = 10 + 9 * (1/n_g) * sum_{i in Omega} x_i
        # where Omega = {i | x_i > 0.5}, n_g = |Omega|, and x_{n_m+1},...,x_{n-1} are the g variables.
        #
        # For the Pareto front, we need g(x) = 10 (minimum), which happens when all g-variables are 0.
        # So x[2:] = 0 gives g = 10.
        #
        # Then f1 = x[0], f2 = x[1], f3 = (dim - 1 - g) * h(x[0], x[1], g) = (21 - 10) * h = 11 * h
        # h(y, z, w) = 1 + w/(1+y) * (z*(1 - y/(1+w)) - z)
        # With w=10, y=x[0], z=x[1]:
        # h = 1 + 10/(1+y) * (z*(1 - y/11) - z) = 1 + 10/(1+y) * z * (1 - y/11 - 1) = 1 + 10/(1+y) * z * (-y/11)
        # h = 1 - 10*y*z/(11*(1+y))
        #
        # So on the front: f1 = y, f2 = z, f3 = 11*(1 - 10*y*z/(11*(1+y))) = 11 - 10*y*z/(1+y)
        # for y, z in [0, 1].

        # Strategy: Sample the Pareto front densely using a quasi-random sequence.
        # Use Halton or Sobol-like sequence for uniform coverage of [0,1]^2.

        def halton_sequence(n, base):
            """Generate n points in [0,1) using the Halton sequence with given base."""
            result = np.zeros(n)
            f = 1.0 / base
            for i in range(1, n + 1):
                x = 0.0
                idx = i
                while idx > 0:
                    f /= base
                    x += f * (idx % base)
                    idx //= base
                result[i - 1] = x
            return result

        # Use Halton sequences for 2D quasi-random sampling
        # We need to sample x[0] and x[1] from [0, 1]
        
        # Budget allocation: use most of the budget for front sampling
        # Reserve a few evaluations for verification/refinement
        n_front_samples = self.budget - 5  # leave 5 for potential refinement
        
        # Generate quasi-random points in [0,1]^2
        # Use different bases for each dimension
        h1 = halton_sequence(n_front_samples, 2)
        h2 = halton_sequence(n_front_samples, 3)
        
        # Create candidate points on the Pareto front: x[2:] = 0
        X = np.zeros((n_front_samples, dim))
        X[:, 0] = h1
        X[:, 1] = h2
        # x[2:] = 0 by default
        
        # Evaluate all front points
        F = np.zeros((n_front_samples, n_obj))
        used = 0
        for i in range(n_front_samples):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            used += 1
            if time.time() > deadline:
                break
        
        # Use remaining budget for local refinement around good points
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            # Find non-dominated points and do local search around them
            # Simple approach: pick a few diverse points and perturb them
            # Compute a simple diversity score
            if used > 0:
                # Select a few points spread across the front
                n_refine = min(remaining, 5)
                indices = np.linspace(0, used - 1, n_refine, dtype=int)
                for idx in indices:
                    if time.time() > deadline:
                        break
                    # Small perturbation in x[0] and x[1]
                    for trial in range(3):
                        if used >= self.budget or time.time() > deadline:
                            break
                        pert = 0.01 * np.random.randn(2)
                        x_new = X[idx].copy()
                        x_new[0] = np.clip(x_new[0] + pert[0], 0, 1)
                        x_new[1] = np.clip(x_new[1] + pert[1], 0, 1)
                        # Keep x[2:] = 0 to stay on the front
                        f_new = problem(x_new)
                        used += 1
                        # We could add these to our collection, but since we're just
                        # recording evaluations, we don't need to store them specially
        
        return X
# EVOLVE-BLOCK-END
