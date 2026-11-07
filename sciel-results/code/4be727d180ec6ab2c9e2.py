# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using MOEA/D-style decomposition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        def is_deadline():
            return (deadline - time.time()) < 3.0

        rng = np.random.default_rng(0)

        # Generate weight vectors on the simplex for decomposition
        # Use a structured set of weights
        n_weights = 30
        W = []
        # Add extreme points
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            W.append(w)
        # Add uniform
        W.append(np.ones(n_obj) / n_obj)
        # Add random weights to fill up to n_weights
        while len(W) < n_weights:
            w = rng.random(n_obj)
            w /= w.sum()
            W.append(w)
        W = np.array(W)

        # Initial sampling: spread points across the domain
        n_init = min(50, budget)
        X = np.zeros((n_init, dim))
        for i in range(n_init):
            X[i] = lo + rng.random(dim) * (hi - lo)

        F = np.array([problem(x) for x in X])
        used = n_init

        # Assign each initial point to the closest weight
        def assign_weights(X, F, W):
            """Assign each point to the weight vector it best satisfies."""
            n = len(X)
            assignments = np.zeros(n, dtype=int)
            for i in range(n):
                # Compute weighted sum for each weight
                scores = F[i] @ W.T  # shape (n_weights,)
                assignments[i] = np.argmin(scores)
            return assignments

        assignments = assign_weights(X, F, W)

        # Main loop: iteratively refine
        max_iter = 50
        for it in range(max_iter):
            if is_deadline() or used >= budget:
                break

            # For each weight, find the best point and try to improve it
            for wi in range(n_weights):
                if is_deadline() or used >= budget:
                    break

                # Find points assigned to this weight
                mask = (assignments == wi)
                if np.any(mask):
                    # Get the best point for this weight
                    idxs = np.where(mask)[0]
                    scores = F[idxs] @ W[wi]
                    best_local = idxs[np.argmin(scores)]
                    x0 = X[best_local].copy()
                else:
                    # No point assigned, use random start
                    x0 = lo + rng.random(dim) * (hi - lo)

                # Try a few local perturbations
                n_perturb = 2
                for p in range(n_perturb):
                    if is_deadline() or used >= budget:
                        break

                    # Gaussian perturbation
                    sigma = 0.1 * (hi - lo)
                    x_new = x0 + rng.normal(0, sigma, dim)
                    x_new = np.clip(x_new, lo, hi)

                    fx_new = problem(x_new)
                    used += 1

                    if used > budget:
                        break

                    # Check if this is non-dominated
                    is_dominated = False
                    for j in range(len(F)):
                        if np.all(F[j] <= fx_new) and np.any(F[j] < fx_new):
                            is_dominated = True
                            break

                    if not is_dominated:
                        # Add to population
                        X = np.vstack([X, x_new])
                        F = np.vstack([F, fx_new])
                        assignments = np.append(assignments, wi)
                        x0 = x_new  # Continue from this new point

                    # Also try a larger step
                    if not is_deadline() and used < budget:
                        sigma_large = 0.3 * (hi - lo)
                        x_large = x0 + rng.normal(0, sigma_large, dim)
                        x_large = np.clip(x_large, lo, hi)

                        fx_large = problem(x_large)
                        used += 1

                        if used <= budget:
                            is_dominated_large = False
                            for j in range(len(F)):
                                if np.all(F[j] <= fx_large) and np.any(F[j] < fx_large):
                                    is_dominated_large = True
                                    break

                            if not is_dominated_large:
                                X = np.vstack([X, x_large])
                                F = np.vstack([F, fx_large])
                                assignments = np.append(assignments, wi)

            # Reassign weights after each iteration
            if len(X) > 0:
                assignments = assign_weights(X, F, W)

        return X

# EVOLVE-BLOCK-END
