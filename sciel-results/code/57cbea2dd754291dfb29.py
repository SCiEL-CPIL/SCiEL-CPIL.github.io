import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 290  # 10s safety margin before 300s limit

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = 0

        def safe_call(x):
            nonlocal calls
            if calls >= budget or time.time() > self.deadline:
                return None
            x = np.clip(x, -5, 5)
            val = func(x)
            calls += 1
            return val

        # Phase 1: Sobol-like global scan for robust initialization
        # Use a quasi-random sequence to spread points across the search space
        n_scan = min(30, budget - 10)
        rng = np.random.default_rng(123)

        # Generate Sobol-like points using a simple LHS (Latin Hypercube) approach
        # This gives good spread without needing scipy
        points = np.zeros((n_scan, dim))
        for i in range(n_scan):
            for d in range(dim):
                points[i, d] = -5 + 10 * ((i * (d + 1) + d) % n_scan + 0.5) / n_scan
        # Add small random jitter to break symmetry
        points += rng.uniform(-0.5, 0.5, size=(n_scan, dim))
        points = np.clip(points, -5, 5)

        best_x = None
        best_f = np.inf

        for i in range(n_scan):
            if time.time() > self.deadline or calls >= budget:
                break
            f = safe_call(points[i])
            if f is None:
                break
            if f < best_f:
                best_f = f
                best_x = points[i].copy()

        if best_x is None:
            # Fallback: evaluate origin
            f = safe_call(np.zeros(dim))
            if f is None:
                return np.zeros(dim), np.inf
            best_f = f
            best_x = np.zeros(dim).copy()

        # Phase 2: (1+1)-ES with anisotropic sigma adaptation
        # For Bent Cigar, one dimension is very stiff (narrow valley)
        # Use separate sigma per dimension, adapted via 1.5/0.87 rule

        # Initialize sigmas: small in all dims, will adapt
        sigma = np.full(dim, 0.5)
        x = best_x.copy()
        fx = best_f

        # Track success counts per dimension for adaptive sigma
        success_count = np.zeros(dim)
        total_count = np.zeros(dim)
        adaptation_window = 10  # adapt every 10 evaluations

        # Main (1+1)-ES loop
        while calls < budget and time.time() < self.deadline:
            # Generate offspring
            y = x + rng.normal(0, 1, dim) * sigma
            y = np.clip(y, -5, 5)

            fy = safe_call(y)
            if fy is None:
                break

            if fy < fx:
                # Success: move to offspring, increase sigma for successful dims
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Determine which dimensions contributed to improvement
                # Simple heuristic: increase sigma for all dims, but more for those
                # where the step was in the improving direction
                for d in range(dim):
                    success_count[d] += 1
                    total_count[d] += 1
            else:
                # Failure: decrease sigma
                for d in range(dim):
                    total_count[d] += 1

            # Adapt sigma periodically
            if np.any(total_count >= adaptation_window):
                for d in range(dim):
                    if total_count[d] >= adaptation_window:
                        rate = success_count[d] / total_count[d]
                        if rate > 0.5:
                            sigma[d] *= 1.5
                        elif rate < 0.3:
                            sigma[d] *= 0.87
                        # Keep sigma in reasonable bounds
                        sigma[d] = np.clip(sigma[d], 1e-8, 10.0)
                        success_count[d] = 0
                        total_count[d] = 0

            # Check for convergence
            if np.all(sigma < 1e-7):
                break

        # Phase 3: Local coordinate descent fine-tuning
        if calls < budget and time.time() < self.deadline:
            step = sigma.copy()
            min_step = 1e-8
            x_cd = best_x.copy()
            fx_cd = best_f
            no_progress = 0

            while calls < budget and time.time() < self.deadline:
                progress = False
                for d in range(dim):
                    if time.time() > self.deadline or calls >= budget:
                        break

                    # Try positive step
                    y = x_cd.copy()
                    y[d] += step[d]
                    y = np.clip(y, -5, 5)
                    fy = safe_call(y)
                    if fy is None:
                        break
                    if fy < fx_cd:
                        x_cd = y
                        fx_cd = fy
                        if fx_cd < best_f:
                            best_f = fx_cd
                            best_x = x_cd.copy()
                        progress = True
                    else:
                        # Try negative step
                        y = x_cd.copy()
                        y[d] -= step[d]
                        y = np.clip(y, -5, 5)
                        fy = safe_call(y)
                        if fy is None:
                            break
                        if fy < fx_cd:
                            x_cd = y
                            fx_cd = fy
                            if fx_cd < best_f:
                                best_f = fx_cd
                                best_x = x_cd.copy()
                            progress = True
                        else:
                            # No improvement in this dimension, shrink step
                            step[d] *= 0.5

                    step[d] = max(step[d], min_step)

                if not progress:
                    no_progress += 1
                    if no_progress > 3:
                        break
                    # Shrink all steps
                    step *= 0.5
                else:
                    no_progress = 0

        return best_x, best_f
