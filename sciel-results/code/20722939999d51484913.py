"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Surrogate-assisted global optimizer with local refinement for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # State
        best_x = None
        best_f = np.inf
        call_count = 0
        xs = []  # list of evaluated points
        fs = []  # list of function values

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            call_count += 1
            xs.append(x.copy())
            fs.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial exploration with diverse starts
        # Use a mix of random points and structured grid
        n_initial = min(50, budget)
        # Latin hypercube-style sampling for diversity
        for i in range(n_initial):
            if time.time() > deadline:
                break
            if i < dim * 2:
                # Coordinate-wise exploration
                x = np.zeros(dim)
                axis = i % dim
                direction = 1 if (i // dim) % 2 == 0 else -1
                x[axis] = direction * 4.0
                x = np.clip(x, -5, 5)
            elif i < dim * 4:
                # Diagonal exploration
                x = np.zeros(dim)
                axis1 = (i // 2) % dim
                axis2 = (i // 2 + 1) % dim
                direction = 1 if (i // (2 * dim)) % 2 == 0 else -1
                x[axis1] = direction * 3.5
                x[axis2] = direction * 3.5
                x = np.clip(x, -5, 5)
            else:
                x = rng.uniform(-4.5, 4.5, dim)
            evaluate(x)

        if time.time() > deadline or call_count >= budget:
            return best_x, best_f

        # Phase 2: Surrogate-guided exploration
        # Use a simple RBF-like surrogate: weighted average of nearby points
        # with inverse distance weighting, plus exploration noise

        def surrogate_predict(x, exploration=0.0):
            """Simple surrogate: inverse distance weighted interpolation + exploration."""
            if len(xs) == 0:
                return 0.0
            x = np.array(x)
            dists = np.array([np.linalg.norm(x - xi) for xi in xs])
            # Inverse distance weighting
            eps = 1e-6
            weights = 1.0 / (dists + eps) ** 2
            weights /= weights.sum()
            pred = np.sum(weights * np.array(fs))
            # Add exploration term (uncertainty based on distance to nearest point)
            if exploration > 0:
                nearest = np.min(dists)
                pred -= exploration * np.exp(-nearest * 0.5)
            return pred

        def surrogate_uncertainty(x):
            """Uncertainty estimate based on distance to nearest evaluated point."""
            if len(xs) == 0:
                return 10.0
            x = np.array(x)
            dists = np.array([np.linalg.norm(x - xi) for xi in xs])
            return np.min(dists)

        def expected_improvement(x, best_val, exploration=1.0):
            """EI-like acquisition: predicted improvement + uncertainty bonus."""
            pred = surrogate_predict(x, exploration=0.0)
            unc = surrogate_uncertainty(x)
            # Improvement over current best
            improvement = max(0, best_val - pred)
            # Exploration bonus
            return improvement + exploration * unc

        # Phase 2a: Surrogate-guided global search
        n_surrogate = min(150, budget - call_count)
        for _ in range(n_surrogate):
            if time.time() > deadline or call_count >= budget:
                break

            # Generate candidate points: mix of random and perturbed best
            candidates = []
            # 10 random points
            for _ in range(10):
                candidates.append(rng.uniform(-4.8, 4.8, dim))
            # 5 points near current best
            for _ in range(5):
                if best_x is not None:
                    candidates.append(np.clip(best_x + rng.normal(0, 0.5, dim), -5, 5))
            # 5 points near second best
            if len(fs) >= 2:
                sorted_idx = np.argsort(fs)
                second_best_x = xs[sorted_idx[1]]
                for _ in range(5):
                    candidates.append(np.clip(second_best_x + rng.normal(0, 0.7, dim), -5, 5))

            # Evaluate acquisition function for each candidate
            best_candidate = None
            best_acq = -np.inf
            for c in candidates:
                acq = expected_improvement(c, best_f, exploration=1.0)
                if acq > best_acq:
                    best_acq = acq
                    best_candidate = c

            if best_candidate is not None:
                evaluate(best_candidate)

        if time.time() > deadline or call_count >= budget:
            return best_x, best_f

        # Phase 3: Local refinement using L-BFGS-B from multiple starting points
        remaining_budget = budget - call_count
        if remaining_budget > 20:
            # Select top starting points for local optimization
            sorted_idx = np.argsort(fs)
            n_starts = min(5, len(sorted_idx))

            for idx in sorted_idx[:n_starts]:
                if time.time() > deadline or call_count >= budget:
                    break
                if remaining_budget < 10:
                    break

                x0 = xs[idx].copy()
                remaining_budget = budget - call_count

                # Count calls made by L-BFGS-B
                calls_used = [0]

                def lbfun(x):
                    calls_used[0] += 1
                    if calls_used[0] > remaining_budget:
                        # Return a large value to stop optimization
                        return 1e10
                    x_clipped = np.clip(x, -5, 5)
                    return float(func(x_clipped))

                try:
                    result = minimize(
                        lbfun,
                        x0,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': min(remaining_budget, 50),
                                 'ftol': 1e-10,
                                 'gtol': 1e-6}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5)
                    # Update call count
                    call_count += calls_used[0]
                    remaining_budget = budget - call_count
                except Exception:
                    pass

        # Phase 4: Final polish - a few more local steps from best
        if time.time() < deadline and call_count < budget:
            polish_steps = min(20, budget - call_count)
            sigma = 0.1
            for _ in range(polish_steps):
                if time.time() > deadline or call_count >= budget:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                if fy < best_f:
                    sigma *= 1.3
                else:
                    sigma *= 0.7

        return best_x, best_f
# EVOLVE-BLOCK-END
