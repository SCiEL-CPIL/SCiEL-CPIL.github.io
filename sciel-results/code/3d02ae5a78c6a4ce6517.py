"""Improved BBOB f22 optimizer: Multi-Start Nelder-Mead with adaptive step sizes and final polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start Nelder-Mead with adaptive step sizes and final polish for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 285.0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0
        n = self.dim

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        def nelder_mead(x_start, f_start, max_steps=80, initial_delta=0.5, tol=1e-12):
            """Nelder-Mead with adaptive step size and convergence tolerance."""
            nonlocal calls_made, best_x, best_f
            dim = self.dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            # Build initial simplex
            simplex = [x_start.copy()]
            simplex_f = [f_start]
            for i in range(dim):
                xi = x_start.copy()
                xi[i] += initial_delta
                xi = np.clip(xi, -5, 5)
                if np.allclose(xi, x_start):
                    xi[i] = x_start[i] + initial_delta * 0.1
                    xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                if not check_time() or calls_made >= self.budget:
                    return
                fx = safe_func(simplex[i])
                simplex_f.append(fx)

            # Adaptive step size tracking
            prev_spread = np.max(simplex_f) - np.min(simplex_f)
            if prev_spread < 1e-15:
                prev_spread = 1e-15

            for step in range(max_steps):
                if not check_time() or calls_made >= self.budget:
                    break

                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Convergence check
                spread = np.max(simplex_f) - np.min(simplex_f)
                if spread < tol:
                    break

                # Adaptive step size: shrink if progress is slow
                if spread < prev_spread * 0.1:
                    initial_delta *= 0.5
                    if initial_delta < 1e-8:
                        break
                prev_spread = spread

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = safe_func(xr)
                if not check_time() or calls_made >= self.budget:
                    break

                if fxr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = safe_func(xe)
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                elif fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = safe_func(xc)
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not check_time() or calls_made >= self.budget:
                                return
                            simplex_f[i] = safe_func(simplex[i])

            # Update global best from simplex
            best_in_simplex = np.argmin(simplex_f)
            if simplex_f[best_in_simplex] < best_f:
                best_f = simplex_f[best_in_simplex]
                best_x = simplex[best_in_simplex].copy()

        def final_polish(x_center, max_steps=50):
            """High-precision polish with very small initial step."""
            nonlocal calls_made, best_x, best_f
            if not check_time() or calls_made >= self.budget:
                return
            # Evaluate center
            fc = safe_func(x_center)
            if not check_time() or calls_made >= self.budget:
                return
            # Run NM with very small delta for precision
            nelder_mead(x_center.copy(), fc, max_steps=max_steps, initial_delta=0.05, tol=1e-14)

        # Phase 1: Structured exploration with proper tracking
        # Use a mix of structured and random points, tracking all (x, f) pairs
        n_explore = 40
        if n_explore > self.budget - 10:
            n_explore = max(10, self.budget - 10)

        candidates = []  # List of (x, f) tuples

        # Generate diverse starting points: combination of grid-like and random
        # Use stratified sampling for better coverage
        for i in range(n_explore):
            if not check_time() or calls_made >= self.budget:
                break
            if i < n_explore // 2:
                # Stratified: divide each dimension into 3 bins, pick one per bin
                x = np.zeros(n)
                for d in range(n):
                    bin_idx = (i + d * 7) % 3
                    x[d] = -5 + (bin_idx + 0.5) * (10.0 / 3) + rng.uniform(-1, 1) * (5.0 / 3)
                x = np.clip(x, -5, 5)
            else:
                # Random uniform
                x = rng.uniform(-5, 5, size=n)
            fx = safe_func(x)
            candidates.append((x.copy(), fx))

        # Sort candidates by fitness
        candidates.sort(key=lambda xf: xf[1])

        # Phase 2: Multi-start NM on top-K candidates
        k_top = min(5, len(candidates))
        for i in range(k_top):
            if not check_time() or calls_made >= self.budget:
                break
            x_start, f_start = candidates[i]
            # Use adaptive initial delta based on rank
            delta = 0.5 * (1.0 - 0.1 * i)
            nelder_mead(x_start.copy(), f_start, max_steps=60, initial_delta=delta)

        # Phase 3: Additional random restarts if budget remains
        remaining = self.budget - calls_made
        if remaining > 20 and check_time():
            n_restarts = min(3, remaining // 30)
            for _ in range(n_restarts):
                if not check_time() or calls_made >= self.budget:
                    break
                x_rand = rng.uniform(-5, 5, size=n)
                f_rand = safe_func(x_rand)
                if check_time() and calls_made < self.budget:
                    nelder_mead(x_rand.copy(), f_rand, max_steps=40, initial_delta=0.5)

        # Phase 4: Final polish on best point
        if check_time() and calls_made < self.budget:
            final_polish(best_x.copy(), max_steps=min(40, self.budget - calls_made))

        # Phase 5: One more polish if budget allows
        if check_time() and calls_made < self.budget and (self.budget - calls_made) > 10:
            final_polish(best_x.copy(), max_steps=min(20, self.budget - calls_made))

        return best_x, best_f

# EVOLVE-BLOCK-END
