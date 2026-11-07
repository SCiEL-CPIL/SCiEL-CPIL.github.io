"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES + step-boundary coordinate bisection for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        DEADLINE = 300.0  # safety margin well under 330s wall-clock

        # Wrap func to count calls
        calls = [0]

        def f(x):
            calls[0] += 1
            return func(x)

        # --- Phase 1: CMA-ES global search ---
        # Initialize with a broad start
        rng = np.random.default_rng(42)
        x = np.zeros(dim)  # centered start; hidden shift is in [-4,4]
        sigma = 2.5
        pop = max(4 + int(3 * np.log(dim)), 8)  # population size

        # CMA-ES parameters
        cc = (4 + dim / (dim + 4)) / ((dim + 4) + 4 + dim / (dim + 4))
        cs = (dim + 1.7) / (dim + 9 + cc)
        c1 = 2 / ((dim + 1.3) ** 2 + dim)
        cmu = min(1 - c1, 2 * (mu0 - 1) / ((dim + 2) ** 2 - 1))
        mu0 = int(pop / 2)
        w = np.log(mu0 + 0.5) - np.log(np.arange(1, mu0 + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c1 = 2 / ((dim + 1.3) ** 2 + dim)
        cmu = min(1 - c1, 2 * (mu_eff - 1) / ((dim + 2) ** 2 - 1))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        step_size = sigma

        best_x = x.copy()
        best_f = f(best_x)
        calls_used = calls[0]

        # CMA-ES loop
        while calls[0] < budget - 80:  # reserve budget for polish
            if time.time() - t0 > DEADLINE - 30:
                break

            # Sample population
            offsprings = []
            for i in range(pop):
                z = rng.standard_normal(dim)
                # Transform using Cholesky of C (approximate via eigen if needed)
                # Use simple: y = x + sigma * (B * C) @ z
                # For dim=5, compute eigendecomposition
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    B = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
                    y = x + step_size * (B @ z)
                except np.linalg.LinAlgError:
                    y = x + step_size * z
                y = np.clip(y, -5, 5)
                offsprings.append(y)

            # Evaluate
            fitness = np.array([f(y) for y in offsprings])

            # Sort
            idx = np.argsort(fitness)
            selected = [offsprings[i] for i in idx[:mu0]]
            selected_f = fitness[idx[:mu0]]

            # Update mean
            x_new = np.sum([w[i] * selected[i] for i in range(mu0)], axis=0)

            # Update evolution paths
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                B_inv = eigvecs @ np.diag(np.sqrt(eigvals))
                diff = (x_new - x) / step_size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B_inv @ diff)
                hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls[0] - calls_used) + 1))) / chi
                pc = (1 - cc) * pc + hs * cc * (x_new - x) / step_size
            except Exception:
                pc = (1 - cc) * pc + cc * (x_new - x) / step_size

            # Update covariance
            delta_h = (1.7 + 2 / (dim + 1)) * (hs > 1 / (1 + cc)) - 1
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                B_inv = eigvecs @ np.diag(np.sqrt(eigvals))
                diff = (x_new - x) / step_size
                rank1 = np.outer(B_inv @ diff, B_inv @ diff)
                C = (1 + c1 + cmu - c1) * C + c1 * (np.outer(pc, pc) + (delta_h - 1) * C) + cmu * np.sum(
                    [w[i] * np.outer(B_inv @ (selected[i] - x) / step_size, B_inv @ (selected[i] - x) / step_size) for i in range(mu0)], axis=0
                )
                # Symmetrize
                C = (C + C.T) / 2
            except Exception:
                pass

            x = x_new
            calls_used = calls[0]

            # Update step size
            step_size *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1))
            step_size = max(step_size, 1e-8)
            step_size = min(step_size, 10.0)

            # Track best
            if selected_f[0] < best_f:
                best_f = selected_f[0]
                best_x = selected[0].copy()

        # --- Phase 2: Step-boundary coordinate bisection polish ---
        # f7 has a step: f(x) = f_opt + ||x - x_opt||_1 (roughly, with a flat region)
        # Once close, we can bisect along each axis to find the boundary.
        remaining = budget - calls[0]
        if remaining > 10:
            # For each coordinate, do a local search to find the step boundary
            # The step in f7 means: moving toward x_opt decreases f until a plateau, then flat.
            # We use a coordinate-wise refinement.
            for coord_iter in range(3):
                for i in range(dim):
                    if calls[0] >= budget - 5:
                        break
                    if time.time() - t0 > DEADLINE - 5:
                        break

                    # Current best point
                    x_cur = best_x.copy()
                    f_cur = best_f

                    # Try moving in + and - direction along axis i
                    # Use a small step to detect direction
                    step = max(step_size * 0.5, 0.01)
                    # Ensure we stay in bounds
                    lo = max(-5, x_cur[i] - step)
                    hi = min(5, x_cur[i] + step)

                    if lo >= hi:
                        continue

                    # Evaluate at lo and hi
                    x_lo = x_cur.copy()
                    x_lo[i] = lo
                    x_hi = x_cur.copy()
                    x_hi[i] = hi

                    f_lo = f(x_lo)
                    f_hi = f(x_hi)

                    # Determine which side is better
                    if f_lo < f_hi:
                        # Move toward lo
                        a, b = lo, x_cur[i]
                        fa, fb = f_lo, f_cur
                    elif f_hi < f_lo:
                        # Move toward hi
                        a, b = x_cur[i], hi
                        fa, fb = f_cur, f_hi
                    else:
                        continue

                    # Bisection to find the step boundary
                    # The function is flat on one side of the boundary and decreasing on the other.
                    # We bisect to find where the change occurs.
                    for _ in range(min(20, (budget - calls[0]) // dim // 3)):
                        if calls[0] >= budget - 5:
                            break
                        if time.time() - t0 > DEADLINE - 5:
                            break
                        mid = (a + b) / 2
                        x_mid = best_x.copy()
                        x_mid[i] = mid
                        f_mid = f(x_mid)

                        # Compare f_mid with fa and fb
                        if f_mid <= fa:
                            # The step is between mid and b (or mid is on the plateau)
                            b = mid
                            fb = f_mid
                        else:
                            a = mid
                            fa = f_mid

                    # After bisection, the boundary is near 'a' (the side where f starts increasing)
                    # Place the point at the boundary
                    x_new = best_x.copy()
                    x_new[i] = a
                    f_new = f(x_new)
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
