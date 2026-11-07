"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with multi-start restarts and trust-region polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin within 330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_counter = [0]

        def check_time():
            return time.time() < deadline

        def check_budget():
            return calls_counter[0] < budget

        def eval_point(x):
            """Evaluate a point, updating best if improved."""
            nonlocal best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls_counter[0] += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nm_simplex(x0, f0, max_calls):
            """Run Nelder-Mead simplex starting from x0 with given budget."""
            nonlocal best_x, best_f
            if not check_time() or not check_budget():
                return

            n = dim
            # Initialize simplex: n+1 points
            simplex = np.zeros((n + 1, n))
            values = np.zeros(n + 1)

            simplex[0] = x0.copy()
            values[0] = f0

            # Initial step size
            step = 0.5
            for i in range(1, n + 1):
                if not check_time() or not check_budget():
                    break
                simplex[i] = simplex[0].copy()
                simplex[i][i - 1] += step
                simplex[i] = np.clip(simplex[i], -5, 5)
                values[i] = eval_point(simplex[i])

            # Sort simplex
            order = np.argsort(values)
            simplex = simplex[order]
            values = values[order]

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            # Adaptive parameters
            stagnation_count = 0
            max_stagnation = 30

            while check_time() and check_budget():
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = eval_point(x_ref)

                if f_ref < values[0]:
                    # Try expansion
                    if check_time() and check_budget():
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5, 5)
                        f_exp = eval_point(x_exp)
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            values[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            values[-1] = f_ref
                    else:
                        simplex[-1] = x_ref
                        values[-1] = f_ref
                    stagnation_count = 0
                elif f_ref < values[-2]:
                    simplex[-1] = x_ref
                    values[-1] = f_ref
                    stagnation_count = 0
                else:
                    # Contraction
                    if f_ref < values[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (centroid - simplex[-1])

                    x_con = np.clip(x_con, -5, 5)
                    f_con = eval_point(x_con)

                    if f_con < max(f_ref, values[-1]):
                        simplex[-1] = x_con
                        values[-1] = f_con
                        stagnation_count = 0
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if not check_time() or not check_budget():
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            values[i] = eval_point(simplex[i])
                        stagnation_count += 1

                # Re-sort
                order = np.argsort(values)
                simplex = simplex[order]
                values = values[order]

                # Check convergence: if simplex is very small, stop
                spread = np.max(values) - np.min(values)
                if spread < 1e-10 * (1 + abs(np.min(values))):
                    break

                # Stagnation check
                if stagnation_count >= max_stagnation:
                    break

        def trust_region_polish(x_start, max_calls):
            """Trust-region method with finite-difference gradient and Hessian."""
            nonlocal best_x, best_f
            if not check_time() or not check_budget():
                return

            x = x_start.copy()
            f = eval_point(x)

            # Trust region radius
            delta = 1.0
            min_delta = 1e-6
            max_delta = 5.0
            eta = 0.25  # Acceptance threshold for actual/predicted ratio

            # Finite difference step
            fd_step = 1e-4

            def compute_gradient(xc):
                """Compute gradient via central finite differences."""
                grad = np.zeros(dim)
                for i in range(dim):
                    xp = xc.copy()
                    xm = xc.copy()
                    xp[i] += fd_step
                    xm[i] -= fd_step
                    xp = np.clip(xp, -5, 5)
                    xm = np.clip(xm, -5, 5)
                    fp = eval_point(xp)
                    fm = eval_point(xm)
                    grad[i] = (fp - fm) / (2 * fd_step)
                return grad

            def compute_hessian(xc):
                """Compute Hessian via central finite differences of gradient."""
                hess = np.zeros((dim, dim))
                h_step = 1e-3
                for i in range(dim):
                    for j in range(i, dim):
                        xpp = xc.copy()
                        xpm = xc.copy()
                        xmp = xc.copy()
                        xmm = xc.copy()
                        xpp[i] += h_step
                        xpp[j] += h_step
                        xpm[i] += h_step
                        xpm[j] -= h_step
                        xmp[i] -= h_step
                        xmp[j] += h_step
                        xmm[i] -= h_step
                        xmm[j] -= h_step
                        xpp = np.clip(xpp, -5, 5)
                        xpm = np.clip(xpm, -5, 5)
                        xmp = np.clip(xmp, -5, 5)
                        xmm = np.clip(xmm, -5, 5)
                        fpp = eval_point(xpp)
                        fpm = eval_point(xpm)
                        fmp = eval_point(xmp)
                        fmm = eval_point(xmm)
                        hess[i, j] = (fpp - fpm - fmp + fmm) / (4 * h_step * h_step)
                        hess[j, i] = hess[i, j]
                return hess

            for iteration in range(50):
                if not check_time() or not check_budget():
                    break

                # Check if we have enough budget for gradient + Hessian
                # Gradient: 2*dim calls, Hessian: dim*(dim+1)/2 * 4 calls (but we can be smarter)
                # For dim=5: gradient=10, Hessian=40, total=50 per iteration
                # Let's just do gradient-based trust region (steepest descent direction)
                # to save budget

                grad = compute_gradient(x)
                if not check_time() or not check_budget():
                    break

                # If gradient is very small, we're at a minimum
                if np.linalg.norm(grad) < 1e-8:
                    break

                # Steepest descent direction
                d = -grad / (np.linalg.norm(grad) + 1e-15)

                # Find step size within trust region
                # Try several step sizes
                best_alpha = 0
                best_f_trial = f
                for alpha in [delta, delta * 0.5, delta * 0.25, delta * 0.125,
                              delta * 0.0625, delta * 0.03125]:
                    if not check_time() or not check_budget():
                        break
                    x_trial = x + alpha * d
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = eval_point(x_trial)
                    if f_trial < best_f_trial:
                        best_f_trial = f_trial
                        best_alpha = alpha

                if best_alpha > 0:
                    # Actual reduction
                    actual = f - best_f_trial
                    # Predicted reduction (linear model)
                    predicted = -np.dot(grad, best_alpha * d)
                    ratio = actual / (predicted + 1e-15)

                    if ratio > eta:
                        # Accept step, increase trust region
                        x = x + best_alpha * d
                        f = best_f_trial
                        delta = min(max_delta, delta * 2.0)
                    elif ratio > 0.1:
                        # Accept step, keep trust region
                        x = x + best_alpha * d
                        f = best_f_trial
                    else:
                        # Reject step, decrease trust region
                        delta = max(min_delta, delta * 0.5)
                        if delta <= min_delta:
                            break
                else:
                    # No improvement found, decrease trust region
                    delta = max(min_delta, delta * 0.5)
                    if delta <= min_delta:
                        break

            # Final coordinate descent polish
            if check_time() and check_budget():
                step_sizes = np.full(dim, 0.01)
                for _ in range(50):
                    if not check_time() or not check_budget():
                        break
                    improved = False
                    for i in range(dim):
                        if not check_time() or not check_budget():
                            break
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[i] += direction * step_sizes[i]
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = eval_point(x_trial)
                            if f_trial < f:
                                x = x_trial
                                f = f_trial
                                improved = True
                                step_sizes[i] *= 1.3
                                break
                        if not improved:
                            step_sizes[i] *= 0.5
                    if not improved:
                        break

            return x, f

        def final_polish(x_start, max_calls):
            """High-precision local search using trust-region + NM."""
            nonlocal best_x, best_f
            if not check_time() or not check_budget():
                return

            remaining = budget - calls_counter[0]
            if remaining < 10:
                return

            # Use trust-region for the main polish
            x, f = trust_region_polish(x_start, remaining)

            # If we still have budget, do a quick NM
            if check_time() and check_budget():
                remaining2 = budget - calls_counter[0]
                if remaining2 >= 10:
                    nm_simplex(x, f, remaining2)

        # Phase 1: Initial exploration with diverse random starts
        n_initial = 20
        candidates = []
        for i in range(n_initial):
            if not check_time() or not check_budget():
                break
            # Use different strategies for diversity
            if i == 0:
                x = np.zeros(dim)  # Center
            elif i == 1:
                x = rng.uniform(-4, 4, dim)  # Random in box
            elif i == 2:
                x = rng.normal(0, 1.5, dim)  # Gaussian
                x = np.clip(x, -5, 5)
            elif i == 3:
                # Corner-ish points
                x = rng.choice([-3.5, 3.5], dim)
            elif i == 4:
                # Another corner
                x = rng.choice([-4.0, 4.0], dim)
            elif i == 5:
                # Slightly off-center
                x = rng.normal(0, 2.5, dim)
                x = np.clip(x, -5, 5)
            else:
                x = rng.uniform(-4, 4, dim)

            f = eval_point(x)
            candidates.append((f, x.copy()))

        # Sort candidates
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Run Nelder-Mead from top candidates
        n_restarts = min(5, len(candidates))

        for i in range(n_restarts):
            if not check_time() or not check_budget():
                break

            x_start = candidates[i][1]
            f_start = candidates[i][0]

            # Run NM from this start
            nm_simplex(x_start, f_start, budget - calls_counter[0])

        # Phase 3: Final high-precision polish
        if check_time() and check_budget() and best_x is not None:
            final_polish(best_x, budget - calls_counter[0])

        return best_x, best_f
# EVOLVE-BLOCK-END
