import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f13 (Sharp Ridge) in 5D with budget 500.
    Strategy:
    1. Coarse random search to locate the ridge basin.
    2. BOBYQA-style trust-region with quadratic model fitting to capture anisotropic curvature.
    3. Multiple restarts from perturbed best points if budget remains.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        best_x = None
        best_f = float('inf')
        calls = 0
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                x = np.asarray(x, dtype=np.float64)
                x = np.clip(x, -5.0, 5.0)
                f_val = float(func(x))
                if np.isfinite(f_val) and f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Coarse random search
        n_coarse = 50
        for _ in range(n_coarse):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0

        # Phase 2: BOBYQA-style trust region optimization
        self._bobyqa(best_x.copy(), best_f, safe_call, deadline)

        # Phase 3: Additional restarts from perturbed best points
        if calls < budget - 20 and time.time() < deadline:
            for restart in range(8):
                if calls >= budget - 10 or time.time() > deadline:
                    break

                noise_scale = 0.5 / (restart + 1)
                perturbation = rng.normal(0, noise_scale, size=dim)
                x_restart = np.clip(best_x + perturbation, -5.0, 5.0)

                f_restart = safe_call(x_restart)
                if f_restart is None:
                    break

                if f_restart < best_f:
                    best_f = f_restart
                    best_x = x_restart.copy()

                # Run trust region from this point
                self._bobyqa(x_restart.copy(), f_restart, safe_call, deadline)

        return best_x, best_f

    def _bobyqa(self, x_start, f_start, safe_call, deadline):
        """
        BOBYQA-style trust region method with quadratic model fitting.
        Maintains a set of interpolation points and fits a quadratic model.
        """
        dim = self.dim
        budget = self.budget
        max_iter = 80

        # Trust region radius
        rho = 1.0
        rho_min = 1e-6
        rho_max = 5.0

        # Initialize interpolation points
        points = [x_start.copy()]
        fvals = [f_start]

        # Add initial points along coordinate axes
        for i in range(dim):
            if time.time() > deadline or self._calls_count() >= budget:
                return
            xi = x_start.copy()
            xi[i] += rho
            fi = safe_call(xi)
            if fi is None:
                return
            points.append(xi.copy())
            fvals.append(fi)

            if time.time() > deadline or self._calls_count() >= budget:
                return
            xi = x_start.copy()
            xi[i] -= rho
            fi = safe_call(xi)
            if fi is None:
                return
            points.append(xi.copy())
            fvals.append(fi)

        # BOBYQA iteration
        for iteration in range(max_iter):
            if time.time() > deadline or self._calls_count() >= budget:
                break

            # Fit quadratic model
            model = self._fit_quadratic_model(points, fvals)
            if model is None:
                break

            # Find minimizer of model within trust region
            x_new, f_model = self._minimize_model(model, x_start, rho)
            if x_new is None:
                break

            # Evaluate new point
            f_new = safe_call(x_new)
            if f_new is None:
                break

            # Compute actual reduction vs predicted reduction
            f_best_current = min(fvals)
            predicted_reduction = f_best_current - f_model
            actual_reduction = f_best_current - f_new

            # Update trust region
            if actual_reduction > 0.75 * predicted_reduction:
                # Success: expand trust region
                rho = min(rho * 2.0, rho_max)
            elif actual_reduction > 0.25 * predicted_reduction:
                # Marginal success: keep trust region
                pass
            else:
                # Failure: shrink trust region
                rho = max(rho * 0.5, rho_min)

            # Update interpolation points
            # Remove worst point if we have enough
            if len(points) > dim + 1:
                # Remove point farthest from best
                best_idx = np.argmin(fvals)
                best_point = points[best_idx]
                distances = [np.linalg.norm(p - best_point) for p in points]
                worst_idx = np.argmax(distances)
                points.pop(worst_idx)
                fvals.pop(worst_idx)

            # Add new point
            points.append(x_new.copy())
            fvals.append(f_new)

            # Check convergence
            if rho < rho_min:
                break

            # Check if we've made good progress
            if actual_reduction < 1e-10:
                if rho < 0.1:
                    break

        # Final local refinement with Nelder-Mead
        if time.time() < deadline and self._calls_count() < budget - 10:
            self._nelder_mead_refine(x_start, f_start, safe_call, deadline)

    def _calls_count(self):
        """Get current call count from safe_call closure."""
        # We need to track this differently - use a shared counter
        if not hasattr(self, '_call_counter'):
            self._call_counter = 0
        return self._call_counter

    def _fit_quadratic_model(self, points, fvals):
        """
        Fit a quadratic model to the interpolation points.
        Model: f(x) = c + g^T(x-x0) + 0.5*(x-x0)^T H (x-x0)
        We use x_start as the reference point.
        """
        dim = self.dim
        n = len(points)

        if n < dim + 1:
            return None

        # Use the best point as reference
        best_idx = np.argmin(fvals)
        x0 = points[best_idx]
        f0 = fvals[best_idx]

        # Build design matrix for quadratic model
        # Parameters: c (1), g (dim), H (dim*dim) -> total 1 + dim + dim*dim
        n_params = 1 + dim + dim * dim

        if n < n_params:
            # Not enough points for full quadratic; use linear model
            # Linear model: f(x) = c + g^T(x-x0)
            A = np.zeros((n, dim + 1))
            for i, p in enumerate(points):
                A[i, 0] = 1.0
                A[i, 1:] = p - x0
            b = np.array(fvals) - f0
            try:
                coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                c = coeff[0]
                g = coeff[1:]
                H = np.zeros((dim, dim))
                return (c, g, H, x0)
            except Exception:
                return None

        # Full quadratic model
        A = np.zeros((n, n_params))
        for i, p in enumerate(points):
            dx = p - x0
            A[i, 0] = 1.0
            A[i, 1:1+dim] = dx
            # Quadratic terms: H_ij * dx_i * dx_j
            for j in range(dim):
                for k in range(j, dim):
                    idx = 1 + dim + j * dim + k
                    A[i, idx] = dx[j] * dx[k]
                    if j != k:
                        A[i, 1 + dim + k * dim + j] = dx[j] * dx[k]

        b = np.array(fvals) - f0
        try:
            coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            c = coeff[0]
            g = coeff[1:1+dim]
            H_flat = coeff[1+dim:]
            H = np.zeros((dim, dim))
            for j in range(dim):
                for k in range(j, dim):
                    val = H_flat[j * dim + k]
                    H[j, k] = val
                    if j != k:
                        H[k, j] = val
            return (c, g, H, x0)
        except Exception:
            return None

    def _minimize_model(self, model, x_ref, rho):
        """
        Minimize the quadratic model within the trust region ||x - x_ref|| <= rho.
        Uses a simple gradient descent on the model with projection.
        """
        if model is None:
            return None, None

        c, g, H, x0 = model
        dim = self.dim

        # Start from x0
        x = x0.copy()
        f_model = c

        # Gradient descent on the model
        lr = 0.1
        for _ in range(100):
            # Gradient of model
            grad = g + H @ x
            if np.linalg.norm(grad) < 1e-12:
                break

            # Step
            x_new = x - lr * grad

            # Project to trust region
            dist = np.linalg.norm(x_new - x_ref)
            if dist > rho:
                x_new = x_ref + rho * (x_new - x_ref) / dist

            # Evaluate model at new point
            dx = x_new - x0
            f_new = c + g @ dx + 0.5 * dx @ H @ dx

            if f_new < f_model:
                x = x_new
                f_model = f_new
            else:
                lr *= 0.5
                if lr < 1e-8:
                    break

        return x, f_model

    def _nelder_mead_refine(self, x_start, f_start, safe_call, deadline):
        """
        Nelder-Mead simplex method for local refinement.
        """
        dim = self.dim
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        step_size = 0.1

        simplex = [x_start.copy()]
        f_values = [f_start]

        for i in range(dim):
            xi = x_start.copy()
            xi[i] += step_size
            fi = safe_call(xi)
            if fi is None:
                return
            simplex.append(xi)
            f_values.append(fi)

        if len(simplex) < dim + 1:
            return

        simplex = np.array(simplex)
        f_values = np.array(f_values)

        no_improve_count = 0
        patience = 15

        for iteration in range(60):
            if time.time() > deadline:
                break

            idx = np.argsort(f_values)
            simplex = simplex[idx]
            f_values = f_values[idx]

            if f_values[0] < self._best_f_global:
                self._best_f_global = f_values[0]
                self._best_x_global = simplex[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            spread_x = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            spread_f = np.max(f_values) - np.min(f_values)

            if np.max(spread_x) < 1e-6 or spread_f < 1e-10:
                break

            if no_improve_count > patience:
                break

            centroid = np.mean(simplex[:-1], axis=0)

            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = safe_call(xr)
            if fr is None:
                break

            if fr < f_values[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = safe_call(xe)
                if fe is None:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
            elif fr < f_values[-2]:
                simplex[-1] = xr
                f_values[-1] = fr
            else:
                if fr < f_values[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5.0, 5.0)
                fc = safe_call(xc)
                if fc is None:
                    break
                if fc <= min(fr, f_values[-1]):
                    simplex[-1] = xc
                    f_values[-1] = fc
                else:
                    for i in range(1, len(simplex)):
                        if time.time() > deadline:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fi = safe_call(simplex[i])
                        if fi is None:
                            break
                        f_values[i] = fi
                    if fi is None:
                        break


# Need to fix the call counting - let's use a cleaner approach
class Optimizer:
    """
    Optimizer for BBOB f13 (Sharp Ridge) in 5D with budget 500.
    Strategy:
    1. Coarse random search to locate the ridge basin.
    2. BOBYQA-style trust region with quadratic model fitting.
    3. Nelder-Mead local refinement.
    4. Multiple restarts from perturbed best points.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        self._calls = 0
        self._best_f = float('inf')
        self._best_x = None
        self._func = func
        self._deadline = deadline
        self._budget = budget
        self._dim = dim

        rng = np.random.default_rng(42)

        def safe_call(x):
            if self._calls >= self._budget:
                return None
            if time.time() > self._deadline:
                return None
            self._calls += 1
            try:
                x = np.asarray(x, dtype=np.float64)
                x = np.clip(x, -5.0, 5.0)
                f_val = float(func(x))
                if np.isfinite(f_val) and f_val < self._best_f:
                    self._best_f = f_val
                    self._best_x = x.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Coarse random search
        n_coarse = 50
        for _ in range(n_coarse):
            if self._calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if self._best_x is None:
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return self._best_x, self._best_f
            self._best_x = x0
            self._best_f = f0

        # Phase 2: BOBYQA-style trust region optimization
        self._bobyqa(self._best_x.copy(), self._best_f, safe_call)

        # Phase 3: Nelder-Mead refinement
        if self._calls < budget - 20 and time.time() < deadline:
            self._nelder_mead(self._best_x.copy(), self._best_f, safe_call)

        # Phase 4: Additional restarts
        if self._calls < budget - 30 and time.time() < deadline:
            for restart in range(6):
                if self._calls >= budget - 15 or time.time() > deadline:
                    break

                noise_scale = 0.4 / (restart + 1)
                perturbation = rng.normal(0, noise_scale, size=dim)
                x_restart = np.clip(self._best_x + perturbation, -5.0, 5.0)

                f_restart = safe_call(x_restart)
                if f_restart is None:
                    break

                if f_restart < self._best_f:
                    self._best_f = f_restart
                    self._best_x = x_restart.copy()

                # Trust region from this point
                self._bobyqa(x_restart.copy(), f_restart, safe_call)

                # NM refinement
                if self._calls < budget - 15 and time.time() < deadline:
                    self._nelder_mead(self._best_x.copy(), self._best_f, safe_call)

        return self._best_x, self._best_f

    def _bobyqa(self, x_start, f_start, safe_call):
        """
        BOBYQA-style trust region with quadratic model fitting.
        """
        dim = self.dim
        max_iter = 50

        rho = 1.0
        rho_min = 1e-5
        rho_max = 5.0

        points = [x_start.copy()]
        fvals = [f_start]

        # Add initial points
        for i in range(dim):
            if time.time() > self._deadline or self._calls >= self._budget:
                return
            xi = x_start.copy()
            xi[i] += rho
            fi = safe_call(xi)
            if fi is None:
                return
            points.append(xi.copy())
            fvals.append(fi)

            if time.time() > self._deadline or self._calls >= self._budget:
                return
            xi = x_start.copy()
            xi[i] -= rho
            fi = safe_call(xi)
            if fi is None:
                return
            points.append(xi.copy())
            fvals.append(fi)

        for iteration in range(max_iter):
            if time.time() > self._deadline or self._calls >= self._budget:
                break

            model = self._fit_quadratic(points, fvals)
            if model is None:
                break

            x_new, f_model = self._minimize_model(model, x_start, rho)
            if x_new is None:
                break

            f_new = safe_call(x_new)
            if f_new is None:
                break

            f_best_current = min(fvals)
            predicted = f_best_current - f_model
            actual = f_best_current - f_new

            if actual > 0.75 * predicted:
                rho = min(rho * 2.0, rho_max)
            elif actual > 0.25 * predicted:
                pass
            else:
                rho = max(rho * 0.5, rho_min)

            if len(points) > dim + 2:
                best_idx = np.argmin(fvals)
                best_point = points[best_idx]
                dists = [np.linalg.norm(p - best_point) for p in points]
                worst_idx = np.argmax(dists)
                points.pop(worst_idx)
                fvals.pop(worst_idx)

            points.append(x_new.copy())
            fvals.append(f_new)

            if rho < rho_min:
                break
            if actual < 1e-12 and rho < 0.05:
                break

    def _fit_quadratic(self, points, fvals):
        dim = self.dim
        n = len(points)

        if n < 2:
            return None

        best_idx = np.argmin(fvals)
        x0 = points[best_idx]
        f0 = fvals[best_idx]

        n_params = 1 + dim + dim * dim

        if n < n_params:
            # Linear model
            A = np.zeros((n, dim + 1))
            for i, p in enumerate(points):
                A[i, 0] = 1.0
                A[i, 1:] = p - x0
            b = np.array(fvals) - f0
            try:
                coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                c = coeff[0]
                g = coeff[1:]
                H = np.zeros((dim, dim))
                return (c, g, H, x0)
            except Exception:
                return None

        A = np.zeros((n, n_params))
        for i, p in enumerate(points):
            dx = p - x0
            A[i, 0] = 1.0
            A[i, 1:1+dim] = dx
            for j in range(dim):
                for k in range(j, dim):
                    idx = 1 + dim + j * dim + k
                    A[i, idx] = dx[j] * dx[k]
                    if j != k:
                        A[i, 1 + dim + k * dim + j] = dx[j] * dx[k]

        b = np.array(fvals) - f0
        try:
            coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            c = coeff[0]
            g = coeff[1:1+dim]
            H_flat = coeff[1+dim:]
            H = np.zeros((dim, dim))
            for j in range(dim):
                for k in range(j, dim):
                    val = H_flat[j * dim + k]
                    H[j, k] = val
                    if j != k:
                        H[k, j] = val
            return (c, g, H, x0)
        except Exception:
            return None

    def _minimize_model(self, model, x_ref, rho):
        if model is None:
            return None, None

        c, g, H, x0 = model
        dim = self.dim

        x = x0.copy()
        f_model = c

        lr = 0.05
        for _ in range(80):
            grad = g + H @ (x - x0)
            if np.linalg.norm(grad) < 1e-12:
                break

            x_new = x - lr * grad

            dist = np.linalg.norm(x_new - x_ref)
            if dist > rho:
                x_new = x_ref + rho * (x_new - x_ref) / dist

            dx = x_new - x0
            f_new = c + g @ dx + 0.5 * dx @ H @ dx

            if f_new < f_model:
                x = x_new
                f_model = f_new
            else:
                lr *= 0.5
                if lr < 1e-8:
                    break

        return x, f_model

    def _nelder_mead(self, x_start, f_start, safe_call):
        dim = self.dim
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        step_size = 0.05

        simplex = [x_start.copy()]
        f_values = [f_start]

        for i in range(dim):
            xi = x_start.copy()
            xi[i] += step_size
            fi = safe_call(xi)
            if fi is None:
                return
            simplex.append(xi)
            f_values.append(fi)

        if len(simplex) < dim + 1:
            return

        simplex = np.array(simplex)
        f_values = np.array(f_values)

        no_improve = 0
        patience = 12

        for _ in range(50):
            if time.time() > self._deadline or self._calls >= self._budget:
                break

            idx = np.argsort(f_values)
            simplex = simplex[idx]
            f_values = f_values[idx]

            if f_values[0] < self._best_f:
                self._best_f = f_values[0]
                self._best_x = simplex[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            spread_f = np.max(f_values) - np.min(f_values)
            if spread_f < 1e-10:
                break
            if no_improve > patience:
                break

            centroid = np.mean(simplex[:-1], axis=0)

            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = safe_call(xr)
            if fr is None:
                break

            if fr < f_values[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = safe_call(xe)
                if fe is None:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
            elif fr < f_values[-2]:
                simplex[-1] = xr
                f_values[-1] = fr
            else:
                if fr < f_values[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5.0, 5.0)
                fc = safe_call(xc)
                if fc is None:
                    break
                if fc <= min(fr, f_values[-1]):
                    simplex[-1] = xc
                    f_values[-1] = fc
                else:
                    for i in range(1, len(simplex)):
                        if time.time() > self._deadline or self._calls >= self._budget:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fi = safe_call(simplex[i])
                        if fi is None:
                            break
                        f_values[i] = fi
                    if fi is None:
                        break
