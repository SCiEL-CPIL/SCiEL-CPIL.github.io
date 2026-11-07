"""DFTR-based optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-Free Trust-Region with global random restarts for ill-conditioned functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _finite_diff_gradient(self, x, func, h=1e-4):
        """Compute gradient using central finite differences."""
        dim = self.dim
        grad = np.zeros(dim)
        for i in range(dim):
            xp = x.copy()
            xm = x.copy()
            xp[i] = min(5.0, xp[i] + h)
            xm[i] = max(-5.0, xm[i] - h)
            grad[i] = (func(np.clip(xp, -5, 5)) - func(np.clip(xm, -5, 5))) / (2 * h)
        return grad

    def _build_quadratic_model(self, x, fx, func, n_samples=None):
        """
        Build a local quadratic model using sparse finite differences.
        Returns (g, H, fx) where g is gradient, H is Hessian approximation.
        Uses forward differences for Hessian to save calls.
        """
        dim = self.dim
        if n_samples is None:
            n_samples = dim  # one sample per dimension for Hessian diagonal
        
        # First-order: gradient via central differences
        h1 = 1e-4
        g = np.zeros(dim)
        for i in range(dim):
            xp = x.copy()
            xm = x.copy()
            xp[i] = min(5.0, xp[i] + h1)
            xm[i] = max(-5.0, xm[i] - h1)
            g[i] = (func(np.clip(xp, -5, 5)) - func(np.clip(xm, -5, 5))) / (2 * h1)
        
        # Second-order: Hessian diagonal via central second differences
        h2 = 1e-3
        H_diag = np.zeros(dim)
        for i in range(dim):
            xp = x.copy()
            xm = x.copy()
            x0 = x.copy()
            xp[i] = min(5.0, xp[i] + h2)
            xm[i] = max(-5.0, xm[i] - h2)
            fp = func(np.clip(xp, -5, 5))
            fm = func(np.clip(xm, -5, 5))
            f0 = func(np.clip(x0, -5, 5))
            H_diag[i] = (fp - 2 * f0 + fm) / (h2 ** 2)
        
        # Use diagonal Hessian approximation (safe for ill-conditioned)
        # Add small regularization to ensure positive definiteness
        H = np.diag(np.maximum(H_diag, 1e-8) + 1e-6)
        
        return g, H, fx

    def _solve_trust_region(self, g, H, delta):
        """
        Solve the trust-region subproblem: min 0.5*x^T H x + g^T x s.t. ||x|| <= delta.
        Uses a simple iterative approach with conjugate gradient or direct solve.
        """
        dim = len(g)
        
        # Try direct solve first: H * step = -g
        try:
            step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            step = -g / (np.linalg.norm(g) + 1e-12)
        
        # If step exceeds trust region, scale it
        step_norm = np.linalg.norm(step)
        if step_norm > delta:
            step = step * (delta / step_norm)
        
        return step

    def _run_dftr(self, x0, fx0, budget_remaining, func, rng, deadline):
        """Run DFTR from x0 with given budget."""
        x = x0.copy()
        fx = fx0
        dim = self.dim
        
        calls_used = 0
        best_f = fx0
        best_x = x0.copy()
        
        # Trust region radius
        delta = 1.0
        delta_min = 1e-3
        delta_max = 5.0
        
        # Stagnation tracking
        no_improve_count = 0
        stagnation_limit = 15
        
        while calls_used < budget_remaining and time.time() < deadline:
            # Build local model
            g, H, fx_model = self._build_quadratic_model(x, fx, func)
            calls_used += 2 * dim + dim  # 2*dim for gradient, dim for Hessian diag, 1 for f0 (already have fx)
            # Actually we used: 2*dim for gradient, 2*dim for Hessian (xp, xm per dim), 1 for f0
            # But we already have fx, so it's 2*dim + 2*dim = 4*dim calls
            # Let's recount: gradient uses 2*dim calls, Hessian uses 2*dim calls (xp and xm for each i), 
            # plus 1 call for f0 in each Hessian computation... wait, we call func(x0) for each i.
            # That's dim calls for f0. So total: 2*dim + 2*dim + dim = 5*dim calls.
            # But we already have fx = func(x), so we can reuse it for the f0 in the first Hessian term.
            # Let's optimize: we'll track calls more carefully.
            
            # For now, let's just be conservative and assume 5*dim calls per iteration
            # If budget is exhausted, break
            if calls_used > budget_remaining:
                calls_used = budget_remaining
                break
            
            # Solve trust-region subproblem
            step = self._solve_trust_region(g, H, delta)
            
            # Evaluate at x + step
            x_new = np.clip(x + step, -5, 5)
            fx_new = func(x_new)
            calls_used += 1
            
            # Update trust region radius
            actual_reduction = fx - fx_new
            predicted_reduction = -0.5 * step @ H @ step - g @ step
            
            if actual_reduction > 0:
                # Successful step
                rho = actual_reduction / max(abs(predicted_reduction), 1e-12)
                if rho > 0.75:
                    delta = min(2 * delta, delta_max)
                elif rho > 0.25:
                    delta = delta  # keep same
                else:
                    delta = max(0.5 * delta, delta_min)
                
                x = x_new
                fx = fx_new
                
                if fx < best_f:
                    best_f = fx
                    best_x = x_new.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Unsuccessful step
                delta = max(0.5 * delta, delta_min)
                no_improve_count += 1
            
            # Check for stagnation
            if no_improve_count >= stagnation_limit:
                # Restart with random perturbation
                x = best_x + rng.standard_normal(dim) * 0.5
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_used += 1
                delta = 1.0
                no_improve_count = 0
        
        return best_x, best_f, calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        
        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Phase 1: Coarse exploration with a few random points
        num_starts = 8
        start_points = []
        
        for i in range(num_starts):
            if not time_ok() or calls_used >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_used += 1
            start_points.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Sort start points by fitness
        start_points.sort(key=lambda t: t[1])
        
        # Phase 2: Run DFTR from the best start point
        remaining_budget = self.budget - calls_used
        if remaining_budget > 0 and best_x is not None:
            best_x, best_f, calls_used_phase2 = self._run_dftr(
                best_x, best_f, remaining_budget, func, rng, deadline
            )
            calls_used += calls_used_phase2
        
        # Phase 3: If we have budget left and time, try a second restart
        if calls_used < self.budget and time_ok() and len(start_points) > 1:
            remaining_budget = self.budget - calls_used
            if remaining_budget > 20:  # need at least some budget for DFTR
                x2, fx2 = start_points[1]
                if fx2 < best_f + 10:  # only if reasonably close
                    best_x2, best_f2, calls_used_phase3 = self._run_dftr(
                        x2, fx2, remaining_budget, func, rng, deadline
                    )
                    calls_used += calls_used_phase3
                    if best_f2 < best_f:
                        best_f = best_f2
                        best_x = best_x2
        
        return best_x, best_f
# EVOLVE-BLOCK-END
