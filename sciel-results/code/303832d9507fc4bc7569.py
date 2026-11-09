# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Derivative-free Gauss-Newton for affine transform fitting + targeted (1+1)-ES."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)
        self.max_calls = self.budget - 2
        self._deadline = None

    def __call__(self, func):
        start = time.time()
        self._deadline = start + 280.0
        d = self.dim
        rng = np.random.default_rng(42)

        self._calls = 0
        self._best_x = None
        self._best_f = np.inf

        def call(x):
            if self._calls >= self.max_calls:
                return None
            self._calls += 1
            return func(np.asarray(x, dtype=float))

        def time_ok():
            return time.time() < self._deadline

        # ---------------- Phase A: Diverse initialization with Sobol + random ----------------
        n_init = 40
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True, seed=42)
            sobol_pts = sampler.random(n_init // 2)
            random_pts = rng.random((n_init - n_init // 2, d))
            init_pts = np.vstack([sobol_pts, random_pts])
        except Exception:
            init_pts = rng.random((n_init, d))
        
        init_pts = init_pts * 10.0 - 5.0
        init_pts = np.clip(init_pts, -5.0, 5.0)

        candidates = []
        for pt in init_pts:
            if not time_ok() or self._calls >= self.max_calls:
                break
            fx = call(pt)
            if fx is None:
                break
            candidates.append((fx, pt.copy()))
            if fx < self._best_f:
                self._best_f, self._best_x = fx, pt.copy()

        candidates.sort(key=lambda x: x[0])
        top_k = min(6, len(candidates))

        # ---------------- Phase B: Affine transform estimation via directional probes ----------------
        # For each top candidate, probe along random directions to estimate local structure
        for i in range(top_k):
            if not time_ok() or self._calls >= self.max_calls:
                break
            x0 = candidates[i][1]
            self._estimate_affine_and_refine(func, call, x0, time_ok, rng)

        # ---------------- Phase C: Final targeted (1+1)-ES from best point ----------------
        if self._best_x is not None and self._calls < self.max_calls and time_ok():
            self._targeted_es(func, call, self._best_x.copy(), time_ok, rng)

        return self._best_x, self._best_f

    def _estimate_affine_and_refine(self, func, call, x0, time_ok, rng):
        """Estimate local affine transform structure and do targeted refinement."""
        d = self.dim
        x = x0.copy()
        
        if self._calls >= self.max_calls or not time_ok():
            return
        
        fx = call(x)
        if fx is None:
            return
        
        if fx < self._best_f:
            self._best_f, self._best_x = fx, x.copy()

        # Probe along random directions to estimate gradient-like structure
        n_probes = 8
        probe_dirs = []
        probe_vals = []
        
        for _ in range(n_probes):
            if self._calls >= self.max_calls or not time_ok():
                break
            direction = rng.standard_normal(d)
            direction /= np.linalg.norm(direction)
            
            # Probe at small step
            step_size = 0.1
            x_plus = x + step_size * direction
            x_plus = np.clip(x_plus, -5.0, 5.0)
            f_plus = call(x_plus)
            if f_plus is None:
                break
            
            x_minus = x - step_size * direction
            x_minus = np.clip(x_minus, -5.0, 5.0)
            f_minus = call(x_minus)
            if f_minus is None:
                break
            
            # Finite difference gradient estimate
            grad_est = (f_plus - f_minus) / (2 * step_size)
            probe_dirs.append(direction)
            probe_vals.append(grad_est)
        
        if len(probe_dirs) < 2:
            return
        
        # Fit a linear model: f(x + t*d) ≈ f(x) + t * (g · d) for random directions d
        # This estimates the gradient g
        D = np.array(probe_dirs)  # (n_probes, d)
        V = np.array(probe_vals)  # (n_probes,)
        
        # Solve D @ g = V in least squares sense
        try:
            g, _, _, _ = np.linalg.lstsq(D, V, rcond=None)
        except Exception:
            g = np.zeros(d)
        
        # Use estimated gradient for descent with adaptive step
        self._gradient_descent(func, call, x, fx, g, time_ok, rng)

    def _gradient_descent(self, func, call, x, fx, grad, time_ok, rng):
        """Adaptive gradient descent with restarts."""
        d = self.dim
        lr = 0.1
        max_iter = 50
        no_improve = 0
        
        for _ in range(max_iter):
            if not time_ok() or self._calls >= self.max_calls:
                break
            
            # Compute current gradient estimate
            if self._calls >= self.max_calls or not time_ok():
                break
            
            eps = 0.01
            grad_new = np.zeros(d)
            for i in range(d):
                xp = x.copy()
                xp[i] += eps
                xp = np.clip(xp, -5.0, 5.0)
                f_plus = call(xp)
                if f_plus is None:
                    break
                
                xm = x.copy()
                xm[i] -= eps
                xm = np.clip(xm, -5.0, 5.0)
                f_minus = call(xm)
                if f_minus is None:
                    break
                
                grad_new[i] = (f_plus - f_minus) / (2 * eps)
            else:
                # Full gradient computed
                step = -lr * grad_new
                x_new = x + step
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = call(x_new)
                if f_new is None:
                    break
                
                if f_new < fx - 1e-8:
                    x, fx = x_new, f_new
                    if fx < self._best_f:
                        self._best_f, self._best_x = fx, x.copy()
                    no_improve = 0
                    lr = min(lr * 1.2, 1.0)
                else:
                    no_improve += 1
                    lr *= 0.5
                    if no_improve > 5:
                        break
                grad = grad_new
            if self._calls >= self.max_calls:
                break

    def _targeted_es(self, func, call, x, time_ok, rng):
        """Targeted (1+1)-ES from the best point found."""
        d = self.dim
        sigma = 0.5
        max_iter = 80
        no_improve = 0
        
        for _ in range(max_iter):
            if not time_ok() or self._calls >= self.max_calls:
                break
            
            # Generate perturbation
            delta = rng.standard_normal(d) * sigma
            x_new = x + delta
            x_new = np.clip(x_new, -5.0, 5.0)
            
            f_new = call(x_new)
            if f_new is None:
                break
            
            if f_new < self._best_f - 1e-10:
                x = x_new
                self._best_f, self._best_x = f_new, x.copy()
                sigma = min(sigma * 1.05, 2.0)
                no_improve = 0
            else:
                no_improve += 1
                sigma *= 0.8
                if no_improve > 15:
                    break

# EVOLVE-BLOCK-END
