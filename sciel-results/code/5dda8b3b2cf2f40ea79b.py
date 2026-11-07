# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Derivative-Free Trust-Region (DFTR) with global random restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 295.0
        dim = self.dim
        rng = np.random.default_rng(42)
        calls = 0

        def feval(x):
            nonlocal calls
            calls += 1
            return func(x)

        # Phase 1: Multi-start coarse search to find initial basin
        # Use a smaller grid to save budget for refinement
        n_grid = 3
        grid = np.array(np.meshgrid(*([np.linspace(-5, 5, n_grid)] * dim)),
                        dtype=float).reshape(-1, dim)
        order = rng.permutation(grid.shape[0])
        best_x = np.zeros(dim)
        best_fx = np.inf
        
        # Evaluate a subset of the grid first
        n_eval_grid = min(grid.shape[0], 50)
        for idx in order[:n_eval_grid]:
            x = grid[idx]
            fx = feval(x)
            if fx < best_fx:
                best_x, best_fx = x.copy(), fx
            if calls >= self.budget or time.time() > deadline:
                break
        
        if not np.isfinite(best_fx):
            x0 = np.zeros(dim)
            best_fx = feval(x0)
            best_x = x0.copy()

        # Phase 2: DFTR refinement
        # Trust-region radius
        r = 1.0
        r_min = 1e-6
        r_max = 10.0
        stagnation_count = 0
        max_stagnation = 10
        
        x = best_x.copy()
        fx = best_fx
        
        while calls < self.budget and time.time() < deadline:
            # Build local quadratic model using sparse finite differences
            # For dim=5, we can use forward differences for gradient and Hessian
            h = max(r * 0.1, 1e-4)
            
            # Gradient estimation
            grad = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xp[i] += h
                xp = np.clip(xp, -5, 5)
                fp = feval(xp)
                xm = x.copy()
                xm[i] -= h
                xm = np.clip(xm, -5, 5)
                fm = feval(xm)
                grad[i] = (fp - fm) / (2 * h)
                if calls >= self.budget or time.time() > deadline:
                    break
            
            if calls >= self.budget or time.time() > deadline:
                break
            
            # Hessian estimation (sparse, diagonal + off-diagonal sampling)
            # For ill-conditioned problems, we need to capture cross-terms
            # Use a simple 2nd-order difference for diagonal, and cross-terms
            H = np.zeros((dim, dim))
            for i in range(dim):
                xp = x.copy()
                xp[i] += h
                xp = np.clip(xp, -5, 5)
                fp = feval(xp)
                xm = x.copy()
                xm[i] -= h
                xm = np.clip(xm, -5, 5)
                fm = feval(xm)
                H[i, i] = (fp - 2*fx + fm) / (h*h)
                if calls >= self.budget or time.time() > deadline:
                    break
                
                # Cross terms with next dimension
                if i + 1 < dim:
                    xpp = x.copy()
                    xpp[i] += h
                    xpp[i+1] += h
                    xpp = np.clip(xpp, -5, 5)
                    fpp = feval(xpp)
                    xpm = x.copy()
                    xpm[i] += h
                    xpm[i+1] -= h
                    xpm = np.clip(xpm, -5, 5)
                    fpm = feval(xpm)
                    xmp = x.copy()
                    xmp[i] -= h
                    xmp[i+1] += h
                    xmp = np.clip(xmp, -5, 5)
                    fmp = feval(xmp)
                    xmm = x.copy()
                    xmm[i] -= h
                    xmm[i+1] -= h
                    xmm = np.clip(xmm, -5, 5)
                    fmm = feval(xmm)
                    H[i, i+1] = (fpp - fpm - fmp + fmm) / (4*h*h)
                    H[i+1, i] = H[i, i+1]
                    if calls >= self.budget or time.time() > deadline:
                        break
            
            if calls >= self.budget or time.time() > deadline:
                break
            
            # Solve trust-region subproblem: min g^T s + 0.5 s^T H s s.t. ||s|| <= r
            # Use eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(H)
            # Regularize if Hessian is not positive definite
            eigvals = np.maximum(eigvals, 1e-8)
            
            # Newton step
            s_newton = np.zeros(dim)
            for i in range(dim):
                s_newton[i] = -eigvecs[:, i] * (eigvecs[:, i] @ grad) / eigvals[i]
            
            # Check if Newton step is within trust region
            norm_s = np.linalg.norm(s_newton)
            if norm_s <= r:
                s = s_newton
            else:
                # Cauchy point: s = -alpha * grad
                alpha = r / max(np.linalg.norm(grad), 1e-10)
                s = -alpha * grad
            
            # Clip step to stay in bounds
            x_trial = np.clip(x + s, -5, 5)
            fx_trial = feval(x_trial)
            
            # Accept or reject
            rho = (fx - fx_trial) / max(r * np.linalg.norm(grad), 1e-10)
            if rho > 0.25:
                # Successful step
                x, fx = x_trial, fx_trial
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                r = min(2 * r, r_max)
                stagnation_count = 0
            elif rho > -0.75:
                # Marginal
                r = 0.5 * r
                stagnation_count += 1
            else:
                # Unsuccessful
                r = 0.25 * r
                stagnation_count += 1
            
            if r < r_min:
                r = r_min
            
            # Stagnation restart
            if stagnation_count >= max_stagnation:
                # Random restart near best point
                x = best_x + rng.standard_normal(dim) * 0.5
                x = np.clip(x, -5, 5)
                fx = feval(x)
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                r = 1.0
                stagnation_count = 0
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
