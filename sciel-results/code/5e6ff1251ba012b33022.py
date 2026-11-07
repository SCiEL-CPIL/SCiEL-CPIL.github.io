"""Exploit quadratic structure of Discus via local Hessian fitting and Newton steps,
combined with CMA-ES for robust exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimizer that exploits the quadratic structure of BBOB f11 (Discus)
    by fitting a local Hessian and taking Newton steps."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 330s limit
        rng = np.random.default_rng(42)

        # Track best
        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls
            if calls >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Dense deterministic sweep to find the narrow axis of Discus
        # BBOB f11 (Discus): f(x) = 0.5 * (||x||^2 + 2 * max_i |x_i|^2)
        
        # First, evaluate origin
        evaluate(np.zeros(dim))

        # Axis-based sweep: for each axis, try positive and negative directions at several radii
        radii = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        for axis in range(dim):
            for r in radii:
                if calls >= budget or time.time() > deadline:
                    break
                # Positive direction
                x = np.zeros(dim)
                x[axis] = r
                evaluate(x)
                # Negative direction
                x[axis] = -r
                if calls < budget and time.time() < deadline:
                    evaluate(x)
            if calls >= budget or time.time() > deadline:
                break

        # Radial sweep in random directions
        n_radial = 20
        for i in range(n_radial):
            if calls >= budget or time.time() > deadline:
                break
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction)
            for r in [1.0, 2.0, 3.0, 4.0, 5.0]:
                if calls >= budget or time.time() > deadline:
                    break
                evaluate(direction * r)

        # Phase 2: Hessian-based Newton optimization
        if calls < budget and best_x is not None:
            self._newton_optimize(func, best_x.copy(), budget - calls, deadline, rng, evaluate)

        return best_x, best_f

    def _fit_hessian(self, func, x0, evaluate, budget_remaining):
        """Fit a local quadratic model (Hessian) around x0 using finite differences."""
        dim = self.dim
        n = dim
        
        # Check if we have enough budget for finite differences
        # Central differences need 2*n evaluations, forward differences need n
        if budget_remaining < 2 * n + 1:
            return None, None, None
        
        # Use central differences for better accuracy
        h = 0.01  # Step size for finite differences
        
        # Evaluate at center
        f0 = evaluate(x0)
        if np.isinf(f0):
            return None, None, None
        
        # Compute gradient and Hessian via finite differences
        grad = np.zeros(n)
        H = np.zeros((n, n))
        
        for i in range(n):
            # Forward difference for gradient
            x_plus = x0.copy()
            x_plus[i] += h
            f_plus = evaluate(x_plus)
            
            x_minus = x0.copy()
            x_minus[i] -= h
            f_minus = evaluate(x_minus)
            
            if np.isinf(f_plus) or np.isinf(f_minus):
                return None, None, None
            
            grad[i] = (f_plus - f_minus) / (2 * h)
            
            # Second derivative for diagonal of Hessian
            H[i, i] = (f_plus - 2 * f0 + f_minus) / (h ** 2)
        
        # Compute off-diagonal Hessian entries via cross partials
        for i in range(n):
            for j in range(i + 1, n):
                x_pp = x0.copy()
                x_pp[i] += h
                x_pp[j] += h
                f_pp = evaluate(x_pp)
                
                x_pm = x0.copy()
                x_pm[i] += h
                x_pm[j] -= h
                f_pm = evaluate(x_pm)
                
                x_mp = x0.copy()
                x_mp[i] -= h
                x_mp[j] += h
                f_mp = evaluate(x_mp)
                
                x_mm = x0.copy()
                x_mm[i] -= h
                x_mm[j] -= h
                f_mm = evaluate(x_mm)
                
                if np.isinf(f_pp) or np.isinf(f_pm) or np.isinf(f_mp) or np.isinf(f_mm):
                    return None, None, None
                
                H[i, j] = H[j, i] = (f_pp - f_pm - f_mp + f_mm) / (4 * h ** 2)
        
        return f0, grad, H

    def _newton_optimize(self, func, x_start, max_evals, deadline, rng, evaluate):
        """Run Newton-based optimization exploiting quadratic structure."""
        dim = self.dim
        n = dim
        x = x_start.copy()
        
        # Newton iterations
        newton_interval = 50  # Fit Hessian every 50 evaluations
        last_fit_call = 0
        
        for it in range(max_evals):
            if time.time() > deadline:
                break
            
            # Fit Hessian periodically
            if (it % newton_interval == 0) or it == 0:
                f0, grad, H = self._fit_hessian(func, x, evaluate, max_evals - it)
                
                if H is not None and grad is not None:
                    # Try Newton step: x_new = x - H^{-1} * grad
                    # Use damped Newton to avoid overshooting
                    try:
                        # Check if H is positive definite
                        eigvals = np.linalg.eigvalsh(H)
                        if np.all(eigvals > 1e-10):
                            # Positive definite, take full Newton step
                            H_inv = np.linalg.inv(H)
                            step = H_inv @ grad
                            
                            # Damp the step to avoid large jumps
                            step_norm = np.linalg.norm(step)
                            if step_norm > 5.0:
                                step *= 5.0 / step_norm
                            
                            x_new = x - step
                            x_new = np.clip(x_new, -5.0, 5.0)
                            
                            f_new = evaluate(x_new)
                            if not np.isinf(f_new) and f_new < np.inf:
                                x = x_new
                        else:
                            # H is not positive definite, use steepest descent
                            grad_norm = np.linalg.norm(grad)
                            if grad_norm > 1e-10:
                                step = grad / grad_norm
                                x_new = x - 0.5 * step
                                x_new = np.clip(x_new, -5.0, 5.0)
                                f_new = evaluate(x_new)
                                if not np.isinf(f_new) and f_new < np.inf:
                                    x = x_new
                    except np.linalg.LinAlgError:
                        # Singular matrix, use steepest descent
                        grad_norm = np.linalg.norm(grad)
                        if grad_norm > 1e-10:
                            step = grad / grad_norm
                            x_new = x - 0.5 * step
                            x_new = np.clip(x_new, -5.0, 5.0)
                            f_new = evaluate(x_new)
                            if not np.isinf(f_new) and f_new < np.inf:
                                x = x_new
                
                # If we can't fit Hessian, fall back to local search
                else:
                    # Simple local search with small steps
                    for _ in range(min(10, max_evals - it)):
                        if time.time() > deadline:
                            break
                        direction = rng.standard_normal(n)
                        direction /= np.linalg.norm(direction)
                        for step_size in [0.5, 0.1, 0.01]:
                            x_try = x + step_size * direction
                            x_try = np.clip(x_try, -5.0, 5.0)
                            f_try = evaluate(x_try)
                            if not np.isinf(f_try):
                                break
                        else:
                            continue
                        break
        
        # Final CMA-ES polish if we still have budget
        if calls_remaining := (max_evals - (it + 1)):
            if calls_remaining > 20 and time.time() < deadline:
                self._cma_es(func, x, calls_remaining, deadline, rng, evaluate)

    def _cma_es(self, func, x0, max_evals, deadline, rng, evaluate):
        """Run CMA-ES starting from x0."""
        dim = self.dim
        n = dim
        
        # CMA-ES parameters
        lambda_ = max(10, 4 + int(3 * np.log(n)))
        mu = lambda_ // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_c = min(c_c, 1.0)
        mueff_inv = 1.0 / mueff
        c_1 = (2 - mueff_inv + c_sigma) / ((n + 1.3) ** 2 + mueff_inv + c_sigma)
        c_mu = (2 * (mueff_inv - 1) + c_sigma) / ((n + 2) ** 2 + mueff_inv + c_sigma)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff_inv - 1) / (n + 1)) - 1) + c_sigma

        # Covariance matrix adaptation
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        x_mean = x0.copy()
        sigma = 1.0  # Initial sigma
        evals = 0

        # Initial population
        pop_x = []
        pop_f = []
        for i in range(lambda_):
            if time.time() > deadline:
                break
            z = rng.standard_normal(n)
            xi = x_mean + sigma * (B @ (D * z))
            fi = evaluate(xi)
            if np.isinf(fi):
                break
            pop_x.append(xi)
            pop_f.append(fi)
            evals += 1
            if evals >= max_evals:
                break

        if len(pop_x) < 2:
            return

        # Sort by fitness
        idx = np.argsort(pop_f)
        pop_x = [pop_x[i] for i in idx]
        pop_f = [pop_f[i] for i in idx]

        while evals < max_evals and time.time() < deadline:
            # Update translation (cm)
            w = weights[:min(mu, len(pop_f))]
            x_mean_new = np.sum(np.array(pop_x[:len(w)]) * w[:, None], axis=0)

            pc = (1 - c_c) * pc + c_c * np.sqrt(mueff) / (sigma * np.sqrt(np.sum(D ** 2))) * (x_mean_new - x_mean)
            hs = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (evals + 1))) / chi_n < 1.4 + 2 / (n + 1)
            C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - hs) * c_c * (2 - c_c) * C) + \
                c_mu * np.sum(w[:, None, None] * (np.array(pop_x[:len(w)]) - x_mean) @ (np.array(pop_x[:len(w)]) - x_mean).T, axis=0)

            # Eigendecomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-10))
            B = B * np.sqrt(D[:, None])

            # Update sigma (cs)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / (sigma * np.sqrt(np.sum(D ** 2))) * (x_mean_new - x_mean)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, 1e6)
            sigma = max(sigma, 1e-12)

            x_mean = x_mean_new

            # Generate new population
            pop_x_new = []
            pop_f_new = []
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                xi = x_mean + sigma * (B @ z)
                fi = evaluate(xi)
                if np.isinf(fi):
                    break
                pop_x_new.append(xi)
                pop_f_new.append(fi)
                evals += 1
                if evals >= max_evals:
                    break

            if len(pop_x_new) < 2:
                break

            idx_new = np.argsort(pop_f_new)
            pop_x = [pop_x_new[i] for i in idx_new]
            pop_f = [pop_f_new[i] for i in idx_new]
# EVOLVE-BLOCK-END
