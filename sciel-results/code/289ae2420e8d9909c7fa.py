"""CMA-ES based optimizer for BBOB f19 (Griewank-Rosenbrock).

Uses multi-start CMA-ES with adaptive restarts. Diverse initial points
help navigate the shifted/rotated landscape. Smaller population in early
phases yields more generations and faster early progress (critical for AOCC).
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

class Optimizer:
    """CMA-ES optimizer with multi-start and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, x0, f0, rng, budget, deadline, sigma0=2.0, pop=None):
        n = self.dim
        if pop is None:
            pop = max(4, int(4 + 3 * np.log(n)))
        mu = max(2, pop // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = float(w.sum() ** 2)
        
        # Step size control
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (n + mueff)) + cs
        
        # Covariance matrix adaptation
        c1 = (2 + (mueff - 2) / (n + 2)) / ((n + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        
        # State variables
        c = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(n)
        Csqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = c.copy()
        best_f = f0
        
        it = 0
        stagnation_count = 0
        last_best_f = f0
        
        while it < budget and time.time() < deadline:
            Y = np.zeros((pop, n))
            F = np.empty(pop)
            
            # Generate population
            for i in range(pop):
                z = rng.standard_normal(n)
                y = Csqrt @ z
                x = c + sigma * y
                x = np.clip(x, -5.0, 5.0)
                F[i] = float(self._f(x))
                Y[i] = y
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x.copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            
            # Sort and update
            idx = np.argsort(F)
            Ys = Y[idx]
            Fs = F[idx]
            
            ymean = (w[:, None] * Ys[:mu]).sum(axis=0)
            d = Csqrt @ ymean
            
            # Update mean
            c = c + cs * d
            c = np.clip(c, -5.0, 5.0)
            
            # Update step size control
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * ymean
            
            # Check if sigma is too small (stagnation)
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (it + 1)) < \
                    (1.4 + 2 / (n + 1)) * chiN:
                p_c = (1 - cs) * p_c + cs * d
                hsig = 1.0
            else:
                hsig = 0.0
            
            # Update covariance matrix
            delta_sig = (1 - (1 - cs) ** 2) / (1 - (1 - cs) ** ds)
            C = ((1 - c1 - cmu + c1 * hsig - cmu * hsig) * C
                 + c1 * np.outer(d, d)
                 + cmu * (w[:, None] * (Ys[:mu] @ Csqrt)).sum(axis=0)[:, None]
                 * (Ys[:mu] @ Csqrt))
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chiN - 1))
            sigma = float(np.clip(sigma, 1e-4, 10.0))
            
            # Refresh square root of C
            try:
                evals, evecs = np.linalg.eigh(C)
                evals = np.clip(evals, 1e-12, None)
                Csqrt = (evecs * np.sqrt(evals)) @ evecs.T
            except np.linalg.LinAlgError:
                C = np.eye(n)
                Csqrt = np.eye(n)
                sigma = sigma0 * 0.5
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                c = best_x.copy()
                
            it += 1
            
            # Early stopping if no progress for a long time
            if it > 10 and best_f == last_best_f and it % 10 == 0:
                stagnation_count += 10
            else:
                last_best_f = best_f
                
            if stagnation_count > 3 * pop:
                break
                
        return best_x, best_f

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        rng = np.random.default_rng(42)
        self._f = func
        n = self.dim
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Multi-start: evaluate several diverse initial points
        # This helps with shifted/rotated landscapes
        init_points = [
            np.zeros(n),
            np.ones(n) * 2.0,
            np.ones(n) * (-2.0),
            np.array([2.0, -2.0, 2.0, -2.0, 2.0][:n]),
            np.array([3.0, 0.0, -3.0, 1.0, -1.0][:n]),
            np.array([0.0, 3.0, 0.0, -3.0, 0.0][:n]),
            np.array([-3.0, 1.0, 3.0, -1.0, 3.0][:n]),
            np.array([1.0, 1.0, 1.0, 1.0, 1.0][:n]),
            np.array([-1.0, -1.0, -1.0, -1.0, -1.0][:n]),
            np.array([4.0, -4.0, 0.0, 2.0, -2.0][:n]),
        ]
        
        # Evaluate diverse starting points (up to 10 evaluations)
        for xp in init_points:
            if time.time() > deadline - 20:
                break
            xp = np.clip(xp, -5.0, 5.0)
            fp = float(func(xp))
            calls += 1
            if fp < best_f:
                best_f = fp
                best_x = xp.copy()
        
        if best_x is None:
            best_x = np.zeros(n)
            best_f = float(func(best_x))
            calls += 1
        
        # Phase 1: Global exploration with moderate sigma and smaller pop
        # Use pop=6 for faster generations (more iterations per budget)
        budget_phase1 = min(calls + 200, self.budget - 50)
        remaining1 = max(0, budget_phase1 - calls)
        
        if remaining1 > 0 and time.time() < deadline - 30:
            x1, f1 = self._cmaes(best_x, best_f, rng, remaining1, deadline, sigma0=2.0, pop=6)
            if f1 < best_f:
                best_f = f1
                best_x = x1.copy()
            calls += remaining1
        
        # Phase 2: Polish with smaller sigma
        if time.time() < deadline - 20 and calls < self.budget:
            remaining2 = self.budget - calls
            x2, f2 = self._cmaes(best_x, best_f, rng, remaining2, deadline, sigma0=0.5, pop=6)
            if f2 < best_f:
                best_f = f2
                best_x = x2.copy()
            calls += remaining2
        
        return best_x, best_f

# EVOLVE-BLOCK-END
