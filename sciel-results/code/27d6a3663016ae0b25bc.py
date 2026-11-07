"""Model-and-optimize optimizer for BBOB f19 (Griewank-Rosenbrock).

Combines a Gaussian-process surrogate for global exploration with a full
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for local refinement.
The GP helps identify promising basins, while CMA-ES adapts to the local
geometry (rotation, scaling) to efficiently descend the valley.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import erf


def _N_pdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def _N_cdf(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


class Optimizer:
    """GP surrogate (model) + CMA-ES (optimize)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._f = None
        self._xs = []
        self._ys = []
        self._best_x = None
        self._best_f = float('inf')
        self._calls = 0
        self._deadline = 0.0

    def _evaluate(self, x):
        """Wrapper to count calls and track best."""
        x = np.clip(x, -5, 5)
        f = float(self._f(x))
        self._xs.append(x.copy())
        self._ys.append(f)
        self._calls += 1
        if f < self._best_f:
            self._best_f = f
            self._best_x = x.copy()
        return f

    # ---- GP Model -------------------------------------------------------
    def _gp_fit(self, X, y):
        n = X.shape[0]
        if n == 0:
            return 0.0, np.zeros(0), 1.0
        if n == 1:
            return float(y[0]), np.zeros(1), 1.0
        
        # Calculate pairwise squared distances
        d = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        d2 = (d * d).sum(axis=2)
        
        # Heuristic length scale: based on median distance
        nonzero = d2[d2 > 1e-10]
        if nonzero.size > 0:
            l = float(np.sqrt(2.0 * np.median(nonzero)))
            l = np.clip(l, 0.05, 10.0)
        else:
            l = 1.0
        l2 = l * l
        
        K = np.exp(-0.5 * d2 / l2)
        K[np.arange(n), np.arange(n)] += 1e-6
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(K + 1e-4 * np.eye(n))
        
        a = np.linalg.solve(L, np.ones(n))
        y0 = float(np.mean(y))
        return y0, a, l2

    def _gp_predict(self, X, Xq, y0, a, l2):
        n = X.shape[0]
        if n == 0:
            return np.zeros(Xq.shape[0]), np.ones(Xq.shape[0])
        
        dx = Xq[:, np.newaxis, :] - X[np.newaxis, :, :]
        d2 = (dx * dx).sum(axis=2)
        Kq = np.exp(-0.5 * d2 / l2)
        
        mu = y0 + Kq @ a
        v = Kq * a[np.newaxis, :]
        var = float(a @ a) - (v * v).sum(axis=1)
        return mu, np.maximum(var, 0.0)

    def _gp_acq(self, X, Xq, b, y0, a, l2, strategy='ei'):
        mu, var = self._gp_predict(X, Xq, y0, a, l2)
        sd = np.sqrt(np.maximum(var, 1e-12))
        
        if strategy == 'ei':
            z = (b - mu) / np.where(sd < 1e-9, 1e-9, sd)
            phi = _N_pdf(z)
            Phi = _N_cdf(z)
            ei = (b - mu) * (1.0 - Phi) + sd * phi
            return np.maximum(ei, 0.0)
        elif strategy == 'ucb':
            gamma = 1.5
            return mu - gamma * sd
        else:
            return mu

    def _optimize_acq(self, X, b, y0, a, l2, rng, nstarts=5):
        """Find the point that maximizes the acquisition function."""
        from scipy.optimize import minimize as _min
        
        def acq_neg(x):
            x = np.clip(x, -5, 5)
            val = self._gp_acq(X, x.reshape(1, -1), b, y0, a, l2, strategy='ei')[0]
            return -val
        
        best_val = -np.inf
        best_x = None
        
        # Generate starting points
        starts = []
        if self._best_x is not None:
            starts.append(self._best_x.copy())
        for _ in range(nstarts):
            starts.append(rng.uniform(-5, 5, self.dim))
            
        for x0 in starts:
            try:
                res = _min(acq_neg, x0, method='L-BFGS-B', bounds=[(-5, 5)] * self.dim, options={'maxiter': 20})
                xv = np.clip(res.x, -5, 5)
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = xv
            except Exception:
                continue
        
        if best_x is None:
            best_x = starts[0]
        return best_x

    # ---- CMA-ES ---------------------------------------------------------
    def _cmaes_step(self, state, rng):
        """
        Perform one iteration of CMA-ES.
        state: dict with keys 'c', 'sigma', 'C', 'Cinv_sqrt', 'p_sigma', 'p_c', 'history'
        Returns: list of (x, f) pairs generated, and updated state.
        """
        n = self.dim
        pop = state['pop']
        mu = pop // 2
        
        c = state['c']
        sigma = state['sigma']
        C = state['C']
        Cinv_sqrt = state['Cinv_sqrt']
        p_sigma = state['p_sigma']
        p_c = state['p_c']
        
        # Parameters
        mueff = state['mueff']
        cs = state['cs']
        ds = state['ds']
        c1 = state['c1']
        cmu = state['cmu']
        chiN = state['chiN']
        
        # Sample
        Y = rng.standard_normal((pop, n))
        X = np.array([c + sigma * (Cinv_sqrt @ y).T for y in Y])
        X = np.clip(X, -5, 5)
        
        # Evaluate
        F = np.array([self._evaluate(x) for x in X])
        
        # Sort
        order = np.argsort(F)
        Y = Y[order]
        X = X[order]
        F = F[order]
        
        # Weights
        w = state['w']
        
        # Mean update
        c_new = sum(w[i] * X[i] for i in range(mu))
        
        # Step size control
        y_sigma = (Cinv_sqrt @ (c_new - c)).T
        p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_sigma
        hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (self._calls + 1 - state['start_call'])) <
                (1 + 2 / (n + 1)) * chiN)
        
        # Covariance matrix control
        p_c = (1 - cs) * p_c + hsig * np.sqrt(cs * (2 - cs) * mueff) * (c_new - c)
        
        rank_mu = np.zeros((n, n))
        for i in range(mu):
            y_i = (Cinv_sqrt @ (X[i] - c)).T
            rank_mu += w[i] * np.outer(y_i, y_i)
        
        rank_one = np.outer(p_c, p_c)
        
        C_new = (1 + c1 * (hsig * (2 - hsig) - 1) + cmu) * C \
                - c1 * rank_one \
                - cmu * rank_mu
        
        # Symmetrize
        C_new = (C_new + C_new.T) / 2
        
        # Update sigma
        sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chiN - 1))
        
        # Update inverse square root
        # Simple update: recompute from Cholesky if needed, or use approximation
        # For efficiency, we recompute Cinv_sqrt every few steps or if unstable
        # Here we use a simple update rule or recompute
        try:
            L = np.linalg.cholesky(C_new + 1e-6 * np.eye(n))
            Cinv_sqrt_new = np.linalg.inv(L.T)
        except np.linalg.LinAlgError:
            # Fallback: keep old Cinv_sqrt
            Cinv_sqrt_new = Cinv_sqrt
        
        state['c'] = c_new
        state['sigma'] = sigma
        state['C'] = C_new
        state['Cinv_sqrt'] = Cinv_sqrt_new
        state['p_sigma'] = p_sigma
        state['p_c'] = p_c
        state['start_call'] = self._calls
        
        return X, F, state

    def _init_cmaes(self, x0, rng):
        n = self.dim
        pop = max(4, int(4 + 3 * np.log(n)))
        mu = pop // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = float(w.sum() ** 2)
        
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (n + mueff)) + cs
        c1 = (2 + (mueff - 2) / (n + 2)) / ((n + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        
        C = np.eye(n)
        Cinv_sqrt = np.eye(n)
        
        state = {
            'c': x0.copy(),
            'sigma': 2.0, # Start wide
            'C': C,
            'Cinv_sqrt': Cinv_sqrt,
            'p_sigma': np.zeros(n),
            'p_c': np.zeros(n),
            'pop': pop,
            'mu': mu,
            'w': w,
            'mueff': mueff,
            'cs': cs,
            'ds': ds,
            'c1': c1,
            'cmu': cmu,
            'chiN': chiN,
            'start_call': self._calls
        }
        return state

    def __call__(self, func):
        self._f = func
        t0 = time.time()
        self._deadline = t0 + 280.0
        rng = np.random.default_rng(0)
        
        # Phase 1: Initialization / GP Exploration
        # Sample a few random points to start
        init_points = 5
        for _ in range(init_points):
            if self._calls >= self.budget or time.time() >= self._deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            self._evaluate(x)
        
        # Initialize GP
        if len(self._xs) > 0:
            X = np.array(self._xs)
            y = np.array(self._ys)
            y0, a, l2 = self._gp_fit(X, y)
        else:
            y0, a, l2 = 0.0, np.zeros(0), 1.0
            
        # Initialize CMA-ES at the best point found
        cmaes_state = self._init_cmaes(self._best_x, rng)
        
        # Phase 2: Hybrid Loop
        last_improvement_call = self._calls
        stagnant_counter = 0
        
        while self._calls < self.budget and time.time() < self._deadline:
            # Check stagnation
            if self._calls - last_improvement_call > 20:
                stagnant_counter += 1
            else:
                stagnant_counter = 0
            
            # If stagnant, use GP to propose a new center for CMA-ES
            if stagnant_counter >= 3:
                # Use GP to find a new promising point
                if len(self._xs) >= 4:
                    X = np.array(self._xs)
                    y = np.array(self._ys)
                    y0, a, l2 = self._gp_fit(X, y)
                    b = self._best_f
                    x_new = self._optimize_acq(X, b, y0, a, l2, rng, nstarts=5)
                    f_new = self._evaluate(x_new)
                    
                    if f_new < self._best_f:
                        # Restart CMA-ES at the new best
                        cmaes_state = self._init_cmaes(self._best_x, rng)
                        stagnant_counter = 0
                        last_improvement_call = self._calls
                    else:
                        # If no improvement, just move CMA-ES center to GP suggestion
                        cmaes_state['c'] = x_new
                        cmaes_state['sigma'] = 1.0
                        stagnant_counter = 0
                else:
                    # Random restart
                    x_new = rng.uniform(-5, 5, self.dim)
                    self._evaluate(x_new)
                    cmaes_state['c'] = self._best_x
                    cmaes_state['sigma'] = 1.0
                    stagnant_counter = 0
            
            # Run one CMA-ES step
            X_cma, F_cma, cmaes_state = self._cmaes_step(cmaes_state, rng)
            
            # Update last improvement if any point in the population improved
            if np.min(F_cma) < self._best_f:
                last_improvement_call = self._calls
                stagnant_counter = 0
        
        return self._best_x
# EVOLVE-BLOCK-END
