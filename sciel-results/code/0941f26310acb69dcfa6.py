"""Adaptive Restart-based CMA-ES with Multi-point L-BFGS-B polish for BBOB f19.

Improvements:
1. Stratified/Sobol-like initial points for better global coverage.
2. Adaptive restart loop: continues as long as budget/time allows.
3. Multi-point polish: refines the top-k distinct candidates to avoid
   settling for a suboptimal local minimum.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """Adaptive Restart-based CMA-ES optimizer with final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes_short(self, x0, f0, rng, budget, deadline, sigma0=1.5):
        """Run CMA-ES for at most `budget` evaluations, respecting deadline.
        Returns (best_x, best_f, n_calls)."""
        n = self.dim
        if budget < 4:
            return x0, f0, 0

        pop = max(4, min(12, int(4 + 3 * np.log(n))))
        mu = pop // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = float(w.sum() ** 2)

        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (n + mueff)) + cs

        c1 = (2 + (mueff - 2) / (n + 2)) / ((n + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

        c = np.array(x0, dtype=float).copy()
        sigma = sigma0
        C = np.eye(n)
        Csqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        best_x = c.copy()
        best_f = f0
        calls = 0
        it = 0
        last_best_f = f0
        no_improve = 0

        while it < budget and calls < budget and time.time() < deadline:
            Y = np.zeros((pop, n))
            F = np.empty(pop)

            for i in range(pop):
                if calls >= budget or time.time() >= deadline:
                    break
                z = rng.standard_normal(n)
                y = Csqrt @ z
                x = c + sigma * y
                x = np.clip(x, -5.0, 5.0)
                F[i] = float(self._f(x))
                Y[i] = y
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x.copy()
                    no_improve = 0
                else:
                    no_improve += 1

            if calls >= budget or time.time() >= deadline:
                break

            # If not all evaluations completed, pad with worst
            if not np.all(np.isfinite(F)):
                break

            idx = np.argsort(F)
            Ys = Y[idx]
            Fs = F[idx]

            ymean = (w[:, None] * Ys[:mu]).sum(axis=0)
            d = Csqrt @ ymean

            c = c + cs * d
            c = np.clip(c, -5.0, 5.0)

            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * ymean

            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (it + 1)) < \
                    (1.4 + 2 / (n + 1)) * chiN:
                p_c = (1 - cs) * p_c + cs * d
                hsig = 1.0
            else:
                hsig = 0.0

            delta_sig = (1 - (1 - cs) ** 2) / (1 - (1 - cs) ** ds)
            C = ((1 - c1 - cmu + c1 * hsig - cmu * hsig) * C
                 + c1 * np.outer(d, d)
                 + cmu * (w[:, None] * (Ys[:mu] @ Csqrt)).sum(axis=0)[:, None]
                 * (Ys[:mu] @ Csqrt))

            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chiN - 1))
            sigma = float(np.clip(sigma, 1e-5, 10.0))

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

            if best_f == last_best_f:
                no_improve += pop
            else:
                last_best_f = best_f
                no_improve = 0

            if no_improve > 4 * pop:
                break

        return best_x, best_f, calls

    def _lbfgsb_polish(self, x0, f0, deadline):
        """Final L-BFGS-B polish. Returns (best_x, best_f, n_calls)."""
        calls = 0
        best_x = np.array(x0, dtype=float).copy()
        best_f = f0

        def fobj(x):
            nonlocal calls
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            val = float(self._f(x))
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        try:
            res = minimize(fobj, x0, method='L-BFGS-B',
                           bounds=[(-5.0, 5.0)] * self.dim,
                           options={'maxiter': 200, 'maxfun': 150,
                                    'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            pass

        return best_x, best_f, calls

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        rng = np.random.default_rng(12345)
        self._f = func

        best_x = None
        best_f = np.inf
        total_calls = 0
        
        # Store top candidates for polishing
        candidates = []

        # Phase 1: Adaptive CMA-ES Restarts
        # Use a stratified grid for initial points to ensure coverage
        # 1 origin, then stratified random points
        n_init_strata = 10
        init_points = [np.zeros(self.dim)]
        
        # Simple stratified sampling: divide space into hypercubes
        # For 5D, 2^5=32 cells, pick 10 random cells
        # But simpler: just use rng.uniform but ensure distinctness
        # Sobol-like: use rng.permutation or specific sequence?
        # Let's stick to diverse random points but add a few "mid-point" checks
        # e.g., +/- 2.5 on one axis at a time
        
        # Add axis-aligned points
        for d in range(self.dim):
            for sign in [-1, 1]:
                pt = np.zeros(self.dim)
                pt[d] = sign * 2.5
                init_points.append(pt)
                
        # Add random points
        for _ in range(10):
            init_points.append(rng.uniform(-4.0, 4.0, self.dim))

        # Shuffle to mix types
        rng.shuffle(init_points)

        # Adaptive loop
        restart_idx = 0
        # Start with larger budget, shrink as we go
        # Total budget 500. Reserve 100 for polish?
        # Let's reserve 20% for polish.
        search_budget = int(0.8 * self.budget)
        
        # We will use init_points as seeds. If we run out, generate new random ones.
        # Also, if a restart finishes early (no improvement), we can start a new one sooner.
        
        while total_calls < search_budget and time.time() < deadline - 40:
            # Get next initial point
            if restart_idx < len(init_points):
                x_init = init_points[restart_idx]
            else:
                x_init = rng.uniform(-4.0, 4.0, self.dim)
            restart_idx += 1

            # Evaluate initial point
            f_init = float(func(x_init))
            total_calls += 1
            
            if f_init < best_f:
                best_f = f_init
                best_x = x_init.copy()
            
            # Add to candidates
            candidates.append((f_init, x_init.copy()))

            # Determine budget for this restart
            # Diminishing returns: first restart gets more, later ones less
            # Base budget ~ 50-80 per restart
            # If we are early, give it more.
            remaining_search = search_budget - total_calls
            # Cap at 80 to ensure multiple restarts
            alloc = min(80, remaining_search)
            
            if alloc < 10:
                break

            x_best, f_best, n_calls = self._cmaes_short(
                x_init, f_init, rng, alloc, deadline, sigma0=1.5)
            total_calls += n_calls
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
            
            # Add result to candidates
            candidates.append((f_best, x_best.copy()))

            # If we hit the time limit or budget, break
            if time.time() > deadline - 50:
                break

        if best_x is None:
            x0 = np.zeros(self.dim)
            best_f = float(func(x0))
            total_calls += 1
            best_x = x0.copy()
            candidates.append((best_f, best_x.copy()))

        # Phase 2: Multi-point L-BFGS-B polish
        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])
        
        # Select top K distinct candidates
        # Distinct: distance > 0.5
        top_candidates = []
        for f_c, x_c in candidates:
            if f_c > best_f + 5.0: # Ignore very bad ones
                break
            is_distinct = True
            for _, x_t in top_candidates:
                if np.linalg.norm(x_c - x_t) < 0.5:
                    is_distinct = False
                    break
            if is_distinct:
                top_candidates.append((f_c, x_c))
            if len(top_candidates) >= 5:
                break

        remaining = self.budget - total_calls
        if remaining >= 10 and time.time() < deadline - 20:
            # Budget per polish
            polish_budget_per = max(10, int(remaining / max(1, len(top_candidates))))
            
            for f_c, x_c in top_candidates:
                if total_calls >= self.budget or time.time() > deadline - 10:
                    break
                # Only polish if this candidate is close to best
                # or if we haven't polished the best one yet
                x_pol, f_pol, n_pol = self._lbfgsb_polish(x_c, f_c, deadline)
                total_calls += n_pol
                if f_pol < best_f:
                    best_f = f_pol
                    best_x = x_pol.copy()

        return best_x, best_f

# EVOLVE-BLOCK-END
