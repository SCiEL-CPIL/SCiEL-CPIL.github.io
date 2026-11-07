"""Adaptive grid search with CMA-ES refinement for BBOB f7 (Step ellipsoidal)."""
import time
import numpy as np


class Optimizer:
    """Multi-resolution adaptive grid search exploiting step structure, with CMA-ES fallback."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _adaptive_grid_search(self, func, budget, deadline, state):
        """
        Multi-resolution adaptive grid search exploiting step structure.
        Uses Sobol-like space-filling for initial scan, then adaptive refinement.
        """
        dim = self.dim
        best_x = None
        best_f = np.inf
        
        # Phase 1: Coarse global scan with stratified sampling
        n_coarse = min(50, max(20, budget // 10))
        if state["calls"] + n_coarse > budget:
            n_coarse = max(1, budget - state["calls"])
        
        # Generate stratified points in [-5, 5]^dim
        points_coarse = np.zeros((n_coarse, dim))
        rng = np.random.default_rng(42)
        
        for d in range(dim):
            # Stratified sampling along each dimension
            perm = rng.permutation(n_coarse)
            for i in range(n_coarse):
                points_coarse[i, d] = (perm[i] + rng.uniform(0, 1)) / n_coarse
        
        points_coarse = points_coarse * 10.0 - 5.0
        
        # Evaluate coarse points
        for i in range(n_coarse):
            if state["calls"] >= budget or time.time() > deadline:
                break
            x = points_coarse[i]
            f = func(x)
            state["calls"] += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if best_x is None:
            return np.zeros(dim), np.inf
        
        # Phase 2: Adaptive refinement around best point
        # Identify regions with sharp gradients (step boundaries)
        remaining = budget - state["calls"]
        if remaining <= 5:
            return best_x, best_f
        
        # Local adaptive search: try perturbations in each dimension
        # to detect step boundaries
        n_local = min(remaining // 2, 100)
        
        # Try small perturbations to find step boundary
        step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]
        
        for step in step_sizes:
            if state["calls"] >= budget or time.time() > deadline:
                break
            
            # Try perturbations in each dimension
            for d in range(dim):
                for direction in [-1, 1]:
                    if state["calls"] >= budget or time.time() > deadline:
                        break
                    
                    x_trial = best_x.copy()
                    x_trial[d] += direction * step
                    x_trial = np.clip(x_trial, -5, 5)
                    
                    f_trial = func(x_trial)
                    state["calls"] += 1
                    
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial.copy()
        
        # Phase 3: Fine-grained local search using adaptive (1+1)-ES
        # on the detected step boundary
        remaining = budget - state["calls"]
        if remaining > 10:
            best_x, best_f = self._local_es(func, best_x, best_f, remaining, deadline, state)
        
        return best_x, best_f

    def _local_es(self, func, x0, f0, budget, deadline, state):
        """Adaptive (1+1)-ES for local refinement."""
        dim = self.dim
        x_best = x0.copy()
        f_best = f0
        sigma = 0.1
        
        for _ in range(budget):
            if state["calls"] >= budget or time.time() > deadline:
                break
            
            # Generate trial point
            x_trial = x_best + sigma * np.random.standard_normal(dim)
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = func(x_trial)
            state["calls"] += 1
            
            if f_trial < f_best:
                f_best = f_trial
                x_best = x_trial.copy()
                sigma *= 1.1  # Increase step on success
            else:
                sigma *= 0.9  # Decrease step on failure
            
            sigma = np.clip(sigma, 1e-6, 1.0)
        
        return x_best, f_best

    def _cmaes(self, func, x0, sigma, budget, deadline, rng, state):
        """Run CMA-ES until budget or deadline. Returns (best_x, best_f)."""
        dim = self.dim
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (dim + 2)) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (2 - c_c)
        c_mu = min(1 - c_1, (2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)) * (c_c + c_1))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        C = np.eye(dim)
        Csqrt = np.eye(dim)
        Cinv_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        best_x = x0.copy()
        best_f = func(x0)
        state["calls"] += 1
        gen = 0
        no_improve = 0

        while state["calls"] < budget:
            if time.time() > deadline:
                break
            gen += 1
            z = rng.standard_normal((lam, dim))
            y = np.array([z[i] @ Cinv_sqrt for i in range(lam)])
            xk = np.array([x0 + sigma * y[i] for i in range(lam)])
            xk = np.clip(xk, -5.0, 5.0)
            fk = np.empty(lam)
            for i in range(lam):
                fk[i] = func(xk[i])
                state["calls"] += 1
                if state["calls"] >= budget or time.time() > deadline:
                    break
            if state["calls"] >= budget:
                for i in range(i + 1, lam):
                    fk[i] = np.inf

            order = np.argsort(fk)
            if fk[order[0]] < best_f:
                best_f = fk[order[0]]
                best_x = xk[order[0]].copy()
                no_improve = 0
            else:
                no_improve += 1

            xnew = np.zeros(dim)
            for i in range(mu):
                xnew += w[i] * xk[order[i]]

            yinv = np.array([Cinv_sqrt @ (xk[order[i]] - x0) / sigma for i in range(mu)])
            v = np.zeros((dim, dim))
            for i in range(mu):
                v += np.outer(w[i] * yinv[i], yinv[i])
            Csqrt = (1 - c_1 - c_mu) * Csqrt + c_mu * (Csqrt @ v + v @ Csqrt) + c_1 * (
                (1 - c_c) * Csqrt + c_c * (xnew - x0) / sigma
            )
            C = Csqrt @ Csqrt.T
            C += 1e-12 * np.eye(dim)
            try:
                L = np.linalg.cholesky(C)
                Cinv_sqrt = np.linalg.inv(L).T
            except np.linalg.LinAlgError:
                C = np.linalg.pinv(C) + 1e-8 * np.eye(dim)
                Cinv_sqrt = np.linalg.inv(np.linalg.cholesky(C + 1e-6 * np.eye(dim))).T

            pc = (1 - c_c) * pc + c_c * (xnew - x0) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_c * (2 - c_c) * mueff) * Cinv_sqrt @ (xnew - x0) / sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e3)

            x0 = xnew

            if no_improve > max(150, 20 * dim):
                break

        return best_x, best_f

    def _nelder_mead(self, func, x0, budget, deadline, state):
        """Nelder-Mead local refinement."""
        dim = self.dim
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [func(x0)]
        state["calls"] += 1
        
        for i in range(dim):
            x_new = x0.copy()
            x_new[i] += 0.5
            x_new = np.clip(x_new, -5, 5)
            simplex.append(x_new)
            fvals.append(func(x_new))
            state["calls"] += 1
            if state["calls"] >= budget or time.time() > deadline:
                break
        
        if state["calls"] >= budget or time.time() > deadline:
            idx = np.argmin(fvals)
            return simplex[idx], fvals[idx]
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        max_iter = 200
        
        for _ in range(max_iter):
            if state["calls"] >= budget or time.time() > deadline:
                break
            
            # Sort simplex
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            state["calls"] += 1
            if state["calls"] >= budget or time.time() > deadline:
                break
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                state["calls"] += 1
                if state["calls"] >= budget or time.time() > deadline:
                    break
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                state["calls"] += 1
                if state["calls"] >= budget or time.time() > deadline:
                    break
                if f_c < f_r:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        state["calls"] += 1
                        if state["calls"] >= budget or time.time() > deadline:
                            break
                    if state["calls"] >= budget or time.time() > deadline:
                        break
            
            # Check for convergence
            if max(fvals) - min(fvals) < 1e-10:
                break
        
        idx = np.argmin(fvals)
        return simplex[idx], fvals[idx]

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        state = {"calls": 0}
        deadline = time.time() + 290.0  # Safety margin

        # Phase 1: Adaptive grid search exploiting step structure
        best_x, best_f = self._adaptive_grid_search(func, budget, deadline, state)
        
        # Phase 2: If budget remains, use CMA-ES for smooth region exploitation
        remaining = budget - state["calls"]
        if remaining > 20 and best_f < np.inf:
            rng = np.random.default_rng(12345)
            # Use CMA-ES starting from best point found
            bx, bf = self._cmaes(func, best_x.copy(), 0.5, budget, deadline, rng, state)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        return best_x, best_f
