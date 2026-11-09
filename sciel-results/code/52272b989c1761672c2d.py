"""Two-phase optimizer: coarse random search + CMA-ES refinement with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Two-phase optimizer: random search for basin finding, then CMA-ES for refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best solution
        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse random search to find basin
        # Use stratified random sampling for better coverage
        n_phase1 = min(80, budget // 6)  # ~13% of budget for exploration
        for i in range(n_phase1):
            if time.time() > deadline:
                break
            # Stratified sampling: divide box into sub-cubes
            # Use random points with some structure
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        # Also try some structured points (corners, center, axis-aligned)
        if time.time() < deadline and call_count < budget:
            # Center
            x = np.zeros(dim)
            evaluate(x)

        # Axis-aligned probes
        for d in range(dim):
            if time.time() > deadline or call_count >= budget:
                break
            for val in [-4.0, -2.0, 2.0, 4.0]:
                if time.time() > deadline or call_count >= budget:
                    break
                x = np.zeros(dim)
                x[d] = val
                evaluate(x)

        # Phase 2: CMA-ES refinement starting from best point found
        remaining = budget - call_count
        if remaining > 10 and best_x is not None:
            self._cma_es(func, best_x, best_f, remaining, rng, deadline)

        return best_x, best_f

    def _cma_es(self, func, x0, f0, budget, rng, deadline):
        """CMA-ES optimizer with time guard."""
        nonlocal best_x, best_f, call_count

        dim = self.dim
        # Population size
        lam = max(4, int(4 + 3 * math.log(dim)))
        
        # Initial sigma
        sigma = 0.5
        x = x0.copy()
        fx = f0

        # CMA-ES parameters
        alpha_mu = 0.5  # for path exponent
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        nu = 1 / (2 * mu)
        
        # Covariance matrix
        C = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Update rates
        c_sigma = (mu + 2) / (dim + 4)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (weights[0] - 0.5) / ((dim + 2) ** 2 + mu))
        
        # Damping for step size control
        chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Track best
        best_x_local = x.copy()
        best_f_local = fx
        call_count_local = 0

        def evaluate_local(x_pt):
            nonlocal best_x_local, best_f_local, call_count_local, best_x, best_f, call_count
            fx_pt = func(x_pt)
            call_count_local += 1
            call_count += 1
            if fx_pt < best_f_local:
                best_f_local = fx_pt
                best_x_local = x_pt.copy()
            if fx_pt < best_f:
                best_f = fx_pt
                best_x = x_pt.copy()
            return fx_pt

        # Main CMA-ES loop
        while call_count_local < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.multivariate_normal(np.zeros(dim), C)
                y = np.clip(y, -5, 5)
                ys.append(y)
            
            if not ys:
                break
            
            # Evaluate
            fs = np.array([evaluate_local(y) for y in ys])
            
            # Sort
            order = np.argsort(fs)
            ys = np.array([ys[i] for i in order])
            fs = fs[order]
            
            # Update best
            if fs[0] < best_f_local:
                best_f_local = fs[0]
                best_x_local = ys[0].copy()
                if best_f_local < best_f:
                    best_f = best_f_local
                    best_x = best_x_local.copy()
            
            # Compute weighted mean
            y_mean = np.zeros(dim)
            for i in range(mu):
                y_mean += weights[i] * ys[i]
            
            # Update evolution paths
            old_sigma = sigma
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C @ (y_mean - x)) / sigma
            # h_sigma check
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1 if (norm_p_sigma / math.sqrt(1 - (1 - c_sigma) ** (2 * (call_count_local + 1) / lam)) / chi_n < (1.4 + 2 / (dim + 1)) * chi_n) else 0
            
            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * (y_mean - x) / sigma
            
            # Update x
            x = y_mean
            
            # Update covariance
            delta_d = (h_sigma - 1) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_d - c_1 - c_mu) * C
            for i in range(mu):
                z = (ys[i] - y_mean) / sigma
                C += c_mu * weights[i] * np.outer(z, z)
            
            # Update sigma
            sigma *= math.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)
            
            # mu_eff
            mu_eff = 1 / np.sum(weights ** 2)

        # Final polish: local Nelder-Mead on best point
        if time.time() < deadline - 5:
            self._nelder_mead(func, best_x_local, remaining_calls=budget - call_count_local, rng=rng, deadline=deadline)

    def _nelder_mead(self, func, x0, remaining_calls, rng, deadline):
        """Simple Nelder-Mead polish."""
        nonlocal best_x, best_f, call_count
        
        dim = self.dim
        if remaining_calls < 10:
            return
        
        # Initialize simplex
        alpha = 0.5
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += alpha if x0[i] + alpha <= 5 else -alpha
            simplex.append(xi)
        
        fs = np.array([func(x) for x in simplex])
        for _ in range(len(fs)):
            call_count += 1
            if fs[_] < best_f:
                best_f = fs[_]
                best_x = simplex[_].copy()
        
        # Nelder-Mead iterations
        gamma = 2.0
        rho = -1.0
        sigma = 0.5
        
        max_iters = min(remaining_calls - dim - 1, 200)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if call_count >= self.budget:
                break
            
            # Sort simplex
            order = np.argsort(fs)
            simplex = [simplex[i] for i in order]
            fs = fs[order]
            
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = simplex[0].copy()
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + gamma * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fxr = func(xr)
            call_count += 1
            
            if fxr < fs[0]:
                # Expansion
                xe = centroid + 2.0 * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fxe = func(xe)
                call_count += 1
                if fxe < fxr:
                    simplex[-1] = xe
                    fs[-1] = fxe
                else:
                    simplex[-1] = xr
                    fs[-1] = fxr
            elif fxr < fs[-2]:
                simplex[-1] = xr
                fs[-1] = fxr
            else:
                # Contraction
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fxc = func(xc)
                call_count += 1
                if fxc < fs[-1]:
                    simplex[-1] = xc
                    fs[-1] = fxc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fs[i] = func(simplex[i])
                        call_count += 1
                        if fs[i] < best_f:
                            best_f = fs[i]
                            best_x = simplex[i].copy()
                        if call_count >= self.budget or time.time() > deadline:
                            break
                    if call_count >= self.budget or time.time() > deadline:
                        break
# EVOLVE-BLOCK-END
