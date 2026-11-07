"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with Nelder-Mead polish for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        call_count = 0
        max_calls = self.budget
        dim = self.dim

        best_x = np.zeros(dim)
        best_f = float('inf')

        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= max_calls:
                raise ValueError("Budget exceeded")
            call_count += 1
            x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
            f_val = float(func(x_clipped))
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # CMA-ES implementation
        def cma_es(x0, sigma0, budget_remaining, max_time):
            nonlocal call_count, best_f, best_x, deadline
            dim = self.dim
            
            # Population size
            lam = int(4 + 3 * np.log(dim))
            if lam < 4:
                lam = 4
            if lam > 20:
                lam = 20
            
            # Weights
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / ((dim + 1) ** 2 + mueff)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            # State
            x_mean = x0.copy().astype(float)
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Precompute
            B, D = np.linalg.eigh(C)
            D_inv = 1.0 / np.sqrt(D)
            B_inv = np.linalg.inv(B)
            
            it = 0
            while call_count < max_calls and (time.time() < deadline - 5):
                # Generate population
                samples = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    samples.append(x_mean + sigma * (B @ (D_inv * z)))
                
                # Evaluate
                vals = []
                for x in samples:
                    if call_count >= max_calls:
                        break
                    try:
                        vals.append(counted_func(x))
                    except Exception:
                        vals.append(np.inf)
                
                if len(vals) < lam:
                    break
                
                # Sort
                order = np.argsort(vals)
                x_sorted = [samples[i] for i in order]
                vals_sorted = [vals[i] for i in order]
                
                # Update mean
                x_new_mean = np.zeros(dim)
                for i in range(mu):
                    x_new_mean += weights[i] * x_sorted[i]
                
                # Update step size control
                if np.linalg.norm(B_inv @ (x_new_mean - x_mean)) < 1e-16:
                    pass
                else:
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi * (B_inv @ (x_new_mean - x_mean)) / sigma
                
                # Update covariance matrix adaptation
                pc = (1 - c_c) * pc + c_c * (x_new_mean - x_mean) / sigma
                delta_hs = np.sqrt((1 - (1 - c_c) ** (2 * (it + 1))) / (1 - (1 - c_c) ** 2)) * (chi + 1 - c_c)
                if np.linalg.norm(pc) / chi < (1.4 + 1 / c_c) * delta_hs:
                    C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1 - c_c) ** 2) * c_c * (1 - c_c) * C) + c_mu / (sigma ** 2) * sum(weights[i] * np.outer(x_sorted[i] - x_mean, x_sorted[i] - x_mean) for i in range(mu))
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi - 1))
                
                # Update mean
                x_mean = x_new_mean
                
                # Recompute B, D
                B, D = np.linalg.eigh(C)
                D_inv = 1.0 / np.sqrt(D)
                B_inv = np.linalg.inv(B)
                
                it += 1
                if it % 50 == 0:
                    break  # Prevent too long CMA-ES runs
            
            return x_mean, sigma

        # Nelder-Mead polish
        def nelder_mead_polish(x0, max_calls_remaining, max_time):
            nonlocal call_count, best_f, best_x, deadline
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            eps = 1e-8
            max_iter = 200
            
            # Initialize simplex
            simplex = [x0.copy().astype(float)]
            f0 = counted_func(simplex[0])
            for i in range(self.dim):
                xi = x0.copy()
                step = 0.05 if abs(x0[i]) > 1e-8 else 0.05
                xi[i] += step
                simplex.append(xi)
                f_i = counted_func(xi)
                simplex[-1] = np.append(simplex[-1], f_i)
            # Actually, let's store (x, f) pairs properly
            simplex = [(x0.copy(), f0)]
            for i in range(self.dim):
                xi = x0.copy()
                step = 0.05 if abs(x0[i]) > 1e-8 else 0.05
                xi[i] += step
                f_i = counted_func(xi)
                simplex.append((xi, f_i))
            
            for it in range(max_iter):
                if call_count >= max_calls or time.time() > deadline - 2:
                    break
                if len(simplex) < self.dim + 1:
                    break
                
                # Sort
                simplex.sort(key=lambda x: x[1])
                
                # Check convergence
                if len(simplex) >= 2:
                    f_range = abs(simplex[-1][1] - simplex[0][1])
                    if f_range < 1e-10:
                        break
                
                # Centroid of all but worst
                centroid = np.mean([s[0] for s in simplex[:-1]], axis=0)
                worst = simplex[-1]
                
                # Reflection
                xr = centroid + alpha * (centroid - worst[0])
                fr = counted_func(xr)
                
                if fr < simplex[-2][1]:
                    # Accept or expand
                    if fr < simplex[0][1]:
                        xe = centroid + gamma * (centroid - worst[0])
                        fe = counted_func(xe)
                        if fe < fr:
                            simplex[-1] = (xe, fe)
                        else:
                            simplex[-1] = (xr, fr)
                    else:
                        simplex[-1] = (xr, fr)
                else:
                    # Contraction
                    if fr < worst[1]:
                        xc = centroid + sigma * (centroid - worst[0])
                        fc = counted_func(xc)
                        if fc < fr:
                            simplex[-1] = (xc, fc)
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i][0] = simplex[0][0] + 0.5 * (simplex[i][0] - simplex[0][0])
                                simplex[i][1] = counted_func(simplex[i][0])
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i][0] = simplex[0][0] + 0.5 * (simplex[i][0] - simplex[0][0])
                            simplex[i][1] = counted_func(simplex[i][0])
            
            # Return best from simplex
            best_in_simplex = min(simplex, key=lambda x: x[1])
            return best_in_simplex[0]

        # Phase 1: CMA-ES with moderate budget
        cma_budget = int(self.budget * 0.6)
        x_cma, sigma_cma = cma_es(np.zeros(dim), 2.0, cma_budget, deadline)
        
        # Phase 2: Nelder-Mead polish from CMA-ES result
        if call_count < max_calls and time.time() < deadline - 5:
            x_nm = nelder_mead_polish(x_cma, max_calls, deadline)
            if x_nm is not None:
                counted_func(x_nm)  # Ensure it's evaluated
        
        # Phase 3: If we have budget left, try a few more random starts with NM
        remaining = max_calls - call_count
        if remaining > 20 and time.time() < deadline - 10:
            for _ in range(min(3, remaining // 20)):
                if call_count >= max_calls or time.time() > deadline - 5:
                    break
                x_rand = rng.uniform(-4, 4, dim)
                x_try = nelder_mead_polish(x_rand, max_calls, deadline)
                if x_try is not None:
                    counted_func(x_try)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
