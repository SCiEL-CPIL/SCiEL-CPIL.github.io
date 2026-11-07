# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats.qmc
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() > deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: Sobol Global Exploration ---
        # Use a structured sampling method for better coverage than pure random
        n_global = min(128, budget // 4)
        
        try:
            sampler = scipy.stats.qmc.Sobol(d, scramble=True, seed=42)
            # Generate 2^k points, k such that 2^k >= n_global
            k = max(1, int(np.ceil(np.log2(n_global))))
            sobol_points = sampler.random(2**k)
            # Scale to [-5, 5]
            sobol_points = sobol_points * 10.0 - 5.0
        except Exception:
            # Fallback to random if Sobol fails
            sobol_points = rng.uniform(-5, 5, (n_global, d))

        for i in range(n_global):
            if check_time() or used_calls >= budget:
                break
            x = sobol_points[i]
            f = func(x)
            used_calls += 1
            try_update(x, f)

        # --- Phase 2: CMA-ES Refinement ---
        if best_x is not None and used_calls < budget and not check_time():
            self._cma_es(func, best_x, best_f, budget, used_calls, check_time, try_update, rng)

        # --- Phase 3: Nelder-Mead Polish ---
        # Use Nelder-Mead for high-precision local optimization
        if best_x is not None and used_calls < budget and not check_time():
            self._nelder_mead(func, best_x, best_f, budget, used_calls, check_time, try_update)

        return best_x, best_f

    def _cma_es(self, func, x0, f0, budget, used_calls, check_time, try_update, rng):
        d = self.dim
        x = x0.copy()
        sigma = 1.0  # Initial step size
        C = np.eye(d)
        C_sqrt = np.eye(d)
        pc = np.zeros(d)
        
        # CMA-ES parameters
        cc = (4 + d / 4) / (d + 4 + 2 * d / 4)
        cs = cc
        c1 = 1 / (d + 4)
        cmu = 1 / (d + 8)
        damps = 1 + 2 * max(0, np.sqrt((d - 1) / (d + 3)) - 1) + c1
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        lambda_ = max(4, int(4 + 3 * np.log(d)))
        mu = lambda_ // 2
        
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()

        stagnation_counter = 0
        max_stagnation = 10 # Restart if no improvement for 10 generations

        while used_calls < budget and not check_time():
            # Generate population
            pop = []
            for i in range(lambda_):
                if used_calls >= budget or check_time():
                    break
                z = rng.standard_normal(d)
                y = C_sqrt @ z
                xi = x + sigma * y
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                used_calls += 1
                pop.append((xi, fi))

            if len(pop) < lambda_:
                break

            pop.sort(key=lambda t: t[1])
            
            # Update best
            if pop[0][1] < f0:
                f0 = pop[0][1]
                try_update(pop[0][0], f0)
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Check for stagnation and restart
            if stagnation_counter > max_stagnation:
                # Restart with smaller sigma
                sigma *= 0.5
                x = best_x.copy() if best_x is not None else x
                C = np.eye(d)
                C_sqrt = np.eye(d)
                pc = np.zeros(d)
                stagnation_counter = 0
                max_stagnation = 5 # Shorter patience after restart

            # Compute mean of top mu
            x_new = np.zeros(d)
            for i in range(min(mu, len(pop))):
                x_new += weights[i] * pop[i][0]

            # Update step size
            y_mean = (x_new - x) / sigma if sigma > 0 else np.zeros(d)
            sigma *= np.exp(cs / damps * (np.linalg.norm(y_mean) / chiN - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update path
            pc = (1 - cs) * pc + cs * y_mean
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs)**2) / chiN < 1.4 + 2 / (d + 1))

            # Update covariance
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cs * (1 - cs) * C)
            for i in range(min(mu, len(pop))):
                yi = (pop[i][0] - x) / sigma
                C += cmu * weights[i] * np.outer(yi, yi)

            x = x_new

            # Recompute C_sqrt
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            except:
                C_sqrt = np.eye(d)

            if sigma < 1e-8:
                break

    def _nelder_mead(self, func, x0, f0, budget, used_calls, check_time, try_update):
        d = self.dim
        # Initialize simplex
        x0 = x0.copy()
        simplex = [x0]
        f_simplex = [f0]
        
        # Create initial simplex points
        for i in range(d):
            xi = x0.copy()
            step = 0.05 if abs(x0[i]) > 1e-8 else 0.05
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            used_calls += 1
            simplex.append(xi)
            f_simplex.append(fi)
            try_update(xi, fi)
            
            if used_calls >= budget or check_time():
                return

        # Nelder-Mead iterations
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        
        max_iter = 50
        for _ in range(max_iter):
            if used_calls >= budget or check_time():
                break
            
            # Sort simplex
            idx = np.argsort(f_simplex)
            simplex = [simplex[i] for i in idx]
            f_simplex = [f_simplex[i] for i in idx]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            used_calls += 1
            try_update(xr, fr)
            
            if f_simplex[0] <= fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
            elif fr < f_simplex[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                used_calls += 1
                try_update(xe, fe)
                
                if fe < fr:
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
            else:
                # Contraction
                if fr < f_simplex[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                    
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                used_calls += 1
                try_update(xc, fc)
                
                if fc < max(fr, f_simplex[-1]):
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                else:
                    # Shrink
                    for i in range(1, d + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fi = func(simplex[i])
                        used_calls += 1
                        f_simplex[i] = fi
                        try_update(simplex[i], fi)
                        
                        if used_calls >= budget or check_time():
                            return

# EVOLVE-BLOCK-END
