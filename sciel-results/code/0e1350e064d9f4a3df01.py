"""Rewritten optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid optimizer combining:
    - Halton quasi-random scan for initial basin coverage
    - CMA-ES for adaptive basin exploration
    - Nelder-Mead + coordinate descent for local refinement
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin for 330s limit

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            x_clipped = np.clip(x, -self.bound, self.bound)
            calls += 1
            val = func(x_clipped)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def halton_sequence(n, base):
            """Generate n Halton sequence points in [0,1)."""
            result = np.zeros(n)
            f = 1.0
            i = 0
            for idx in range(1, n + 1):
                i = idx
                while i > 0:
                    i, remainder = divmod(i, base)
                    f /= base
                    result[idx - 1] += f * remainder
            return result

        # Phase 1: Halton quasi-random scan for diverse initial coverage
        n_scan = min(40, budget // 12)
        halton_points = []
        for d in range(dim):
            halton_points.append(halton_sequence(n_scan, d + 2))
        
        for i in range(n_scan):
            if calls >= budget or time.time() > deadline:
                break
            x = np.array([halton_points[d][i] for d in range(dim)])
            x = x * 10.0 - 5.0  # Map [0,1] to [-5,5]
            eval_and_track(x)

        # Also evaluate origin and a few structured points
        structured_points = [
            np.zeros(dim),
            np.array([3.0] * dim),
            np.array([-3.0] * dim),
        ]
        for sp in structured_points:
            if calls >= budget or time.time() > deadline:
                break
            eval_and_track(sp)

        # Phase 2: CMA-ES for basin exploration
        if calls < budget - 50 and time.time() < deadline:
            # Use CMA-ES with adaptive covariance
            # Start from the best point found so far
            x0 = best_x.copy()
            
            # CMA-ES parameters
            pop_size = max(4, dim + 3)
            max_iter = min(20, (budget - calls - 30) // pop_size)
            
            # Initialize CMA-ES state
            rng = np.random.default_rng(42)
            mean = x0.copy()
            sigma = 1.5
            C = np.eye(dim)  # Covariance matrix
            B = np.eye(dim)  # Eigenvectors
            D = np.ones(dim)  # Eigenvalues
            pc = np.zeros(dim)  # Evolution path for covariance
            ps = np.zeros(dim)  # Evolution path for sigma
            c_sigma = 1.0 / (dim + 3)
            c_c = (4 + dim / dim) / (dim + 4 + 2 * dim / (dim + 3))
            mu = pop_size // 2  # Number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_c = (4 + dim / dim) / (dim + 4 + 2 * dim / (dim + 3))
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma

            for iteration in range(max_iter):
                if calls >= budget - 30 or time.time() > deadline:
                    break
                
                # Generate population
                population = []
                for i in range(pop_size):
                    if calls >= budget - 10 or time.time() > deadline:
                        break
                    # Sample from multivariate normal using current C
                    z = rng.standard_normal(dim)
                    x_new = mean + sigma * (B @ (D * z))
                    x_new = np.clip(x_new, -5.0, 5.0)
                    population.append(x_new)
                
                if len(population) < mu:
                    break
                
                # Evaluate population
                f_values = []
                for x in population:
                    f_val = eval_and_track(x)
                    f_values.append(f_val)
                
                if len(f_values) < mu:
                    break
                
                # Sort by fitness
                indices = np.argsort(f_values)
                selected = [population[i] for i in indices[:mu]]
                selected_f = [f_values[i] for i in indices[:mu]]
                
                # Update mean
                new_mean = sum(weights[i] * selected[i] for i in range(mu))
                
                # Update evolution paths
                if np.linalg.norm(new_mean - mean) > 1e-10:
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (new_mean - mean) / (sigma * np.sqrt(np.sum(D ** 2)))
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
                sigma = np.clip(sigma, 1e-10, 5.0)
                
                # Update covariance
                if iteration % 2 == 0 or np.linalg.norm(ps) / np.sqrt(2 * dim) > 1.4 + 2 / dim:
                    delta = np.outer(ps, ps) - np.eye(dim)
                    C = (1 + c_c * (2 - c_c)) * C + c_c * delta
                    C = (C + C.T) / 2  # Symmetrize
                
                # Update eigendecomposition periodically
                if iteration % 5 == 0:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.clip(D_sq, 1e-20, None)
                    D = np.sqrt(D_sq)
                
                mean = new_mean
                mean = np.clip(mean, -5.0, 5.0)
            
            # Track best from CMA-ES
            if best_f > float('inf'):
                best_f = min(selected_f) if selected_f else best_f

        # Phase 3: Local refinement with Nelder-Mead
        if calls < budget - 20 and time.time() < deadline:
            # Use Nelder-Mead starting from the best point
            x0 = best_x.copy()
            
            def wrapper(x):
                nonlocal calls
                if calls >= budget or time.time() > deadline:
                    return best_f
                x_clipped = np.clip(x, -5.0, 5.0)
                calls += 1
                val = func(x_clipped)
                return val
            
            # Nelder-Mead with reasonable initial simplex
            initial_simplex_size = 0.5
            options = {
                'maxiter': min(100, (budget - calls) // 2),
                'xatol': 1e-6,
                'fatol': 1e-8,
            }
            
            try:
                result = minimize(wrapper, x0, method='Nelder-Mead', options=options)
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0).copy()
            except Exception:
                pass

        # Phase 4: Coordinate descent fine-tuning
        if calls < budget - 10 and time.time() < deadline:
            x_curr = best_x.copy()
            f_curr = best_f
            
            # Multiple rounds of coordinate descent
            for round_idx in range(3):
                if calls >= budget - 5 or time.time() > deadline:
                    break
                
                improved = False
                for d in range(dim):
                    if calls >= budget - 5 or time.time() > deadline:
                        break
                    
                    # Golden-section search along coordinate d
                    lo = max(-5.0, x_curr[d] - 1.0)
                    hi = min(5.0, x_curr[d] + 1.0)
                    
                    if lo >= hi:
                        continue
                    
                    gr = (np.sqrt(5) - 1) / 2
                    a, b = lo, hi
                    c = b - gr * (b - a)
                    d_val = a + gr * (b - a)
                    
                    def func_1d(val, d=d, x=x_curr):
                        nonlocal calls
                        if calls >= budget or time.time() > deadline:
                            return f_curr
                        x_temp = x.copy()
                        x_temp[d] = val
                        x_temp = np.clip(x_temp, -5.0, 5.0)
                        calls += 1
                        val = func(x_temp)
                        return val
                    
                    fc = func_1d(c)
                    fd = func_1d(d_val)
                    
                    for _ in range(15):
                        if calls >= budget - 3 or time.time() > deadline:
                            break
                        if fc < fd:
                            b = d_val
                            d_val = c
                            fd = fc
                            c = b - gr * (b - a)
                            if abs(c - d_val) < 1e-10:
                                break
                            fc = func_1d(c)
                        else:
                            a = c
                            c = d_val
                            fc = fd
                            d_val = a + gr * (b - a)
                            if abs(c - d_val) < 1e-10:
                                break
                            fd = func_1d(d_val)
                    
                    opt_val = c if fc < fd else d_val
                    opt_f = min(fc, fd)
                    
                    if opt_f < f_curr:
                        x_curr[d] = opt_val
                        f_curr = opt_f
                        improved = True
                
                if not improved:
                    break
            
            if f_curr < best_f:
                best_f = f_curr
                best_x = x_curr.copy()

        # Phase 5: If budget remains, do a few more random restarts with local refinement
        rng_final = np.random.default_rng(12345)
        while calls < budget - 30 and time.time() < deadline:
            # Random restart
            x_start = rng_final.uniform(-5, 5, dim)
            eval_and_track(x_start)
            
            # Quick local refinement from this point
            if calls < budget - 15 and time.time() < deadline:
                x_local = x_start.copy()
                for d in range(dim):
                    if calls >= budget - 10 or time.time() > deadline:
                        break
                    # Simple 3-point search along coordinate d
                    center = x_local[d]
                    step = 0.5
                    candidates = [
                        np.clip(center - step, -5, 5),
                        center,
                        np.clip(center + step, -5, 5),
                    ]
                    best_val = float('inf')
                    best_cand = center
                    for cand in candidates:
                        if calls >= budget - 5 or time.time() > deadline:
                            break
                        x_temp = x_local.copy()
                        x_temp[d] = cand
                        x_temp = np.clip(x_temp, -5, 5)
                        val = eval_and_track(x_temp)
                        if val < best_val:
                            best_val = val
                            best_cand = cand
                    x_local[d] = best_cand
                
                if best_val < best_f:
                    best_f = best_val
                    best_x = x_local.copy()

        return best_x, best_f

# EVOLVE-BLOCK-END
