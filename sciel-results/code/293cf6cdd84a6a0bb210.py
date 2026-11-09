"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Robust CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    - Random sampling phase for initialization with PCA-based covariance init.
    - Corrected CMA-ES implementation with covariance matrix adaptation.
    - Stagnation detection with restarts (reset covariance, increase sigma).
    - Time guard to stay within wall-clock limits.
    - Final high-precision polish with Nelder-Mead simplex method.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Random sampling to find a good initial point
        n_samples = min(80, budget // 8)
        best_x = None
        best_fx = float('inf')
        samples = []

        for i in range(n_samples):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            samples.append((fx, x))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        calls_used = n_samples

        # Initialize covariance from top samples (PCA-like)
        samples.sort(key=lambda t: t[0])
        n_init = max(5, min(15, len(samples) // 4))
        init_points = np.array([s[1] for s in samples[:n_init]])
        init_mean = init_points.mean(axis=0)
        init_centered = init_points - init_mean
        try:
            C_init = np.cov(init_centered, rowvar=False) + 1e-6 * np.eye(dim)
            # Scale to reasonable magnitude
            eigvals_init = np.linalg.eigvalsh(C_init)
            if np.max(eigvals_init) > 0:
                C_init = C_init / np.max(eigvals_init) * 4.0  # scale to ~[-2,2] range
        except:
            C_init = np.eye(dim)

        # Phase 2: CMA-ES with restarts
        # CMA-ES Parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        # Adaptation rates
        c_sigma = (pop_size + 2) / (dim + 4)
        c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        cmu = min(0.5, c1 * (pop_size // 2 - 1) / 2)
        d_sigma = 1 + 2 * max(0, (dim / pop_size - 1) / 3) + cmu
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State variables
        m = best_x.copy()
        sigma = 1.5  # Slightly larger initial step
        C = C_init.copy()
        try:
            C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
        except np.linalg.LinAlgError:
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        stall_count = 0
        max_stall = 20
        restart_sigma_mult = 1.5
        consecutive_restarts = 0
        max_consecutive_restarts = 3

        while calls_used < budget:
            if time.time() > deadline - 5.0:
                break

            # Generate population
            offspring = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = m + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            evaluated = []
            for y in offspring:
                if calls_used >= budget or time.time() > deadline - 5.0:
                    break
                fy = func(y)
                calls_used += 1
                evaluated.append((fy, y))

            if len(evaluated) < pop_size:
                break

            # Sort by fitness
            evaluated.sort(key=lambda t: t[0])

            # Update best
            if evaluated[0][0] < best_fx:
                best_fx = evaluated[0][0]
                best_x = evaluated[0][1].copy()
                stall_count = 0
            else:
                stall_count += 1

            # Select top mu individuals
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * evaluated[i][1]

            # Calculate step in normalized space
            # z_w = sum w_i * (y_i - m) / sigma * C^{-1/2}
            # We have C_sqrt, so C^{-1/2} = C_sqrt^{-1} = C_sqrt.T^{-1}
            # Actually, y_i = m + sigma * C_sqrt @ z_i => z_i = C_sqrt^{-1} (y_i - m) / sigma
            # We need z_w = sum w_i * z_i
            z_w = np.zeros(dim)
            for i in range(mu):
                # Solve C_sqrt @ z_i = (y_i - m) / sigma
                # Using np.linalg.solve for stability
                try:
                    z_i = np.linalg.solve(C_sqrt, (evaluated[i][1] - m) / sigma)
                except np.linalg.LinAlgError:
                    z_i = (evaluated[i][1] - m) / sigma
                z_w += weights[i] * z_i

            # Update p_sigma
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (dim + 1)) * chi_n) else 0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_w
            
            # Update sigma
            if h_sigma:
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            sigma = float(np.clip(sigma, 0.001, 10.0))

            # Update p_c
            h_c = 1 if (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c1) ** 2) < (1.4 + 2 / (dim + 1)) * np.sqrt((c1 / (2 - c1)) - 1)) else 0
            p_c = (1 - c1) * p_c + h_c * np.sqrt(c1 * (2 - c1) * mu_eff) * z_w

            # Update Covariance C
            # C = (1 - c1 - cmu) * C + c1 * (p_c p_c^T + (1 - h_c) * c1 * (1 - c1) * C) + cmu * sum w_i y_i y_i^T
            # Note: y_i here is in normalized space (z_i)
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(p_c, p_c) + (1 - h_c) * c1 * (1 - c1) * C)
            
            for i in range(mu):
                try:
                    z_i = np.linalg.solve(C_sqrt, (evaluated[i][1] - m) / sigma)
                except np.linalg.LinAlgError:
                    z_i = (evaluated[i][1] - m) / sigma
                C += cmu * weights[i] * np.outer(z_i, z_i)

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update mean
            m = y_w
            m = np.clip(m, -5, 5)

            # Recompute C_sqrt
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(dim)

            # Restart logic
            if stall_count >= max_stall:
                consecutive_restarts += 1
                # Reset covariance to identity, increase sigma
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                sigma = max(sigma * restart_sigma_mult, 1.0)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stall_count = 0
                max_stall = min(max_stall + 10, 60)
                restart_sigma_mult *= 1.2
                
                # If too many consecutive restarts, do a fresh random search
                if consecutive_restarts >= max_consecutive_restarts:
                    n_fresh = min(20, budget - calls_used)
                    if n_fresh > 0:
                        for _ in range(n_fresh):
                            if calls_used >= budget or time.time() > deadline - 5.0:
                                break
                            x = rng.uniform(-5, 5, dim)
                            fx = func(x)
                            calls_used += 1
                            if fx < best_fx:
                                best_fx = fx
                                best_x = x.copy()
                        m = best_x.copy()
                        sigma = 2.0
                        consecutive_restarts = 0
                        restart_sigma_mult = 1.5
                        max_stall = 20

        # Phase 3: Final polish with Nelder-Mead simplex method
        if calls_used < budget and time.time() < deadline - 2.0:
            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5  # shrink
            max_iter_nm = min(100, (budget - calls_used) // dim)
            
            # Initialize simplex
            simplex = []
            simplex_fx = []
            
            # Start with best point
            x0 = best_x.copy()
            f0 = func(x0)
            calls_used += 1
            simplex.append(x0.copy())
            simplex_fx.append(f0)
            if f0 < best_fx:
                best_fx = f0
                best_x = x0.copy()
            
            # Add perturbed points
            step = 0.1
            for i in range(dim):
                xi = x0.copy()
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls_used += 1
                simplex.append(xi.copy())
                simplex_fx.append(fi)
                if fi < best_fx:
                    best_fx = fi
                    best_x = xi.copy()
            
            simplex = np.array(simplex)
            simplex_fx = np.array(simplex_fx)
            
            for _ in range(max_iter_nm):
                if calls_used >= budget or time.time() > deadline - 1.0:
                    break
                
                # Sort by function value
                order = np.argsort(simplex_fx)
                simplex = simplex[order]
                simplex_fx = simplex_fx[order]
                
                # Update best
                if simplex_fx[0] < best_fx:
                    best_fx = simplex_fx[0]
                    best_x = simplex[0].copy()
                
                # Compute centroid of all but worst
                centroid = simplex[:-1].mean(axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_used += 1
                if f_ref < best_fx:
                    best_fx = f_ref
                    best_x = x_ref.copy()
                
                if f_ref < simplex_fx[0]:
                    # Expansion
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls_used += 1
                    if f_exp < best_fx:
                        best_fx = f_exp
                        best_x = x_exp.copy()
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_fx[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_fx[-1] = f_ref
                elif f_ref < simplex_fx[-2]:
                    simplex[-1] = x_ref
                    simplex_fx[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < simplex_fx[-1]:
                        x_con = centroid + rho * (simplex[-1] - centroid)
                    else:
                        x_con = centroid - rho * (centroid - simplex[-1])
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls_used += 1
                    if f_con < best_fx:
                        best_fx = f_con
                        best_x = x_con.copy()
                    if f_con < f_ref:
                        simplex[-1] = x_con
                        simplex_fx[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_i = func(simplex[i])
                            calls_used += 1
                            simplex_fx[i] = f_i
                            if f_i < best_fx:
                                best_fx = f_i
                                best_x = simplex[i].copy()
                        if calls_used >= budget or time.time() > deadline - 1.0:
                            break

        return best_x, best_fx
# EVOLVE-BLOCK-END
