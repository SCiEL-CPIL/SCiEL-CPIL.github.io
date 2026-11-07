import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        def time_ok():
            return time.time() < deadline

        best_x = None
        best_f = float('inf')
        evals = 0

        def evaluate(x):
            nonlocal best_x, best_f, evals
            x = np.clip(x, -5, 5)
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Deterministic Space-Filling Initial Exploration
        # Use a Halton sequence to cover the space efficiently and deterministically.
        # This avoids random clustering and ensures good coverage of [-5, 5]^5.
        
        def halton_sequence(n, dim):
            """Generate n points in dim dimensions using Halton sequence."""
            points = []
            for i in range(1, n + 1):
                point = []
                for d in range(dim):
                    prime = 2 ** (d + 1)  # 2, 4, 8, 16, 32... (using powers of 2 for simplicity, though technically primes are better)
                    # Actually, let's use proper primes: 2, 3, 5, 7, 11
                    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                    p = primes[d] if d < len(primes) else 2
                    
                    f = 1.0
                    r = 0.0
                    i_temp = i
                    while i_temp > 0:
                        f /= p
                        r += f * (i_temp % p)
                        i_temp //= p
                    point.append(r)
                # Map from [0, 1]^dim to [-5, 5]^dim
                point_mapped = [2 * 5 * (p - 0.5) - 5 for p in point]  # Slightly shifted to avoid boundaries
                points.append(point_mapped)
            return np.array(points)

        n_init = 30
        if budget <= n_init:
            # If budget is very small, just do a few random calls
            for _ in range(min(budget, 10)):
                x = np.random.uniform(-5, 5, dim)
                evaluate(x)
            return best_x, best_f

        # Generate Halton points
        halton_points = halton_sequence(n_init, dim)
        
        # Also add some random points for robustness
        n_random = 10
        random_points = np.random.uniform(-5, 5, (n_random, dim))
        
        # Combine and shuffle
        all_points = np.vstack([halton_points, random_points])
        np.random.shuffle(all_points)
        
        for x in all_points:
            if not time_ok() or evals >= budget:
                break
            evaluate(x)

        if best_x is None:
            # Fallback if no evaluation happened
            x = np.zeros(dim)
            evaluate(x)

        # Phase 2: Surrogate-Guided Search (Simplified Kriging/Expected Improvement)
        # We'll use a simpler approach: maintain a set of evaluated points and their values,
        # and use a simple distance-based heuristic to select the next point.
        # For better performance, we'll combine this with CMA-ES.
        
        # Actually, let's use a hybrid: CMA-ES for global search, with occasional
        # re-sampling from the surrogate model.
        
        # For now, let's stick with a robust CMA-ES implementation, but with
        # better initialization from the Halton exploration.
        
        # CMA-ES Parameters
        lam = max(10, dim + 3)  # Population size
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_lambda = np.sqrt(dim * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2)))

        # Start CMA-ES from the best point found so far
        m = best_x.copy()
        sigma = 1.5  # Start with a moderate sigma
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        no_improve_count = 0
        stagnation_threshold = 30  # Evaluations without improvement

        # Track evaluated points for potential surrogate use
        X_evaluated = [best_x.copy()]
        F_evaluated = [best_f]

        while evals + lam <= budget and time_ok():
            # Sample
            Z = np.random.randn(lam, dim)
            X = m + sigma * (Z @ L.T)
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.array([evaluate(x) for x in X])
            
            # Update tracking
            for i in range(lam):
                X_evaluated.append(X[i].copy())
                F_evaluated.append(F[i])

            # Sort
            order = np.argsort(F)
            F_sorted = F[order]
            X_sorted = X[order]

            # Update stagnation counter
            if F_sorted[0] < best_f - 1e-10:
                no_improve_count = 0
            else:
                no_improve_count += lam

            # Update Mean
            m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

            # Update Sigma and Covariance
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-18, None)

            v = (m_new - m) / sigma
            v_rot = eigvecs.T @ v
            v_scaled = v_rot / np.sqrt(eigvals)
            v_inv_half = eigvecs @ v_scaled

            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + \
                      np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v_inv_half

            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_lambda * (np.linalg.norm(p_sigma) / chi_lambda - 1))

            # h_sigma
            h_sigma_thresh = (1 + 1.4 / mu_eff) * np.sqrt((1 - c_c) * (2 - c_c))
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / chi_lambda) < h_sigma_thresh else 0.0

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * v_inv_half

            # Update C
            Y = (X_sorted[:mu] - m) / sigma
            M2 = (Y.T * weights) @ Y

            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * M2
            C = (C + C.T) / 2

            # Update m
            m = m_new

            # Update Cholesky
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = 1.0
                m = best_x.copy()

            # Stagnation Restart with Surrogate-Inspired Resampling
            if no_improve_count > stagnation_threshold:
                # Reset around best point with larger step
                m = best_x.copy()
                sigma = max(sigma * 1.5, 0.5)
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                no_improve_count = 0
                
                # Occasionally, try a point far from the current best to escape local minima
                if evals + 1 < budget and time_ok():
                    x_far = np.random.uniform(-5, 5, dim)
                    f_far = evaluate(x_far)
                    X_evaluated.append(x_far.copy())
                    F_evaluated.append(f_far)

        # Phase 3: Local Refinement
        # Use remaining budget for fine-tuning with coordinate descent
        if evals < budget and best_x is not None and time_ok():
            step = 0.1
            x = best_x.copy()
            f = best_f
            
            while evals < budget and time_ok():
                improved = False
                for i in range(dim):
                    if evals >= budget:
                        break
                    for direction in [-1, 1]:
                        x_trial = x.copy()
                        x_trial[i] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = evaluate(x_trial)
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-5:
                        break
                else:
                    step = min(step * 1.1, 0.5)

        return best_x, best_f
