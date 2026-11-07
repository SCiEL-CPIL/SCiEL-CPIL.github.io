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
        deadline = start_time + 290  # Safety margin before 300s limit

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

        # Phase 1: Initial Exploration
        # Use a mix of random and structured points to find a good starting region.
        n_init = 20
        if budget <= n_init:
            x = np.random.uniform(-5, 5, dim)
            f = func(x)
            return x, f

        # Random points
        X_rand = np.random.uniform(-5, 5, (n_init // 2, dim))
        for x in X_rand:
            if not time_ok():
                break
            evaluate(x)

        # Grid-like points (sparse) to catch shifted minima
        # For dim=5, a full grid is too expensive, so use a few key points
        # Center and corners of the box
        key_points = []
        key_points.append(np.zeros(dim))
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = -4
            key_points.append(p)
            p[i] = 4
            key_points.append(p)
        
        for x in key_points:
            if evals >= budget:
                break
            if not time_ok():
                break
            evaluate(x)

        # Phase 2: CMA-ES with Adaptive Strategy
        # We'll implement a robust CMA-ES variant.
        # Key parameters for dim=5
        lam = 11
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
        sigma = 2.0  # Start with a large sigma to explore
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        no_improve_count = 0
        stagnation_threshold = 40  # Evaluations without improvement

        while evals + lam <= budget and time_ok():
            # Sample
            Z = np.random.randn(lam, dim)
            X = m + sigma * (Z @ L.T)
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.array([evaluate(x) for x in X])

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

            # Stagnation Restart
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

        # Phase 3: Local Refinement (Nelder-Mead or Coordinate Descent)
        # Use remaining budget for fine-tuning
        if evals < budget and best_x is not None and time_ok():
            # Simple coordinate descent with adaptive step
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
