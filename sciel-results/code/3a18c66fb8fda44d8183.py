"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: CMA-ES-like with Hessian Newton polish for Discus."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # --- Phase 1: Radial sweep initialization to find anisotropic axis ---
        # Discus has a narrow valley; radial sweep helps find the right direction
        best_x = None
        best_f = np.inf

        # Start with center
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_x, best_f = x0.copy(), f0

        # Radial sweep: try points along different directions at various radii
        n_radial = 20
        for i in range(n_radial):
            # Generate diverse directions
            if i == 0:
                direction = np.ones(dim) / np.sqrt(dim)
            else:
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)

            # Try multiple radii
            for r in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
                x = np.clip(r * direction, -5, 5)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Also try negative directions
        for i in range(n_radial):
            if i == 0:
                direction = -np.ones(dim) / np.sqrt(dim)
            else:
                direction = -rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)

            for r in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
                x = np.clip(r * direction, -5, 5)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # --- Phase 2: CMA-ES-like evolution strategy ---
        # Use the best point found as starting point
        x = best_x.copy()
        fx = best_f

        # Estimate initial sigma from the spread of good points
        sigma = 0.5  # moderate initial step

        # Track best
        best_x_es = x.copy()
        best_f_es = fx

        # CMA-ES-like parameters
        lam = max(4, dim)  # population size
        c1 = (2 + 2 / (dim + 1)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, 2 * (2 / (dim + 1) - 1) / ((dim + 2) ** 2 + 4))
        cs = (dim + 1.3) / (dim + 5)
        ds = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 1)) - 1) + cs
        cc = (4 + dim / dim) / (4 + dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        D = np.ones(dim)  # eigenvalues of C
        B = np.eye(dim)   # eigenvectors of C

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Budget management
        calls_used = n_radial * 16 + n_radial * 16 + 1  # radial sweep calls + center
        remaining = budget - calls_used
        if remaining < 0:
            remaining = 0

        # Main evolution loop
        for gen in range(max(1, remaining // lam)):
            if time.time() > deadline - 5:
                break
            if remaining <= 0:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                # Transform using eigendecomposition: B @ diag(D) @ z
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                offspring.append((y, fy))
                remaining -= 1

            # Sort by fitness
            offspring.sort(key=lambda t: t[1])

            # Update best
            if offspring[0][1] < best_f_es:
                best_f_es = offspring[0][1]
                best_x_es = offspring[0][0].copy()

            # Selection weights
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights ** 2)

            # Mean update
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * offspring[i][0]

            # Update evolution paths
            d_mean = x_new - x
            x = x_new

            # Sigma adaptation
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (B.T @ (D @ (B.T @ (d_mean / sigma))))
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (remaining // lam + 1))) / chi_n < 1.4 + 2 / (dim + 1)
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10)

            # Covariance matrix adaptation
            if h_sigma:
                p_c = (1 - c1) * p_c + np.sqrt(c1 * (2 - c1) * mu_eff) * d_mean / sigma
            else:
                p_c = (1 - c1) * p_c

            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                zi = B.T @ (D @ (B.T @ ((offspring[i][0] - x_new) / sigma)))
                rank_mu += weights[i] * np.outer(zi, zi)

            C = (1 + c1 * (np.outer(p_c, p_c) - C) + cmu * rank_mu) * C
            # Ensure symmetry
            C = (C + C.T) / 2

            # Re-estimate eigenvalues periodically
            if gen % 10 == 0:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, 1e12)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                    C_inv = np.linalg.inv(C)
                except:
                    pass

            if time.time() > deadline - 30:
                break

        # --- Phase 3: Hessian Newton polish ---
        # Fit local Hessian via finite differences and do Newton steps
        x = best_x_es.copy()
        fx = best_f_es

        # Check if we have budget left
        if remaining > 10 and time.time() < deadline - 10:
            # Estimate Hessian using finite differences
            def estimate_hessian(x, func, dim, h=1e-4):
                H = np.zeros((dim, dim))
                f0 = func(x)
                for i in range(dim):
                    for j in range(i, dim):
                        xi = x.copy()
                        xj = x.copy()
                        xij = x.copy()
                        xji = x.copy()

                        xi[i] += h
                        xi[j] += h if i != j else 0
                        xj[i] -= h
                        xj[j] -= h if i != j else 0
                        xij[i] += h
                        xij[j] -= h if i != j else 0
                        xji[i] -= h
                        xji[j] += h if i != j else 0

                        fxi = func(np.clip(xi, -5, 5))
                        fxj = func(np.clip(xj, -5, 5))
                        fxij = func(np.clip(xij, -5, 5))
                        fxji = func(np.clip(xji, -5, 5))

                        if i == j:
                            H[i, i] = (fxi + fxj - 2 * f0) / (h ** 2)
                        else:
                            H[i, j] = H[j, i] = (fxi - fxj - fxij + fxji) / (4 * h ** 2)
                return H

            # Do a few Newton iterations
            for newton_iter in range(5):
                if remaining < 3 or time.time() > deadline - 5:
                    break

                try:
                    # Estimate gradient
                    grad = np.zeros(dim)
                    h = 1e-4
                    for i in range(dim):
                        xp = x.copy()
                        xm = x.copy()
                        xp[i] += h
                        xm[i] -= h
                        grad[i] = (func(np.clip(xp, -5, 5)) - func(np.clip(xm, -5, 5))) / (2 * h)
                        remaining -= 2

                    # Estimate Hessian
                    H = estimate_hessian(x, func, dim, h=1e-3)
                    remaining -= 2 * dim * dim  # approximate count

                    # Regularize Hessian
                    eigvals, eigvecs = np.linalg.eigh(H)
                    eigvals = np.clip(eigvals, 1e-6, None)
                    H_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T

                    # Newton step
                    try:
                        step = np.linalg.solve(H_reg, -grad)
                    except:
                        step = -grad / (np.linalg.norm(grad) + 1e-8)

                    # Damped Newton step
                    alpha = 1.0
                    for _ in range(20):
                        x_trial = np.clip(x + alpha * step, -5, 5)
                        f_trial = func(x_trial)
                        remaining -= 1
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            if fx < best_f_es:
                                best_f_es = fx
                                best_x_es = x.copy()
                            break
                        alpha *= 0.5

                except Exception:
                    break

                if time.time() > deadline - 5:
                    break

        # Final call to ensure we've made at least one call
        if calls_used + remaining >= budget:
            # We've used most of our budget
            pass

        # Return the best point found
        return best_x_es, best_f_es
# EVOLVE-BLOCK-END
