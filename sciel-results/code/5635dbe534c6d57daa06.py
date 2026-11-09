"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    
    Improvements over baseline:
    1. Larger population size for better covariance estimation in high-anisotropy landscapes.
    2. Robust initialization with PCA-based covariance from a diverse initial sample.
    3. Careful handling of numerical stability and step size adaptation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # --- Parameter Configuration ---
        # Larger population for better exploration of the anisotropic landscape
        # Standard CMA-ES suggests lambda ~ 4 + 3*log(dim). For dim=5, this is ~10.
        # We use 12 to allow for more robust selection and covariance updates.
        lam = 12
        mu = lam // 2  # Select top half

        # Weights for the selected points
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Expected norm of standard normal vector
        chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim * dim) + 1 / (12 * dim ** 3))

        # --- Phase 1: Initialization ---
        # We use a small number of initial samples to estimate the orientation
        # of the search space, which is particularly useful for ill-conditioned functions.
        n_init = 20
        if n_init > budget - lam:
            n_init = max(2, budget - lam)

        init_samples = []
        init_fits = []
        calls = 0

        # Sample initial points. We use a wide range to ensure coverage.
        # The minimizer is in [-4, 4]^5, so [-5, 5] is safe.
        for i in range(n_init):
            x_init = rng.uniform(-5, 5, size=dim)
            f_init = func(x_init)
            init_samples.append(x_init)
            init_fits.append(f_init)
            calls += 1
            if calls >= budget:
                break

        init_samples = np.array(init_samples)
        init_fits = np.array(init_fits)

        # Initialize mean x at the best initial point
        best_idx = np.argmin(init_fits)
        x = init_samples[best_idx].copy()
        best_x = x.copy()
        best_f = init_fits[best_idx]

        # Initialize covariance C based on the spread of the initial samples.
        # We use a weighted covariance where better points have higher weight.
        # This helps align the covariance with the funnel direction if detectable.
        f_min = init_fits.min()
        f_max = init_fits.max()
        if f_max > f_min:
            # Exponential weighting: better points get significantly more weight
            w_init = np.exp(-(init_fits - f_min) / (f_max - f_min + 1e-10))
        else:
            w_init = np.ones(len(init_samples))
        
        w_init /= w_init.sum()
        
        # Calculate weighted mean and covariance
        mean_init = np.sum(w_init[:, None] * init_samples, axis=0)
        # Use the best point as the center for covariance estimation to focus on the promising region
        centered = init_samples - best_x
        C = np.zeros((dim, dim))
        for i in range(len(init_samples)):
            C += w_init[i] * np.outer(centered[i], centered[i])
        
        # Normalize
        C /= (1 - (w_init ** 2).sum()) if (1 - (w_init ** 2).sum()) > 1e-10 else 1.0
        
        # Ensure C is positive definite and well-scaled
        # Add a small regularization to avoid singularities
        C += 1e-6 * np.eye(dim)
        
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-8, None)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Initialize step size
        # Start with a moderate step size. The initial covariance scale is captured in C.
        # We set sigma=1.0 and let the covariance matrix handle the scaling.
        sigma = 1.0
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # --- Phase 2: CMA-ES Loop ---
        while calls < budget:
            # Sample population
            ys = np.zeros((lam, dim))
            fs = np.zeros(lam)

            # Pre-compute Cholesky decomposition for efficiency
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                # Fallback if C is not positive definite
                L = np.eye(dim)

            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x + sigma * (L @ z)
                
                # Clip to box [-5, 5]
                ys[i] = np.clip(y, -5, 5)
                fs[i] = func(ys[i])
                calls += 1
                if calls >= budget:
                    break

            if calls >= budget:
                break

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = ys[idx]
            fs_sorted = fs[idx]

            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = ys_sorted[0].copy()

            # Weighted mean of selected points
            y_w = np.sum(weights[:, None] * (ys_sorted[:mu] - x), axis=0)

            # Update evolution path ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w

            # h_sigma: check if ps is within expected range
            h_sigma = 1.0
            if sigma > 1e-10:
                ps_norm = np.linalg.norm(ps)
                # Approximate the expected norm. 
                # The term (1 - (1 - c_sigma)^(2*gen)) approximates the decay.
                # We use a simple heuristic here to avoid tracking generation count strictly.
                exp_norm = chi_n * np.sqrt(1 - (1 - c_sigma) ** 2)
                if exp_norm > 1e-10:
                    h_sigma = 1.0 if ps_norm / exp_norm < 1.4 * (1 + 2 / (dim + 1)) else 0.0

            # Update evolution path pc
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + \
                c_1 * np.outer(pc, pc) + \
                np.sum([weights[i] * np.outer(ys_sorted[i] - x, ys_sorted[i] - x) for i in range(mu)], axis=0) * c_mu

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update step size
            ps_norm = np.linalg.norm(ps)
            sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x = x + y_w

        return best_x, best_f
# EVOLVE-BLOCK-END
