"""CMA-ES optimizer for BBOB f20 (Schwefel) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with covariance matrix adaptation for rotated Schwefel."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0  # safety margin before 300s kill

        # CMA-ES parameters
        lam = 4 * dim          # population size = 20
        mu = lam // 2          # number of parents = 10
        # log weights for top-mu
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        best_x = x.copy()
        best_f = func(x)
        calls = 1

        def sample_batch():
            nonlocal calls, deadline
            # Generate lam samples
            samples = np.empty((lam, dim))
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    # Pad remaining with current best (won't be evaluated)
                    samples[i] = best_x
                    continue
                # z ~ N(0,1)
                z = np.random.standard_normal(dim)
                # y = C_inv_sqrt @ z  (use Cholesky-like: C_inv_sqrt is maintained)
                y = C_inv_sqrt @ z
                samples[i] = x + sigma * y
                calls += 1
            return samples

        def update_covariance(selected, weights):
            nonlocal C, C_inv_sqrt, pc, ps, sigma, c_sigma, c_c, c_1, c_mu, chi_n
            # selected: (mu, dim)
            # Compute C update
            diff = selected - x
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + delta_h * c_sigma * (1 - c_sigma) * C) + c_mu * sum w_i * diff_i @ diff_i.T
            # h_sigma = (||ps|| / (1 - (1 - c_sigma)^(1/10)) ) / chi_n < 2.1/chi_n
            h_sigma = (np.linalg.norm(ps) / (1 - (1 - c_sigma) ** (1 / 10.0))) / chi_n
            delta_h = (1 - h_sigma) * c_sigma * (1 - c_sigma)

            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (np.outer(pc, pc) + delta_h * C)
            for i in range(mu):
                C_new += c_mu * weights[i] * np.outer(diff[i], diff[i])
            # Symmetrize
            C_new = 0.5 * (C_new + C_new.T)
            C = C_new

            # Update C_inv_sqrt via eigendecomposition (dim=5, cheap)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ diff.mean(axis=0))

            # Update pc
            pc = (1 - c_c) * pc + h_sigma * c_c * diff.mean(axis=0)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

        # Main loop
        while calls < budget and time.time() < deadline:
            # Sample
            samples = np.empty((lam, dim))
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                z = np.random.standard_normal(dim)
                y = C_inv_sqrt @ z
                samples[i] = x + sigma * z  # simplified: use z directly, C applied via C_inv_sqrt for ps
                # Actually: sample = x + sigma * C^{1/2} z. We have C_inv_sqrt, so C^{1/2} z = ?
                # Let me redo: we want C^{1/2} z. We have C_inv_sqrt = C^{-1/2}.
                # So C^{1/2} z = (C_inv_sqrt^{-1}) z. But that's expensive.
                # Alternative: just use z directly and let C adapt via the update.
                # Actually the standard CMA-ES samples y = C^{1/2} z. Let me compute C^{1/2}.
                calls += 1
            # Redo properly
            samples = np.empty((lam, dim))
            # Compute C^{1/2} from eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            C_half = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            eval_indices = []
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                z = np.random.standard_normal(dim)
                samples[i] = np.clip(x + sigma * (C_half @ z), -5, 5)
                eval_indices.append(i)
                calls += 1

            n_eval = len(eval_indices)
            if n_eval == 0:
                break

            # Evaluate
            fvals = np.empty(n_eval)
            for j, i in enumerate(eval_indices):
                fvals[j] = func(samples[i])

            # Sort by fitness
            order = np.argsort(fvals)
            sorted_samples = samples[eval_indices[order]]
            sorted_fvals = fvals[order]

            # Update best
            if sorted_fvals[0] < best_f:
                best_f = sorted_fvals[0]
                best_x = sorted_samples[0].copy()

            # Update mean
            x = np.sum(w[:, None] * sorted_samples[:mu], axis=0)

            # Update covariance and step size
            update_covariance(sorted_samples[:mu], w)

        return best_x, best_f
# EVOLVE-BLOCK-END
