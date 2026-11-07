"""Rewritten optimizer: Multi-start CMA-ES for Sharp Ridge optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Optimizer using Multi-start initialization followed by CMA-ES.
    CMA-ES is robust to the anisotropy of sharp ridge functions by adapting
    its covariance matrix to the local landscape shape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin for 300s limit

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = clip(x)
            f = float(func(x))
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # ---------- Phase 1: Multi-start Initialization ----------
        # We need to find the basin of the sharp ridge.
        # Random sampling is effective for locating the rough region.
        n_starts = 20
        for _ in range(n_starts):
            if not elapsed_ok() or evaluated >= budget:
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            eval_point(x)

        if best_x is None:
            return None, np.inf

        # ---------- Phase 2: CMA-ES Refinement ----------
        # Use CMA-ES to exploit the anisotropic landscape.
        # It will shrink sigma in the perpendicular direction and maintain it along the ridge.
        
        x_mean = best_x.copy()
        
        # CMA-ES parameters
        # Lambda is the population size. For dim=5, lambda=5 is efficient for budget.
        # Budget remaining is ~480. 480 / 5 = 96 generations.
        lambda_ = 5
        
        # Initial sigma
        sigma = 1.0
        
        # Covariance matrix C
        C = np.eye(dim)
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Weights for recombination
        # Use standard weights: w_i = log(lambda/2 + 0.5) - log(i)
        weights = np.array([np.log(lambda_ / 2 + 0.5) - np.log(i + 1) for i in range(lambda_)])
        weights /= np.sum(weights)
        mu = lambda_ // 2  # Number of parents used for covariance update
        
        # Effective population size for mu-weighted recombination
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)
        
        # Update strategy parameters
        c_sigma = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ - 2) / (dim + 1) - 1) + c_sigma)
        cs = (dim + lambda_ + 2) / (dim + 5 + 3 * lambda_)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * ((mu_eff - 2) + 1) / ((dim + 2) ** 2 + mu_eff))

        # Helper to get transformation matrix from covariance
        def get_transformation(C_mat):
            try:
                L = np.linalg.cholesky(C_mat)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.clip(eigvals, 1e-10, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
            return L

        T = get_transformation(C)

        while evaluated < budget and elapsed_ok():
            # Generate population
            pop_x = []
            pop_f = []
            pop_z = []
            
            for i in range(lambda_):
                if evaluated >= budget or not elapsed_ok():
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (T @ z)
                x_i = clip(y)
                f_i = eval_point(x_i)
                pop_x.append(x_i)
                pop_f.append(f_i)
                pop_z.append(z)
            
            if len(pop_x) < lambda_:
                break

            # Sort by fitness
            indices = np.argsort(pop_f)
            pop_x = [pop_x[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]
            pop_z = [pop_z[i] for i in indices]

            # Update mean
            x_new_mean = np.sum([weights[i] * pop_x[i] for i in range(mu)], axis=0)
            
            # Update sigma
            # Calculate norm of z_mean
            z_mean = np.sum([weights[i] * pop_z[i] for i in range(mu)], axis=0)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (T_inv := np.linalg.inv(T)) @ z_mean
            # Note: T_inv is not explicitly stored, but we can compute it or use the fact that 
            # standard CMA-ES updates sigma based on ||ps|| / E||N(0,I)||.
            # For simplicity and robustness in a short script, we often approximate or 
            # use the standard formula:
            # sigma = sigma * exp(c_sigma/d_sigma * (||ps|| / (sqrt(mu_eff) * E||N||) - 1))
            # E||N(0, I_d)|| approx sqrt(d) * (1 - 1/(4d) + 1/(32d^2))
            norm_ps = np.linalg.norm(ps)
            expected_norm = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2))
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_ps / (np.sqrt(mu_eff) * expected_norm) - 1))
            
            # Update covariance
            # pc = (1-cs)*pc + cs*sqrt(mu_eff)*(x_new_mean - x_mean)/sigma
            pc = (1 - cs) * pc + cs * np.sqrt(mu_eff) * (x_new_mean - x_mean) / sigma
            
            # Rank-1 update
            C = (1 + c1 * (mu_eff - 2) - c1) * C + c1 * np.outer(pc, pc)
            
            # Rank-mu update
            if cmu > 0:
                diff = np.array([(pop_x[i] - x_mean) / sigma for i in range(mu)])
                # diff is (mu, dim)
                # weights are (mu,)
                # We need sum of w_i * diff_i * diff_i^T
                # Efficient way: (diff.T * weights) @ diff ? No.
                # diff_weighted = diff * weights[:, None]
                # rank_mu = (diff_weighted.T @ diff)
                diff_weighted = diff * weights[:mu, None]
                rank_mu = diff_weighted.T @ diff
                C += cmu * rank_mu

            x_mean = x_new_mean
            
            # Update transformation matrix
            T = get_transformation(C)
            
            # Convergence check
            if np.std(pop_f) < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
