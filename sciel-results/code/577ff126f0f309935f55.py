"""Optimizer for BBOB f4 (Büche-Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np

class Optimizer:
    """
    Hybrid optimizer for BBOB f4 (Büche-Rastrigin).
    Strategy:
    1. Sobol Sequence for dense global exploration (first 80 calls).
    2. Cross-Entropy Method (CEM) to adaptively concentrate samples in promising regions.
       CEM naturally handles the rotated/shifted landscape by updating mean and covariance.
    3. Final (1+1)-ES polish from the best CEM result for high-precision refinement.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        t0 = time.time()
        deadline = t0 + 285.0

        calls = 0
        best_x = None
        best_f = float('inf')

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def check_deadline():
            nonlocal calls
            return calls >= budget or time.time() > deadline

        # --- Phase 1: Coarse exploration with Sobol-like sampling ---
        n_phase1 = 80
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True, seed=42)
            next_pow2 = 1
            while next_pow2 < n_phase1:
                next_pow2 *= 2
            points = sampler.random(next_pow2)[:n_phase1]
        except Exception:
            points = rng.random((n_phase1, d))
        
        points = points * 10.0 - 5.0

        candidates = []
        for i in range(n_phase1):
            if check_deadline():
                break
            x = clip(points[i])
            fx = float(func(x))
            calls += 1
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            x0 = np.zeros(d)
            best_f = float(func(x0))
            best_x = x0.copy()
            calls += 1
            candidates.append((best_f, best_x.copy()))

        candidates.sort(key=lambda item: item[0])

        # --- Phase 2: Cross-Entropy Method (CEM) ---
        # CEM iteratively samples from a Gaussian, keeps the top 'elite' fraction,
        # and updates mean/covariance to concentrate in the best region.
        
        remaining = budget - calls
        if remaining <= 0:
            return best_x, best_f

        # CEM parameters
        n_samples = 30  # samples per iteration
        n_elite = 8     # number of elite samples to keep
        n_iters = max(1, min(15, (remaining - 50) // n_samples))  # reserve 50 calls for polish
        
        # Initialize distribution from best candidates
        # Use top 10 candidates to initialize mean and covariance
        top_k = min(10, len(candidates))
        top_points = np.array([c[1] for c in candidates[:top_k]])
        mu = np.mean(top_points, axis=0)
        cov = np.cov(top_points, rowvar=False)
        if cov.shape != (d, d):
            cov = np.eye(d) * 1.0
        # Add small regularization to ensure positive definiteness
        cov += 1e-6 * np.eye(d)

        # Run CEM iterations
        for it in range(n_iters):
            if check_deadline():
                break
            
            # Generate samples
            samples = []
            for _ in range(n_samples):
                if check_deadline():
                    break
                try:
                    x = rng.multivariate_normal(mu, cov)
                except np.linalg.LinAlgError:
                    # Fallback if covariance is not positive definite
                    x = mu + rng.standard_normal(d) * np.sqrt(np.diag(cov))
                x = clip(x)
                fx = float(func(x))
                calls += 1
                samples.append((fx, x.copy()))
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            if len(samples) < n_elite:
                break

            # Sort by fitness and keep elite
            samples.sort(key=lambda s: s[0])
            elites = [s[1] for s in samples[:n_elite]]
            elites_arr = np.array(elites)
            
            # Update distribution parameters
            mu_new = np.mean(elites_arr, axis=0)
            cov_new = np.cov(elites_arr, rowvar=False)
            if cov_new.shape != (d, d):
                cov_new = np.eye(d) * 0.5
            # Blend with previous to avoid abrupt changes
            mu = 0.5 * mu + 0.5 * mu_new
            cov = 0.5 * cov + 0.5 * cov_new
            cov += 1e-6 * np.eye(d)

        # --- Phase 3: Final (1+1)-ES Polish ---
        # Use the best point found by CEM as starting point for fine-grained search
        remaining = budget - calls
        if remaining <= 0:
            return best_x, best_f

        x_cur = best_x.copy()
        f_cur = best_f
        sigma = 0.5  # Start with smaller step for polish
        sigma_min = 1e-6
        sigma_max = 1.0
        no_improve_count = 0
        max_no_improve = 15

        while not check_deadline():
            y = clip(x_cur + sigma * rng.standard_normal(d))
            fy = float(func(y))
            calls += 1

            if fy < f_cur:
                x_cur = y.copy()
                f_cur = fy
                sigma *= 1.05
                no_improve_count = 0

                if f_cur < best_f:
                    best_f = f_cur
                    best_x = x_cur.copy()
            else:
                sigma *= 0.95
                no_improve_count += 1

            sigma = np.clip(sigma, sigma_min, sigma_max)

            if no_improve_count >= max_no_improve:
                # Try a small perturbation around best
                if remaining - (budget - calls) < 20:
                    break
                x_cur = clip(best_x + rng.standard_normal(d) * 0.3)
                f_cur = float(func(x_cur))
                calls += 1
                if f_cur < best_f:
                    best_f = f_cur
                    best_x = x_cur.copy()
                sigma = 0.3
                no_improve_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
