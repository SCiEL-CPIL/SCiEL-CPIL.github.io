import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Time guard: 330s limit, use 320s as safety margin
        start_time = time.time()
        time_limit = 320.0

        # CMA-ES parameters
        # Increased population size to 12 for better diversity in multimodal landscapes (f15)
        # Standard formula is 4 + 3*log(dim) which is ~10 for dim=5.
        # Finding suggests 12 is beneficial.
        lam = 12
        mu = lam // 2  # number of parents

        # Initial population
        x0 = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Weights
        w = np.zeros(lam)
        w[:mu] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        # CMA-ES parameters
        # Using standard Hansen recommendations
        cs = (1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1)) / ((dim + 2) + mueff)
        cc = (4 + mueff / dim) / ((dim + 2) + 2 * mueff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        dsigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + cs

        # Cholesky of C
        cholC = np.linalg.cholesky(C)

        # Initial population evaluation
        best_x = None
        best_f = np.inf
        calls = 0

        # Generate initial population
        pop = []
        for i in range(lam):
            if calls >= budget or time.time() - start_time > time_limit:
                break
            xi = x0 + sigma * (cholC @ rng.standard_normal(dim))
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            calls += 1
            pop.append((fi, xi))
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
        
        if not pop:
            return np.zeros(dim), np.inf

        # Sort by fitness
        pop.sort(key=lambda t: t[0])

        # CMA-ES loop
        while calls < budget:
            # Check time
            if time.time() - start_time > time_limit:
                break

            # Update mean
            xmean_old = x0.copy()
            x0 = np.sum([w[i] * pop[i][1] for i in range(mu)], axis=0)

            # Update step size control variables
            # ps update
            # Note: The vector (x0 - xmean_old) / sigma is the step in the standard normal space
            # transformed by C. For ps, we need the step in the "C-space" normalized by sigma?
            # Standard CMA-ES: ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * C^{-1/2} (x0 - xmean_old) / sigma
            # Since we sample x = x0 + sigma C^{1/2} z, the step is sigma C^{1/2} z.
            # The update for ps typically uses the inverse transformation of the step.
            # However, many implementations simplify this. Let's use the robust version from inspiration 1.
            # Inspiration 1 uses: ps = (1-cs)ps + sqrt(...) * (cholC.T @ (x0 - xmean_old)) / sigma
            # This is effectively C^{-1/2} (x0 - xmean_old) / sigma.
            
            try:
                step_in_c_space = (cholC.T @ (x0 - xmean_old)) / sigma
            except:
                step_in_c_space = (x0 - xmean_old) / sigma

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * step_in_c_space
            
            # hsig check
            # The threshold is (1.4 + 2/(dim+1)) * chiN
            # chiN is the expected norm of a standard normal vector in dim dimensions
            chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            # The decay factor for the path length
            # We use a simplified check based on the expected norm
            hsig = 1.0 if np.linalg.norm(ps) < (1.4 + 2.0 / (dim + 1.0)) * chiN else 0.0

            # pc update
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x0 - xmean_old) / sigma

            # Update covariance matrix
            rank1 = np.outer(pc, pc)
            delta_h = (1.0 if hsig else 0.0) * (2 - cc)
            rankmu = np.zeros((dim, dim))
            for i in range(mu):
                zi = (pop[i][1] - xmean_old) / sigma
                rankmu += w[i] * np.outer(zi, zi)
            
            # The update formula:
            # C = (1 + c1*delta_h - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
            C = (1 + c1 * delta_h - c1 - cmu) * C + c1 * rank1 + cmu * rankmu

            # Symmetrize and ensure positive definite
            C = (C + C.T) / 2.0
            # Add small regularization if needed
            try:
                cholC = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-4 * np.eye(dim)
                C = (C + C.T) / 2.0
                cholC = np.linalg.cholesky(C)

            # Update sigma
            # The norm of ps is compared to chiN
            sigma *= np.exp(cs / dsigma * (np.linalg.norm(ps) / chiN - 1))

            # Generate new population
            new_pop = []
            for i in range(lam):
                if calls >= budget:
                    break
                # Check time
                if time.time() - start_time > time_limit:
                    break
                xi = x0 + sigma * (cholC @ rng.standard_normal(dim))
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                new_pop.append((fi, xi))
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            if calls >= budget or time.time() - start_time > time_limit:
                break

            # Sort by fitness
            new_pop.sort(key=lambda t: t[0])
            pop = new_pop

        if best_x is None:
            return np.zeros(dim), np.inf
        return best_x, best_f
