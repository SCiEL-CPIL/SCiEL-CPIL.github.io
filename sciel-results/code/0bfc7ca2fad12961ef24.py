"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with time guard for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        # We'll track the global best
        best_x = np.zeros(dim)
        best_f = np.inf

        # Multi-start: we'll do several restarts with different initial centers
        # For f22 (21 peaks), a good strategy is to explore broadly first,
        # then refine.

        # Strategy: 
        # 1. Initial broad sampling to find promising regions
        # 2. CMA-ES from the best initial point
        # 3. If time permits, restart from other promising points

        rng = np.random.default_rng(42)

        # Phase 1: Initial sampling - spread out to find good basins
        n_initial = min(50, budget // 10)
        if budget < 100:
            n_initial = max(5, budget // 10)

        points = []
        for i in range(n_initial):
            if time.time() > deadline:
                break
            # Use a mix of random and quasi-random (Sobol-like via Halton)
            if i == 0:
                x = np.zeros(dim)
            elif i < 5:
                # Corners and midpoints of the domain
                if i == 1:
                    x = np.full(dim, -4.0)
                elif i == 2:
                    x = np.full(dim, 4.0)
                elif i == 3:
                    x = rng.uniform(-4, 4, dim)
                else:
                    x = rng.uniform(-5, 5, dim)
            else:
                x = rng.uniform(-5, 5, dim)
            x = np.clip(x, -5, 5)
            fx = func(x)
            points.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort by fitness
        points.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES from the best point
        # CMA-ES parameters
        lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))  # population size
        sigma = 1.0  # initial step size
        
        # CMA-ES state
        x_mean = best_x.copy()
        f_mean = best_f
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cc = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        cs = (2 + 2 / (dim + 1)) / (dim + 2 + 2 / (dim + 1))
        c1 = (2 / ((dim + 1.3) ** 2 + lam)) * (1 - cc - cs / 2)
        cmu = min(c1, 2 * (1 - c1 - cc) / ((dim + 2) ** 2 - 1))

        calls_made = n_initial
        stagnation_counter = 0

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate lambda offspring
            try:
                A = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Reset covariance if it becomes invalid
                C = np.eye(dim)
                A = np.eye(dim)
                sigma = 1.0

            offspring = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (A @ z)
                # Sanitize to prevent NaN/Inf propagation
                y = np.nan_to_num(y, nan=0.0, posinf=5.0, neginf=-5.0)
                y = np.clip(y, -5, 5)
                try:
                    fy = func(y)
                except Exception:
                    fy = 1e10
                offspring.append((fy, y))
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(offspring) < lam:
                break

            # Sort offspring
            offspring.sort(key=lambda t: t[0])

            # Update mean
            weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights[:lam] ** 2)

            new_mean = np.zeros(dim)
            for i in range(lam):
                new_mean += weights[i] * offspring[i][1]
            
            # Update step size (CSA)
            hsig = np.sqrt((mu_eff + 2 - mu_eff / dim) / (dim + 2) - 1)
            ps = (1 - cs) * ps + cs * np.sign(hsig) * (A.T @ ((new_mean - x_mean) / sigma))
            sigma *= np.exp(cs / dim * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update covariance
            delta_sig = hsig * np.sqrt(1 - (1 - cs) ** (2 * (calls_made - n_initial) / (dim + 1)))
            pc = (1 - cc) * pc + cc * delta_sig * (new_mean - x_mean) / sigma
            rank_c = np.outer(pc, pc)
            rank_mu = np.zeros((dim, dim))
            for i in range(lam):
                diff = (offspring[i][1] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(diff, diff)
            C = (1 - c1 - cmu) * C + c1 * rank_c + cmu * rank_mu

            x_mean = new_mean
            f_mean = offspring[0][0]

            # Stagnation check
            if offspring[0][0] >= f_mean - 1e-10:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            # If stagnated for too long, do a restart from a different point
            if stagnation_counter >= 5 and calls_made < budget - 50:
                # Pick another promising point from initial sampling
                restarted = False
                if len(points) > 1:
                    # Try the next best untried point
                    for fx_p, x_p in points[1:]:
                        if np.linalg.norm(x_p - x_mean) > 0.5:
                            x_mean = x_p.copy()
                            sigma = 2.0
                            C = np.eye(dim)
                            pc = np.zeros(dim)
                            ps = np.zeros(dim)
                            stagnation_counter = 0
                            restarted = True
                            break
                
                if not restarted:
                    # Random restart
                    x_mean = rng.uniform(-4, 4, dim)
                    sigma = 2.0
                    C = np.eye(dim)
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                    stagnation_counter = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
