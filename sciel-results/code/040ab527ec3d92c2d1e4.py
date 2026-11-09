# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer combining Structured Initialization, Adaptive (1+1)-ES, 
    SPSA-based escape mechanisms, and Nelder-Mead polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_exceeded():
            return time.time() > deadline

        # ---------------------------------------------------------
        # Phase 1: Structured Initialization
        # ---------------------------------------------------------
        
        # 1. Halton Sequence for uniform coverage
        def halton(idx, base):
            q = 1.0
            r = 0.0
            while idx > 0:
                q /= base
                r += q * (idx % base)
                idx //= base
            return r

        primes = [2, 3, 5, 7, 11]
        n_halton = 20
        for i in range(1, n_halton + 1):
            pt = np.zeros(dim)
            for j in range(dim):
                pt[j] = halton(i, primes[j])
            pt = pt * 10.0 - 5.0
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            update_best(pt, f)
            if calls >= budget or time_exceeded():
                return best_x, best_f

        # 2. Strategic Points (Origin, +/- 1, +/- 2, etc.)
        strategic_points = [
            np.zeros(dim),
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
            np.ones(dim) * 4.0,
            np.ones(dim) * -4.0,
        ]
        for pt in strategic_points:
            if calls >= budget or time_exceeded():
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            update_best(pt, f)

        # 3. Integer Lattice Points (Targeting Rastrigin minima)
        # Sample random integer vectors in [-4, 4]
        n_lattice = 15
        for _ in range(n_lattice):
            if calls >= budget or time_exceeded():
                break
            pt = np.array([rng.integers(-4, 5) for _ in range(dim)], dtype=float)
            f = func(pt)
            calls += 1
            update_best(pt, f)

        # 4. Random Uniform Points
        n_random = 10
        for _ in range(n_random):
            if calls >= budget or time_exceeded():
                break
            pt = rng.uniform(-5, 5, dim)
            f = func(pt)
            calls += 1
            update_best(pt, f)

        # ---------------------------------------------------------
        # Phase 2: Adaptive (1+1)-ES with SPSA Escape
        # ---------------------------------------------------------
        
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        sigma_min = 0.005
        sigma_max = 4.0
        
        success_window = []
        window_size = 15
        stagnation = 0
        no_improve_count = 0
        
        # SPSA parameters
        spsa_a = 0.1  # Step size coefficient
        spsa_c = 0.1  # Perturbation coefficient
        spsa_iters = 5 # Number of SPSA steps to try on stagnation

        polish_start = int(budget * 0.85)

        while calls < budget:
            if time_exceeded():
                break

            in_polish = calls >= polish_start
            if in_polish:
                sigma = min(sigma, 0.1)

            # Generate offspring using (1+1)-ES
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            success = fy < fx
            success_window.append(success)
            if len(success_window) > window_size:
                success_window.pop(0)

            # 1/5th rule
            if len(success_window) >= window_size:
                success_rate = sum(success_window) / len(success_window)
                if success_rate > 0.3:
                    sigma = min(sigma * 1.2, sigma_max)
                elif success_rate < 0.1:
                    sigma = max(sigma * 0.7, sigma_min)

            if success:
                x, fx = y, fy
                stagnation = 0
            else:
                stagnation += 1

            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Stagnation handling
            progress_ratio = calls / budget
            stagnation_limit = int(30 * (1.0 - 0.5 * progress_ratio))
            
            if in_polish:
                stagnation_limit = 15

            if stagnation >= stagnation_limit:
                # Try SPSA escape
                spsa_success = False
                spsa_x = x.copy()
                spsa_f = fx
                
                # SPSA loop
                for k in range(spsa_iters):
                    if calls >= budget or time_exceeded():
                        break
                    
                    # Generate random direction delta in {-1, 1}^dim
                    delta = rng.choice([-1, 1], size=dim)
                    
                    # Perturbed points
                    x_plus = np.clip(spsa_x + spsa_c * delta, -5, 5)
                    x_minus = np.clip(spsa_x - spsa_c * delta, -5, 5)
                    
                    f_plus = func(x_plus)
                    calls += 1
                    f_minus = func(x_minus)
                    calls += 1
                    
                    # Gradient estimate
                    grad_est = (f_plus - f_minus) / (2 * spsa_c) * delta
                    
                    # Update
                    a_k = spsa_a / (k + 1) # Simple decay
                    spsa_x = np.clip(spsa_x - a_k * grad_est, -5, 5)
                    
                    # Evaluate new point? 
                    # SPSA usually evaluates the new point in the next iteration's perturbation 
                    # or we can evaluate it now to update best.
                    # To save calls, we assume the update is good and evaluate in next loop?
                    # No, we need to know if it improved to update best_x.
                    # Let's evaluate spsa_x.
                    f_new = func(spsa_x)
                    calls += 1
                    
                    if f_new < spsa_f:
                        spsa_f = f_new
                        spsa_success = True
                    else:
                        # If it didn't improve, maybe revert or keep? 
                        # SPSA is stochastic, so we keep moving.
                        pass

                if spsa_success and spsa_f < fx:
                    x = spsa_x
                    fx = spsa_f
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        no_improve_count = 0
                    stagnation = 0
                    sigma = 1.0 # Reset sigma
                else:
                    # SPSA failed, do a restart
                    if rng.random() < 0.5:
                        x = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
                    else:
                        x = rng.uniform(-5, 5, dim)
                    fx = func(x)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        no_improve_count = 0
                    sigma = 1.5
                    stagnation = 0
                    success_window = []

            if calls >= budget:
                break

        # ---------------------------------------------------------
        # Phase 3: Nelder-Mead Polish
        # ---------------------------------------------------------
        if calls < budget and time.time() < deadline:
            nm_budget = min(budget - calls, 50)
            if nm_budget >= dim + 2:
                best_x, best_f = self._nelder_mead(func, best_x, best_f, dim, nm_budget, deadline)

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, dim, budget, deadline):
        """Nelder-Mead simplex method for final polish."""
        simplex = [x0.copy()]
        fvals = [f0]
        calls = 1

        step = 0.1
        for i in range(dim):
            xi = x0.copy()
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            calls += 1
            simplex.append(xi)
            fvals.append(fi)
            if fi < f0:
                f0 = fi
                x0 = xi.copy()

        if calls >= budget:
            return x0, f0

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5

        while calls < budget and time.time() < deadline:
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)

            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
            fr = func(xr)
            calls += 1

            if fr < fvals[0]:
                xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                    if fe < f0:
                        f0 = fe
                        x0 = xe.copy()
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    if fr < f0:
                        f0 = fr
                        x0 = xr.copy()
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
                if fr < f0:
                    f0 = fr
                    x0 = xr.copy()
            else:
                if fr < fvals[-1]:
                    xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                else:
                    xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                fc = func(xc)
                calls += 1

                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                    if fc < f0:
                        f0 = fc
                        x0 = xc.copy()
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = np.clip(x0 + sigma_nm * (simplex[i] - x0), -5, 5)
                        fi = func(simplex[i])
                        calls += 1
                        fvals[i] = fi
                        if fi < f0:
                            f0 = fi
                            x0 = simplex[i].copy()

            if calls >= budget:
                break

        return x0, f0
# EVOLVE-BLOCK-END
