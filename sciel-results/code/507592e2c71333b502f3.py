"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Custom CMA-ES with Island Model and Error Repair for f22.
    
    Strategy:
    1. Use a population of 10 candidates (islands).
    2. Implement standard CMA-ES updates (mean, sigma, covariance C).
    3. Use 'error repair' (clipping/projection) for bounds [-5, 5].
    4. Run for the full budget, tracking global best.
    5. No external CMA library dependency to ensure speed and control.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Global best tracking
        x_global_best = np.random.uniform(-5, 5, dim)
        f_global_best = np.inf
        calls = 0

        def check_time():
            return (time.time() - start_time) < time_limit

        def evaluate(x):
            nonlocal calls, f_global_best, x_global_best
            # Error Repair: Clip to box
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < f_global_best:
                f_global_best = f
                x_global_best = x.copy()
            return f

        # Phase 1: Sparse Random Sampling to identify promising basins
        # f22 is multimodal. Start with a few random points to seed the islands.
        n_init = 20
        initial_points = []
        for _ in range(n_init):
            if not check_time() or calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = evaluate(x)
            initial_points.append((f, x))
        
        if not initial_points:
            return x_global_best, f_global_best

        # Sort initial points to use the best ones as seeds for islands
        initial_points.sort(key=lambda t: t[0])
        
        # Phase 2: Island-based CMA-ES
        # We will run a single main CMA-ES loop but restart it if stagnation is detected,
        # or simply run a robust CMA-ES that adapts well. 
        # Given the "Island Model" inspiration, let's implement a simple 2-island setup
        # where we split the budget or run sequentially with periodic migration.
        # However, for simplicity and robustness within 300s, a single well-tuned CMA-ES
        # with a larger population is often superior to complex island logic if the 
        # initial sampling is good. 
        # Let's stick to a single, high-quality CMA-ES instance starting from the best point,
        # but with a "restart" mechanism if no improvement for a long time.
        
        # Parameters for CMA-ES
        pop_size = 10 # As per directive: single population of 10 candidates
        # Actually, directive says "single population of 10 candidates" for AHA.
        # Let's implement the AHA-inspired adaptive step size with a population.
        
        # State variables for CMA-ES
        x_mean = initial_points[0][1].copy()
        sigma = 2.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Adaptation parameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        # Main Loop
        # We will use the remaining budget
        while calls < budget and check_time():
            # Generate population
            offspring = []
            for i in range(pop_size):
                if not check_time() or calls >= budget:
                    break
                z = rng.standard_normal(dim)
                # Apply C_sqrt
                y = C_sqrt @ z
                x_off = x_mean + sigma * y
                # Error Repair
                x_off = np.clip(x_off, -5.0, 5.0)
                f_off = func(x_off)
                calls += 1
                offspring.append((f_off, x_off, y))
                
            if not offspring:
                break

            # Sort offspring
            offspring.sort(key=lambda t: t[0])
            
            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * offspring[i][2]
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (C_sqrt @ y_w)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C
            if np.linalg.norm(p_sigma) < (1.4 + 1/dim) * chi_n:
                v = C_sqrt @ y_w
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * v
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(offspring[i][2], offspring[i][2])
                
                C = (1 + c_c * (2 - c_c) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                
                # Re-orthogonalize C_sqrt periodically to avoid drift
                # Simple approach: just update C_sqrt via eigendecomposition if needed
                # For dim=5, this is cheap.
                if calls % 10 == 0:
                    vals, vecs = np.linalg.eigh(C)
                    # Clip negative eigenvalues due to numerical error
                    vals = np.maximum(vals, 1e-12)
                    C_sqrt = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
                    # Ensure C is symmetric
                    C = (C + C.T) / 2
            
            x_mean = x_mean + sigma * (C_sqrt @ y_w)
            
            # Stagnation check / Restart logic
            # If we haven't improved global best in last 100 calls, restart from best
            # with reduced sigma
            if len(offspring) > 0 and offspring[0][0] < f_global_best:
                f_global_best = offspring[0][0]
                x_global_best = offspring[0][1].copy()
                # Reset stagnation counter implicitly by not restarting
            else:
                # Simple heuristic: if sigma becomes too small or too large, adjust
                if sigma < 1e-6:
                    # Restart from global best with fresh sigma
                    x_mean = x_global_best.copy()
                    sigma = 1.0
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)

        return x_global_best, f_global_best
# EVOLVE-BLOCK-END
