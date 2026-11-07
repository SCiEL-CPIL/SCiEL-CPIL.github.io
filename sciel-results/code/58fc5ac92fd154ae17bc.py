"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Adaptive covariance-lite optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned).
    
    Strategy:
    - Phase 1: Small random sampling to find initial good point
    - Phase 2: Adaptive search with:
      * Coordinate-wise step adaptation (like (1+1)-ES)
      * Periodic covariance estimation from recent best points
      * Stagnation detection with random restarts
      * Dynamic step size scaling based on success rate
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        n = self.dim
        budget = self.budget

        if budget < 10:
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)
            for _ in range(budget - 1):
                x = rng.uniform(-5, 5, n)
                f = func(x)
                if f < best_f:
                    best_f, best_x = f, x
            return best_x, best_f

        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            if time.time() > deadline:
                return None
            calls_made += 1
            return func(x)

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Phase 1: Small random sampling (10-15 calls)
        n_random = max(10, min(15, budget // 10))
        for _ in range(n_random):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            f = safe_call(x)
            if f is None:
                break
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = safe_call(best_x)
            if best_f is None:
                return best_x, np.inf

        # Phase 2: Adaptive covariance-lite search
        # Maintain a history of recent best points for covariance estimation
        recent_points = [(best_x.copy(), best_f)]
        max_history = 20
        
        # Coordinate-wise step sizes
        sigma = np.ones(n) * 0.5
        sigma_min = 1e-5
        sigma_max = 5.0
        
        # Stagnation tracking
        last_improvement_call = calls_made
        stagnation_threshold = 30
        
        # Success/failure tracking for adaptive step scaling
        success_count = 0
        fail_count = 0
        window_size = 10
        
        while calls_made < budget and time.time() < deadline:
            # Check for stagnation
            if calls_made - last_improvement_call > stagnation_threshold:
                # Try random restarts near best point
                for _ in range(3):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    x_trial = best_x + 2.0 * rng.standard_normal(n)
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = safe_call(x_trial)
                    if f_trial is None:
                        break
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial.copy()
                        last_improvement_call = calls_made
                        success_count += 1
                        fail_count = 0
                        recent_points.append((best_x.copy(), best_f))
                        if len(recent_points) > max_history:
                            recent_points.pop(0)
                        break
                if calls_made - last_improvement_call > stagnation_threshold:
                    # Still stuck, try larger jumps
                    for _ in range(2):
                        if calls_made >= budget or time.time() > deadline:
                            break
                        x_trial = rng.uniform(-5, 5, n)
                        f_trial = safe_call(x_trial)
                        if f_trial is None:
                            break
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = x_trial.copy()
                            last_improvement_call = calls_made
                            success_count += 1
                            fail_count = 0
                            recent_points.append((best_x.copy(), best_f))
                            if len(recent_points) > max_history:
                                recent_points.pop(0)
                            break
                    # Reset stagnation counter
                    last_improvement_call = calls_made
                    # Reset step sizes
                    sigma = np.ones(n) * 1.0
            
            # Periodic covariance estimation (every 15 calls)
            if len(recent_points) >= 5 and calls_made % 15 == 0:
                points = np.array([p[0] for p in recent_points[-5:]])
                # Estimate covariance from recent points
                if len(points) >= 2:
                    cov = np.cov(points.T)
                    if np.all(np.isfinite(cov)):
                        # Use diagonal of covariance to adapt sigma
                        diag = np.diag(cov)
                        diag = np.clip(diag, 1e-10, 100)
                        sigma = np.sqrt(diag) * 0.5
                        sigma = np.clip(sigma, sigma_min, sigma_max)
            
            # Generate trial point using adaptive sigma
            z = rng.standard_normal(n)
            x_trial = best_x + sigma * z
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = safe_call(x_trial)
            if f_trial is None:
                break
            
            if f_trial < best_f:
                # Success
                best_f = f_trial
                best_x = x_trial.copy()
                last_improvement_call = calls_made
                success_count += 1
                fail_count = 0
                
                # Increase sigma slightly on success
                sigma *= 1.03
                sigma = np.clip(sigma, sigma_min, sigma_max)
                
                # Add to history
                recent_points.append((best_x.copy(), best_f))
                if len(recent_points) > max_history:
                    recent_points.pop(0)
            else:
                # Failure
                fail_count += 1
                success_count = 0
                
                # Decrease sigma on failure
                sigma *= 0.95
                sigma = np.clip(sigma, sigma_min, sigma_max)
            
            # Adaptive step scaling based on success rate in window
            window_total = success_count + fail_count
            if window_total >= window_size:
                success_rate = success_count / window_total
                if success_rate > 0.7:
                    # Too easy, increase exploration
                    sigma *= 1.1
                elif success_rate < 0.3:
                    # Too hard, decrease exploitation
                    sigma *= 0.9
                sigma = np.clip(sigma, sigma_min, sigma_max)
                # Reset counters
                success_count = 0
                fail_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
