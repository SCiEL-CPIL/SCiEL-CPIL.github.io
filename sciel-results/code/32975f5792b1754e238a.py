"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Parallel portfolio of adaptive CMA-ES restarts for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill

        # Global best tracking
        best_x = np.zeros(dim)
        best_f = np.inf
        rng = np.random.default_rng(42)

        # Initialize CMA-ES parameters
        # lambda = population size, typically 4 + floor(3 * log(N))
        pop_size = 4 + int(3 * np.log(dim))
        if pop_size < 4:
            pop_size = 4
        if pop_size > 10:
            pop_size = 10
            
        # Initial sigma
        sigma0 = 2.0
        # CMA-ES parameters
        c_sigma = (pop_size + 2) / (dim + pop_size + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((pop_size / 2 + 1) - dim - 2) / ((dim + 2) ** 2 + pop_size))
        
        # Weights for mean update
        mu = pop_size // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Restarts strategy:
        # We will run multiple CMA-ES instances in parallel, each with a different starting point.
        # We allocate budget dynamically.
        
        # Phase 1: Global exploration with a few random starts to find basins
        # Let's use 3-5 restarts depending on budget
        n_restarts = 3
        if budget > 1000:
            n_restarts = 5
            
        # Initial points for restarts
        # Use a spread of points
        start_points = []
        # Origin
        start_points.append(np.zeros(dim))
        # Random points
        for i in range(1, n_restarts):
            start_points.append(rng.uniform(-3, 3, dim))
            
        # State for each restart
        # Each restart has: mean, sigma, C, ps, pc, calls_made, best_f_local, best_x_local
        restarts = []
        for sp in start_points:
            # Evaluate initial point
            f_sp = func(sp)
            if f_sp < best_f:
                best_f = f_sp
                best_x = sp.copy()
                
            # Initialize CMA-ES state
            mean = sp.copy()
            sigma = sigma0
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            calls = 1 # 1 for the initial point
            
            # Pre-generate some random numbers? No, generate on the fly.
            
            restarts.append({
                'mean': mean,
                'sigma': sigma,
                'C': C,
                'ps': ps,
                'pc': pc,
                'calls': calls,
                'best_f': f_sp,
                'best_x': sp.copy(),
                'active': True
            })

        # Main loop
        # We distribute calls to active restarts.
        # A simple strategy: Round-robin among active restarts.
        # If a restart converges (sigma too small or no improvement for long time), deactivate it.
        
        total_calls = sum(r['calls'] for r in restarts)
        
        while total_calls < budget:
            if time.time() > deadline:
                break
                
            # Select active restarts
            active_indices = [i for i, r in enumerate(restarts) if r['active']]
            if not active_indices:
                break
                
            # Pick one restart to update (round-robin or random)
            # Round-robin is deterministic and fair
            idx = active_indices[total_calls % len(active_indices)]
            r = restarts[idx]
            
            # Generate population
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(r['C'])
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset C
                r['C'] = np.eye(dim)
                r['sigma'] = sigma0
                r['ps'] = np.zeros(dim)
                r['pc'] = np.zeros(dim)
                L = np.eye(dim)
                
            ys = rng.standard_normal((pop_size, dim))
            # Transform ys
            # y = mean + sigma * L @ y
            # Note: L is lower triangular, so L @ y is efficient
            # But we need to apply L to each row of ys
            # ys_transformed = ys @ L.T ? 
            # C = L L^T. So x = mean + sigma * L @ y.
            # If y is row vector, x = mean + sigma * (L @ y.T).T = mean + sigma * y @ L.T
            
            # Optimization: Compute L.T once
            LT = L.T
            xs = r['mean'] + r['sigma'] * (ys @ LT)
            
            # Clip to box
            xs = np.clip(xs, -5, 5)
            
            # Evaluate population
            fs = np.array([func(x) for x in xs])
            r['calls'] += pop_size
            total_calls += pop_size
            
            # Update best
            min_idx = np.argmin(fs)
            if fs[min_idx] < r['best_f']:
                r['best_f'] = fs[min_idx]
                r['best_x'] = xs[min_idx].copy()
                if fs[min_idx] < best_f:
                    best_f = fs[min_idx]
                    best_x = xs[min_idx].copy()
                    
            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            
            # Update mean
            y_mean = np.sum(weights[:, np.newaxis] * xs_sorted[:mu], axis=0)
            new_mean = r['mean'] + y_mean
            
            # Update step size control variables
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * L^-1 * (new_mean - mean)
            # L^-1 * v = solve(L, v)
            delta_mean = new_mean - r['mean']
            # Solve L z = delta_mean / sigma
            # z = L^-1 @ (delta_mean / sigma)
            # Since L is lower triangular, use np.linalg.solve
            try:
                z = np.linalg.solve(L, delta_mean / r['sigma'])
            except np.linalg.LinAlgError:
                z = delta_mean / r['sigma']
                
            r['ps'] = (1 - c_sigma) * r['ps'] + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
            hsig = (np.linalg.norm(r['ps']) / np.sqrt(1 - (1 - c_sigma) ** (2 * (pop_size + 1)))) * np.sqrt(dim)
            
            # Update pc
            # pc = (1 - c_c) * pc + hsig * sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (new_mean - mean)
            hsig = 1 if np.linalg.norm(r['ps']) / np.sqrt(1 - (1 - c_sigma) ** (2 * (pop_size + 1))) < (1 + 1e-5) * np.sqrt(dim) else 0
            # Wait, the condition is usually: ||ps|| / E||N(0,I)|| < (1 + epsilon) * sqrt(dim)
            # E||N(0,I)|| = sqrt(dim) * (1 - 1/(4dim) + ...)
            # Let's use the standard formula
            phisig = np.sqrt(dim)
            hsig = 1 if np.linalg.norm(r['ps']) / phisig < (1 + 1e-5) * np.sqrt(1 - (1 - c_sigma) ** (2 * (pop_size + 1))) else 0
            
            r['pc'] = (1 - c_c) * r['pc'] + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_mean / r['sigma']
            
            # Update C
            # C = (1 - c_1 - c_mu + hsig * c_c * (2 - c_c)) * C + c_1 * pc pc^T + c_mu * sum wi (yi - mean)(yi - mean)^T
            # Note: yi - mean is the vector from old mean to new point? No, it's the step.
            # In CMA-ES, the update uses the steps from the *old* mean.
            # So we need to compute (xs_sorted[i] - r['mean'])
            
            # Term 1: pc pc^T
            term_pc = np.outer(r['pc'], r['pc'])
            
            # Term 2: sum wi (xi - mean)(xi - mean)^T
            # xs_sorted[:mu] are the best mu points
            # steps = xs_sorted[:mu] - r['mean']
            steps = xs_sorted[:mu] - r['mean']
            # Weighted outer product sum
            # term_mu = sum(weights[i] * outer(steps[i], steps[i]))
            # Efficient way: (steps.T * weights) @ steps
            term_mu = (steps.T * weights) @ steps
            
            # Coefficients
            c1_term = c_1
            cmu_term = c_mu
            decay = 1 - c_1 - c_mu + hsig * c_c * (2 - c_c)
            
            r['C'] = decay * r['C'] + c1_term * term_pc + cmu_term * term_mu
            
            # Update sigma
            r['sigma'] = r['sigma'] * np.exp(c_sigma / d_sigma * (np.linalg.norm(r['ps']) / phisig - 1))
            
            # Update mean
            r['mean'] = new_mean
            
            # Check for convergence/stagnation
            # 1. Sigma too small
            if r['sigma'] < 1e-10:
                r['active'] = False
            # 2. No improvement in last 100 calls (approx)
            # We can track last improvement call count
            # For simplicity, if sigma is very small or we've used a lot of calls, deactivate
            # A more robust check: if the range of fs is very small
            if fs_sorted[-1] - fs_sorted[0] < 1e-8 * max(1, abs(fs_sorted[0])):
                r['active'] = False
                
            # If all restarts are inactive, break
            if not any(r['active'] for r in restarts):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
