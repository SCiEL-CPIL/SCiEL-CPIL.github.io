"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid (1+1)-ES with CMA-ES-like covariance adaptation for BBOB f19.
    Strategy:
    1. Exploration phase (10% budget): Random sampling to find a good starting basin.
    2. Exploitation phase (90% budget): 
       - 90% of this phase: (1+1)-ES with CMA-ES covariance (rank-1, no eigendecomp) for global refinement.
       - 10% of this phase: Tight (1+1)-ES for final convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, keep 30s safety margin
        deadline = start_time + 270.0
        
        dim = self.dim
        budget = self.budget
        
        global_best_x = None
        global_best_f = float('inf')
        calls_made = 0
        
        def call_func(x):
            nonlocal calls_made, global_best_x, global_best_f
            if calls_made >= budget:
                return global_best_f
            val = func(x)
            calls_made += 1
            if val < global_best_f:
                global_best_f = val
                global_best_x = x.copy()
            return val

        # Phase 1: Exploration (10% of budget)
        # Goal: Find a point closer to the optimum than random center.
        n_explore = max(10, int(0.1 * budget))
        
        # Start with center (0,0,0,0,0) as it's often good for Griewank-like
        x_center = np.zeros(dim)
        f_center = call_func(x_center)
        
        # Random sampling
        for _ in range(n_explore - 1):
            if time.time() > deadline or calls_made >= budget:
                break
            x_rand = np.random.uniform(-5, 5, dim)
            call_func(x_rand)
        
        if global_best_x is None:
            global_best_x = x_center
            global_best_f = f_center

        # Phase 2: Exploitation (90% of budget)
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return global_best_x, global_best_f
            
        # Split remaining into CMA-ES (90%) and Polish (10%)
        n_cma = int(0.9 * remaining_budget)
        n_polish = remaining_budget - n_cma
        
        # --- CMA-ES-like (1+1)-ES ---
        # Initialize
        x_mean = global_best_x.copy()
        f_mean = global_best_f
        
        # Covariance matrix C (identity initially)
        C = np.eye(dim)
        # Step size sigma
        # Initial sigma: 0.2 * range is reasonable. Range is 10.
        sigma = 0.2 * 10.0 
        # Learning rates for CMA-ES
        mu = 1
        # c_sigma, c_1, c_mu
        c_sigma = 1.0 / (dim + 4)
        c_1 = (2 / ((dim + 1.3)**2 + 10))
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/(dim+2)) / ((dim+2)**2 + mu_eff))
        # mu_eff for mu=1 is 1
        mu_eff = 1.0
        # Path lengths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # We need to adapt sigma. Standard CMA-ES uses a path p_sigma.
        # For (1+1), we simplify:
        # If we accept, we might increase sigma slightly, if reject decrease?
        # Or use the standard CMA-ES sigma update based on p_sigma.
        
        # Let's use a simpler adaptive sigma:
        # If accepted, sigma *= 1.01 (if improving fast?) 
        # Actually, let's stick to the CMA-ES sigma adaptation rule:
        # sigma = sigma * exp(c_sigma/chi_N * (||p_sigma||/chi_N - 1))
        # chi_N = E||N(0,I)|| = sqrt(dim)*(1 - 1/(4dim) + 1/(21dim^2))
        chi_N = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim*dim))
        
        # Loop for CMA-ES phase
        # Each iteration costs 1 evaluation (the proposed point).
        # We know f(x_mean).
        for i in range(n_cma):
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Sample direction
            # z ~ N(0, I)
            z = np.random.standard_normal(dim)
            # Transform z using C
            # x = x_mean + sigma * C @ z
            # For (1+1), we don't need full C if we just want a good direction, 
            # but C helps align with the valley.
            # To avoid matrix sqrt or eigen, we can use C directly if it's well-conditioned.
            # However, C updates can make it ill-conditioned.
            # Simpler: Use C to generate a direction, but ensure stability.
            # x_new = x_mean + sigma * (C @ z)
            # Note: This assumes C is a covariance matrix.
            
            # Generate x_new
            x_new = x_mean + sigma * (C @ z)
            
            # Clip to box
            x_new = np.clip(x_new, -5, 5)
            
            f_new = call_func(x_new)
            
            # Update CMA-ES parameters
            # Path update
            # h_sigma = 1 if ||p_sigma|| < (1.4 + 2/(dim+1)) * chi_N
            # For simplicity, we'll use a fixed h_sigma=1.0 or similar.
            # p_sigma = (1 - c_sigma) * p_sigma + h_sigma * sqrt(c_sigma * (2 - c_sigma) * mu) * (C^-1/2 * (x_new - x_mean)/sigma)
            # This requires C^-1/2. That's expensive.
            
            # Alternative: Since we are in (1+1), we can use a simpler heuristic for sigma and C.
            # If f_new < f_mean:
            #   Accept
            #   Update C: C = (1 - c_1) * C + c_1 * (x_new - x_mean) @ (x_new - x_mean).T / sigma^2 ?
            #   This is a rank-1 update.
            #   Update sigma: if accepted, keep or increase?
            #   If rejected, decrease sigma?
            
            # Let's implement a simplified adaptive (1+1)-ES:
            # 1. If f_new < f_mean:
            #      Accept.
            #      x_mean = x_new
            #      f_mean = f_new
            #      # Update C towards the direction of the step
            #      step = (x_new - x_mean_old) / sigma
            #      C = (1 - c_1) * C + c_1 * np.outer(step, step)
            #      # Normalize C to prevent blowup?
            #      # Sigma adaptation:
            #      # If we improved significantly, maybe increase sigma?
            #      # If we improved slightly, keep?
            #      # Standard (1+1)-ES often uses sigma *= 1.0 if accepted, 0.8 if rejected?
            #      # Or use the CMA-ES path.
            #      
            #      # Let's try:
            #      # sigma *= 1.01 (slow increase)
            #      # But if we are stuck, we need to decrease.
            #      
            # 2. If f_new >= f_mean:
            #      Reject.
            #      sigma *= 0.9 (decrease step size)
            #      # Maybe update C slightly towards the failed direction? No, that's bad.
            #      # Keep C.
            
            # This simple heuristic is robust but might not exploit the valley shape as well as full CMA-ES.
            # Given the "inspiration" that (1+1)-ES improved score, let's stick to a robust (1+1)-ES.
            # We will use a "success rate" based sigma adaptation.
            # Target success rate: 0.2 (1/5) is often good for (1+1)-ES.
            
            # Let's maintain a history of last 10 acceptances.
            # If success rate > 0.2, increase sigma. If < 0.2, decrease.
            
            # For now, let's use the simple rule:
            # Accept -> sigma *= 1.0
            # Reject -> sigma *= 0.9
            # And update C on acceptance to align with the successful step.
            
            if f_new < f_mean:
                # Accept
                step = (x_new - x_mean) / sigma
                # Update C (Rank-1)
                # C = (1 - c_1) * C + c_1 * outer(step, step)
                C = (1 - c_1) * C + c_1 * np.outer(step, step)
                # Prevent C from becoming too large or small
                # Normalize trace to dim?
                # trace(C) should be roughly dim.
                trace_C = np.trace(C)
                if trace_C > 0:
                    C = C * (dim / trace_C)
                
                x_mean = x_new
                f_mean = f_new
                # Sigma adaptation
                # If we are making progress, we might want to keep sigma or increase slightly.
                # But if we are in a valley, we want to align.
                # Let's try: sigma *= 1.0
                # Or maybe increase if the step was large and successful?
                # Let's keep sigma constant on success for stability.
                # sigma *= 1.0
            else:
                # Reject
                # Decrease sigma
                sigma *= 0.9
                # Ensure sigma doesn't get too small
                sigma = max(sigma, 1e-4)
            
            # Check time
            if time.time() > deadline:
                break

        # Phase 3: Polish (10% of remaining, or 10% of total if not used)
        # Use a very tight (1+1)-ES to converge to the minimum.
        if n_polish > 0 and time.time() < deadline and calls_made < budget:
            x_mean = global_best_x.copy()
            f_mean = global_best_f
            sigma = 0.01 * 10.0 # Small initial step
            C = np.eye(dim)
            c_1 = 0.05 # Faster adaptation for polish? Or slower?
            # Let's use a simple random walk with decreasing sigma.
            
            for i in range(n_polish):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                z = np.random.standard_normal(dim)
                x_new = x_mean + sigma * z
                x_new = np.clip(x_new, -5, 5)
                
                f_new = call_func(x_new)
                
                if f_new < f_mean:
                    x_mean = x_new
                    f_mean = f_new
                    # sigma *= 1.0
                else:
                    sigma *= 0.95
                    sigma = max(sigma, 1e-6)
                    
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
