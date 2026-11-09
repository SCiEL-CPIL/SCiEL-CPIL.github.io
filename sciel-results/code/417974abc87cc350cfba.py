"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear-structure exploit for BBOB f5 (Linear slope).
    
    Strategy:
    1. Evaluate center and use large-step forward differences to estimate gradient signs.
    2. Jump directly to the predicted optimal vertex (and its opposite) for early progress.
    3. Refine with a lightweight (1+1)-ES to handle any non-linearity or boundary effects.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        
        d = self.dim
        budget = self.budget
        
        # Track best point and value
        best_x = None
        best_f = np.inf
        calls_made = 0

        def _eval(x):
            nonlocal best_x, best_f, calls_made
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Estimate gradient via large forward differences ---
        # Start from center (0,0,...,0)
        x0 = np.zeros(d)
        f0 = _eval(x0)
        
        # Use large steps (h=5) to probe near the boundaries for strong signal
        # This maximizes early progress by evaluating points close to potential optima
        step = 5.0
        grad = np.zeros(d)
        
        for i in range(d):
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Probe in the positive direction
            xp = np.zeros(d)
            xp[i] = step
            xp = np.clip(xp, -5, 5)
            fp = _eval(xp)
            
            # If the probe didn't improve and we have budget, check the negative direction
            # to disambiguate the gradient sign if the function is flat or noisy
            if fp >= f0 and calls_made < budget and time.time() < deadline:
                xm = np.zeros(d)
                xm[i] = -step
                xm = np.clip(xm, -5, 5)
                fm = _eval(xm)
                
                # Determine gradient sign based on which direction was better
                if fm < f0:
                    grad[i] = (f0 - fm) / step  # Negative gradient (min is in negative dir)
                elif fp < f0:
                    grad[i] = (fp - f0) / step  # Positive gradient (min is in positive dir)
                else:
                    # Both worse than center? Unlikely for linear, but keep 0
                    grad[i] = 0.0
            else:
                # fp < f0, so positive direction is better, gradient is effectively negative
                # Wait, if f(xp) < f0, then moving +x decreases f, so gradient is negative.
                # Let's be consistent: grad[i] approx (f(x+h) - f(x))/h
                # If f(x+h) < f(x), grad < 0.
                # The optimal bound is where x_i goes to -5 if grad > 0, else 5.
                # If grad < 0, we want x_i = 5.
                # If grad > 0, we want x_i = -5.
                grad[i] = (fp - f0) / step
        
        # --- Phase 2: Jump to predicted optimal vertex ---
        # For a linear function f(x) = c + g^T x, the minimum on [-5,5]^d
        # is at the vertex where each coordinate is at the bound opposite to the sign of g[i].
        x_pred = np.zeros(d)
        for i in range(d):
            if grad[i] > 0:
                x_pred[i] = -5.0
            else:
                x_pred[i] = 5.0
        
        if calls_made < budget and time.time() < deadline:
            _eval(x_pred)
            
        # Also evaluate the opposite vertex to be safe (in case of sign error or non-linearity)
        x_pred_alt = -x_pred
        if calls_made < budget and time.time() < deadline:
            _eval(x_pred_alt)

        # --- Phase 3: Refinement with (1+1)-ES ---
        # Since f5 is linear, the vertex should be optimal. 
        # However, if there's any non-linearity or if we missed the true optimum due to 
        # discretization or noise, a local search around the current best helps.
        
        x = best_x.copy()
        fx = best_f
        sigma = 0.1
        rng = np.random.default_rng(12345)
        
        # Run refinement until budget or time runs out
        while calls_made < budget and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = _eval(y)
            
            if fy < fx:
                x = y.copy()
                fx = fy
                # Slightly increase sigma to explore further if we found improvement
                sigma *= 1.1
            else:
                # Decrease sigma to converge
                sigma *= 0.9
                
            if sigma < 1e-8:
                break
        
        return x, best_f
# EVOLVE-BLOCK-END
