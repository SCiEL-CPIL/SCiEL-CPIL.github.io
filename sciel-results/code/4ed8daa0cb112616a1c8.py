"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using a hybrid approach:
    1. Simulated Annealing for global exploration with probabilistic acceptance.
    2. Adaptive Coordinate Descent for rapid local refinement.
    
    Key structural change from CMA-ES-like:
    - Uses Simulated Annealing (SA) to escape local minima in rotated Schwefel instances.
    - Geometrically cooling temperature schedule allows large initial steps to explore the landscape.
    - Switches to adaptive coordinate descent once the temperature is low enough for precise convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        current_x = np.zeros(dim)
        current_f = np.inf

        # Helper to evaluate function with safety checks
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() >= deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                fx = func(x)
            except Exception:
                fx = np.inf
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Simulated Annealing
        # Initial temperature high enough to accept large jumps
        T = 10.0
        T_min = 1e-6
        cooling_rate = 0.995  # Geometric cooling
        step_scale = 1.5      # Initial step size scale
        
        # Initialize with origin
        current_f = evaluate(current_x)
        best_f = current_f
        best_x = current_x.copy()
        
        # SA Loop
        while calls_made < budget and time.time() < deadline:
            if T < T_min:
                break
            
            # Generate candidate
            # Step size scales with temperature to allow large moves early
            move = rng.normal(0, step_scale * np.sqrt(T / 10.0 + 0.1), dim)
            x_trial = current_x + move
            x_trial = np.clip(x_trial, -5.0, 5.0)
            
            f_trial = evaluate(x_trial)
            
            if f_trial < current_f:
                # Always accept improvement
                current_x = x_trial
                current_f = f_trial
            else:
                # Accept worse solution with probability exp(-delta/T)
                delta = f_trial - current_f
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        current_x = x_trial
                        current_f = f_trial
                    # else reject, keep current
            
            # Cool down
            T *= cooling_rate
            
            # Dynamically adjust step scale to keep exploration effective
            # If T is high, steps are large. If T is low, steps shrink.
            # The sqrt(T/10 + 0.1) term above handles some of this.
            
        # Phase 2: Adaptive Coordinate Descent for Polish
        # Start from the best point found by SA
        if calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            # Adaptive steps per dimension
            steps = np.ones(dim) * 0.5
            no_improve_count = 0
            max_no_improve = 5
            
            while calls_made < budget and time.time() < deadline:
                improved = False
                
                for d in range(dim):
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    
                    # Try positive step
                    x_trial = x.copy()
                    x_trial[d] = np.clip(x_trial[d] + steps[d], -5.0, 5.0)
                    f_trial = evaluate(x_trial)
                    
                    if f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        steps[d] *= 1.2  # Increase step on success
                        improved = True
                        no_improve_count = 0
                    else:
                        # Try negative step
                        x_trial_neg = x.copy()
                        x_trial_neg[d] = np.clip(x_trial_neg[d] - steps[d], -5.0, 5.0)
                        f_trial_neg = evaluate(x_trial_neg)
                        
                        if f_trial_neg < fx:
                            x = x_trial_neg
                            fx = f_trial_neg
                            steps[d] *= 1.2
                            improved = True
                            no_improve_count = 0
                        else:
                            # Both failed, shrink step
                            steps[d] *= 0.5
                            if steps[d] < 1e-4:
                                steps[d] = 1e-4
                    
                    if calls_made >= budget or time.time() >= deadline:
                        break
                
                if not improved:
                    no_improve_count += 1
                    if no_improve_count >= max_no_improve:
                        # Try a small random perturbation to escape potential plateaus
                        if calls_made < budget and time.time() < deadline:
                            x = np.clip(x + rng.normal(0, 0.1, dim), -5.0, 5.0)
                            fx = evaluate(x)
                            no_improve_count = 0
                        else:
                            break
                else:
                    no_improve_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
