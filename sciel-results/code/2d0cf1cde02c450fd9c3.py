"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid SA + (1+1)-ES for BBOB-style Schwefel on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # Phase 1: Simulated Annealing for global exploration
        # Phase 2: (1+1)-ES with 1/5 success rule for local refinement
        
        def time_exceeded():
            return time.time() > deadline
        
        # Start from origin
        x = np.zeros(dim)
        try:
            f = func(x)
        except Exception:
            f = np.inf
        
        best_x = x.copy()
        best_f = f
        evals = 1
        
        # SA parameters
        T0 = 5.0  # Initial temperature
        alpha = 0.995  # Cooling rate
        sa_budget = min(100, budget // 5)  # Use up to 100 evals for SA
        T = T0
        
        # Phase 1: Simulated Annealing
        for _ in range(sa_budget):
            if evals >= budget or time_exceeded():
                break
            
            # Generate candidate with large step
            step_size = T * 0.8
            direction = rng.standard_normal(dim)
            candidate = x + step_size * direction
            np.clip(candidate, -5, 5, out=candidate)
            
            try:
                f_candidate = func(candidate)
            except Exception:
                f_candidate = np.inf
            
            evals += 1
            
            # Accept if better, or with probability exp(-delta/T)
            if f_candidate < f:
                x = candidate
                f = f_candidate
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            else:
                delta = f_candidate - f
                if delta > 0 and T > 1e-10:
                    accept_prob = np.exp(-delta / T)
                    if rng.random() < accept_prob:
                        x = candidate
                        f = f_candidate
            
            T *= alpha
        
        # Phase 2: (1+1)-ES with 1/5 success rule
        sigma = 1.5  # Reduced step size for local search
        
        successes = 0
        total_attempts = 0
        
        while evals < budget and not time_exceeded():
            # Generate candidate
            direction = rng.standard_normal(dim)
            candidate = x + sigma * direction
            np.clip(candidate, -5, 5, out=candidate)
            
            try:
                f_candidate = func(candidate)
            except Exception:
                f_candidate = np.inf
            
            evals += 1
            total_attempts += 1
            
            if f_candidate < f:
                x = candidate
                f = f_candidate
                successes += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            # Update sigma using 1/5 success rule
            success_rate = successes / max(1, total_attempts)
            if success_rate > 0.2:
                sigma *= 1.05
            elif success_rate < 0.2:
                sigma *= 0.95
            
            # Reset success counter every 10 attempts for responsiveness
            if total_attempts % 10 == 0:
                successes = 0
                total_attempts = 0
            
            # If sigma becomes too small, we've likely converged
            if sigma < 1e-10:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
