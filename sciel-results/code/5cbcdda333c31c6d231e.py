"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start Simulated Annealing for BBOB Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 300s limit
        
        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # Simulated Annealing Parameters
        # We use a multi-start strategy. 
        # Phase 1: High temperature, large steps to explore the [-5, 5]^5 space.
        # Phase 2: Cooling, smaller steps to refine.
        
        # Since we don't know the scale of the function, we assume a typical 
        # BBOB scale where f ~ ||x - x_opt||^2.
        # Initial temperature should be large enough to accept moves that increase f.
        # If f ~ 100, T=100 allows moves up to ~100 increase.
        
        T_initial = 50.0
        T_final = 1e-6
        alpha = 0.95  # Cooling rate per step (geometric cooling)
        
        # We will run multiple restarts if time permits, but a single long 
        # annealing run with a good initial T often suffices for Sphere.
        # However, for robustness against bad starts, we do a few restarts 
        # if we have budget left.
        
        best_f = float('inf')
        best_x = np.zeros(dim)
        calls = 0
        
        # We allocate budget to a primary run, and keep a reserve for restarts.
        # Let's use 80% of budget for the main run.
        main_budget = int(budget * 0.8)
        
        # Main Annealing Run
        T = T_initial
        # Start at 0, which is the center of the box. 
        # The minimizer is in [-4, 4]^5.
        x = np.zeros(dim)
        fx = func(x)
        calls += 1
        
        if fx < best_f:
            best_f = fx
            best_x = x.copy()
            
        # Step size initial value. 
        # In 5D, a step of 1.0 is a reasonable start.
        step_size = 2.0 
        
        for i in range(main_budget - 1):
            if calls >= budget or time.time() > deadline:
                break
                
            # Generate a random neighbor
            # Using a Gaussian perturbation scaled by step_size
            dx = rng.standard_normal(dim) * step_size
            x_new = x + dx
            x_new = np.clip(x_new, -5, 5)
            
            fx_new = func(x_new)
            calls += 1
            
            df = fx_new - fx
            
            # Metropolis criterion
            if df < 0:
                # Accept if better
                x = x_new
                fx = fx_new
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Accept if worse with probability exp(-df/T)
                # To avoid overflow/underflow issues, we can compare log(random) < -df/T
                # But for simplicity and speed, direct exp is fine for moderate T.
                # If T is very small, exp(-df/T) is 0.
                if T > 1e-10:
                    prob = np.exp(-df / T)
                    if rng.random() < prob:
                        x = x_new
                        fx = fx_new
                        # We don't update best_f here because df >= 0, 
                        # so fx_new >= fx. But if fx was not best, this won't help best_f.
                        # Actually, if fx was not best, and we move to a worse point, 
                        # we definitely don't improve best_f.
                        # If fx WAS best, and we move to a worse point, we don't improve best_f.
                        # So best_f only updates on df < 0.
            
            # Cool down
            T *= alpha
            # Slowly reduce step size as well to help convergence
            # step_size *= 0.999? No, T handles the acceptance. 
            # But large steps late in the game are wasted calls.
            # Let's scale step size with T.
            step_size = 2.0 * (T / T_initial + 0.1)
            # Ensure step size doesn't get too small
            if step_size < 0.01:
                step_size = 0.01

        # If we have budget left, we can do a local polish or restart.
        # For Sphere, a simple local gradient-free search (like Nelder-Mead or 
        # just small random steps) around best_x is effective.
        # However, the prompt asks for a structural change. 
        # A restart with a lower T from the current best might be good.
        
        # Let's use remaining budget for a "Local Refinement" phase.
        # This is effectively a low-temperature annealing or just random hill climbing.
        # Since Sphere is convex, once we are close, simple (1+1) ES with small sigma works.
        # But to keep it distinct, let's do a few short restarts from best_x with small T.
        
        remaining_budget = budget - calls
        if remaining_budget > 0:
            # Local search around best_x
            # Use a small step size
            local_step = 0.1
            T_local = 1.0
            x = best_x.copy()
            fx = best_f
            
            for _ in range(remaining_budget):
                if calls >= budget or time.time() > deadline:
                    break
                
                dx = rng.standard_normal(dim) * local_step
                x_new = x + dx
                x_new = np.clip(x_new, -5, 5)
                
                fx_new = func(x_new)
                calls += 1
                
                df = fx_new - fx
                if df < 0:
                    x = x_new
                    fx = fx_new
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    if T_local > 1e-10:
                        prob = np.exp(-df / T_local)
                        if rng.random() < prob:
                            x = x_new
                            fx = fx_new
                
                T_local *= 0.99
                local_step *= 0.999
                if local_step < 0.001:
                    local_step = 0.001

        return best_x, best_f
# EVOLVE-BLOCK-END
