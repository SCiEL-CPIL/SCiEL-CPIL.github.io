"""Optimizer for BBOB-style f15 (Rastrigin rotated) with Multi-start (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5 success rule for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0

        # Strategy: Hybrid Multi-start (1+1)-ES with Lattice Refinement
        # 1. Use diverse initial points to escape local minima.
        # 2. Use (1+1)-ES for broad exploration and convergence to basin.
        # 3. When sigma becomes small (converged), switch to a discrete lattice-based
        #    random walk to exploit Rastrigin's periodic structure and fine-tune.
        
        n_restarts = 6
        calls_per_restart = budget // n_restarts

        # Define diverse initial points
        # 1. Origin
        # 2-6. Random uniform points in different sub-regions
        init_points = [np.zeros(dim)]
        for i in range(n_restarts - 1):
            # Spread initial points across the domain
            center = rng.uniform(-4, 4, dim)
            init_points.append(center)

        for r in range(n_restarts):
            # Check time and budget
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Initialize restart point
            x = init_points[r].copy()
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # (1+1)-ES loop for this restart
            sigma = 1.5  # Initial step size
            max_calls_this_restart = calls_per_restart - 1 # 1 call for init
            if max_calls_this_restart < 0:
                continue

            # Phase 1: Continuous (1+1)-ES
            for _ in range(max_calls_this_restart):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # If sigma is very small, we are likely in a local minimum.
                # Switch to lattice-based refinement for the remainder of this restart's budget
                if sigma < 0.1:
                    # Lattice refinement phase
                    # Use a small step size corresponding to the grid of local minima (approx 1.0)
                    # But since we don't know the exact scale, we use a small step.
                    # The directive suggests "4-neighbor or 8-neighbor moves on a fixed grid".
                    # For Rastrigin, the local minima are at integer coordinates (in the rotated frame).
                    # A step of 0.5 or 0.1 might help jump between nearby minima or fine-tune.
                    # Let's use a small continuous step but with a different adaptation rule
                    # or just continue with smaller sigma.
                    # Actually, let's try a specific "lattice" step:
                    # Try moving along axes by a small amount.
                    
                    # For simplicity and robustness, we continue the (1+1)-ES but with a floor on sigma
                    # to prevent it from becoming too small to escape tiny noise, 
                    # or we switch to a pure random walk with small steps.
                    
                    # Let's implement a simple lattice-like step:
                    # Pick a random axis and move by a small fixed amount.
                    axis = rng.integers(0, dim)
                    step = 0.1 * rng.choice([-1, 1])
                    y = x.copy()
                    y[axis] += step
                    y = np.clip(y, -5, 5)
                    
                    fy = func(y)
                    calls_made += 1
                    
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        # If it fails, try another axis or shrink step?
                        # For now, just continue.
                        pass
                    continue

                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    # Success
                    x = y
                    fx = fy
                    sigma *= 1.5
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Failure
                    sigma *= 0.87

        # If we still have budget, we can do a final refinement on the best point found
        if best_x is not None and calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5 # Smaller sigma for refinement
            remaining = budget - calls_made
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                
                # Lattice-based refinement for the final phase
                # This exploits the periodicity.
                # Try small steps along axes.
                axis = rng.integers(0, dim)
                # Use a step size that is likely to be smaller than the distance between minima
                step = 0.1 * rng.choice([-1, 1])
                y = x.copy()
                y[axis] += step
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
