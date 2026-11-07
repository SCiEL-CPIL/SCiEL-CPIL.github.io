"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        dim = self.dim
        budget = self.budget
        deadline = self.start_time + 290  # 10s safety margin
        
        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrinkage
        
        best_x_global = None
        best_f_global = np.inf
        calls_made = 0
        
        # We will run multiple independent Nelder-Mead instances.
        # Each instance starts with a simplex of size dim+1.
        # Then it iterates.
        # We need to manage the budget carefully.
        
        # Strategy:
        # 1. Pick a random starting point x0.
        # 2. Build a simplex around x0.
        # 3. Run Nelder-Mead until local convergence or budget/time limit.
        # 4. If budget remains, restart with a new random x0 (possibly biased towards best found so far).
        
        # To maximize AOCC (area under curve of progress), we want to find good solutions early.
        # Nelder-Mead converges fast locally.
        # We should use a small number of restarts to keep the "best so far" improving steadily.
        
        # Let's estimate cost per NM run.
        # Initial simplex: dim+1 calls.
        # Iterations: Usually < 100 * dim for d=5. Let's say ~50-100 calls per run is typical for good convergence.
        # With budget 500, we can afford ~5-10 runs.
        
        # Let's do up to 10 restarts, or until budget is exhausted.
        # To ensure early progress, the first few restarts should be purely random to sample the space.
        # Later restarts can be centered near the current best to refine.
        
        rng = np.random.default_rng(42)
        
        for run_idx in range(20):  # Max 20 restarts, but will break on budget/time
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            # Choose starting point
            if run_idx == 0:
                # Start at center? Or random?
                # Center is (0,0,0,0,0). The minimum is in [-4,4]^5.
                # Let's start with a few random points to ensure coverage.
                x0 = rng.uniform(-5, 5, dim)
            else:
                # Bias towards the best found so far?
                # If we have a good best, we can try to refine it.
                # But NM is local. Restarting near best might just find the same local min.
                # Better to keep exploring until budget is low, then refine.
                # Let's use a mix: 50% random, 50% near best.
                if run_idx < 5:
                    x0 = rng.uniform(-5, 5, dim)
                else:
                    # Gaussian perturbation around best
                    if best_x_global is None:
                        x0 = rng.uniform(-5, 5, dim)
                    else:
                        # Scale perturbation to cover a reasonable range
                        # If we are close to optimum, step should be smaller.
                        # We don't know the scale. Let's use a fixed small scale for refinement.
                        # Or just random again? 
                        # Actually, for BBOB, global random restarts are often effective with NM.
                        # Let's stick to random for simplicity and robustness, 
                        # but maybe shrink the range if we are confident?
                        # No, let's just use random. NM is fast.
                        x0 = rng.uniform(-5, 5, dim)
            
            # Initialize Simplex
            # x0 is the first vertex.
            # Other vertices are x0 + delta * e_i
            # Delta size: 5% of box size? Box is 10. So delta = 0.5.
            delta = 0.5
            
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(dim):
                simplex[i+1] = x0
                simplex[i+1][i] += delta
            
            # Evaluate initial simplex
            fits = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                x = np.clip(simplex[i], -5, 5)
                f = func(x)
                fits[i] = f
                calls_made += 1
                simplex[i] = x  # Store clipped version
                
            if calls_made >= budget or time.time() > deadline:
                # Update global best from this partial simplex
                min_idx = np.argmin(fits)
                if fits[min_idx] < best_f_global:
                    best_f_global = fits[min_idx]
                    best_x_global = simplex[min_idx].copy()
                break
            
            # Main Nelder-Mead Loop
            # We limit the number of iterations per run to ensure we can do multiple runs.
            # Let's allow up to 50 iterations per run.
            max_iter = 50
            
            for _ in range(max_iter):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Sort simplex by fitness
                order = np.argsort(fits)
                simplex = simplex[order]
                fits = fits[order]
                
                # Check convergence: difference between best and worst is small
                if abs(fits[-1] - fits[0]) < 1e-8:
                    break
                
                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # 1. Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_made += 1
                
                if fr < fits[0]:
                    # 2. Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_made += 1
                    
                    if fe < fr:
                        simplex[-1] = xe
                        fits[-1] = fe
                    else:
                        simplex[-1] = xr
                        fits[-1] = fr
                elif fr < fits[-2]:
                    # 3. Simple Reflection
                    simplex[-1] = xr
                    fits[-1] = fr
                else:
                    # 4. Contraction
                    # Outside contraction if fr < fits[-1], otherwise inside
                    if fr < fits[-1]:
                        xc = centroid + rho * (simplex[-1] - centroid)
                        fc = func(xc)
                        calls_made += 1
                        if fc <= fr:
                            simplex[-1] = xc
                            fits[-1] = fc
                        else:
                            # 5. Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                if calls_made >= budget or time.time() > deadline:
                                    break
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fits[i] = func(simplex[i])
                                calls_made += 1
                    else:
                        xc = centroid - rho * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        calls_made += 1
                        if fc <= fits[-1]:
                            simplex[-1] = xc
                            fits[-1] = fc
                        else:
                            # 5. Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                if calls_made >= budget or time.time() > deadline:
                                    break
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fits[i] = func(simplex[i])
                                calls_made += 1
                
                # Update global best
                min_idx = np.argmin(fits)
                if fits[min_idx] < best_f_global:
                    best_f_global = fits[min_idx]
                    best_x_global = simplex[min_idx].copy()
                    
                if calls_made >= budget or time.time() > deadline:
                    break
            
            if calls_made >= budget or time.time() > deadline:
                break

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
