import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer: Global Random Search followed by Nelder-Mead Local Refinement.
    Designed for BBOB f13 (Sharp Ridge) in 5D with budget 500.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        # Safety margin: Evaluator kills at 330s. We target finishing well before.
        deadline = time.time() + 285.0 
        
        best_x = None
        best_f = float('inf')
        calls = 0
        rng = np.random.default_rng(1234) # Fixed seed for reproducibility within run
        
        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                f_val = func(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Random Search
        # Use more samples to better locate the basin in 5D
        n_random = 25
        
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            # Fallback if all random calls failed (unlikely)
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0

        # Phase 2: Nelder-Mead Local Refinement with ridge-aware initialization
        x_start = best_x.copy()
        
        # Nelder-Mead parameters - tuned for sharp ridge
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initialize simplex around x_start with larger initial step
        # to capture the ridge structure
        step_size = 1.5
        
        simplex = [x_start]
        f_values = [best_f]
        
        # Create other vertices along axes
        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            xi = x_start.copy()
            xi[i] += step_size
            fi = safe_call(xi)
            if fi is None:
                break
            simplex.append(xi)
            f_values.append(fi)
        
        if len(simplex) < dim + 1:
            # If we couldn't initialize fully, just return best so far
            return best_x, best_f

        simplex = np.array(simplex)
        f_values = np.array(f_values)
        
        # Track progress to detect stagnation
        last_improvement_call = calls
        best_f_at_last_improvement = best_f
        
        # Main Nelder-Mead Loop
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Sort simplex by function value (best first)
            idx = np.argsort(f_values)
            simplex = simplex[idx]
            f_values = f_values[idx]
            
            # Update global best and track improvement
            if f_values[0] < best_f - 1e-12:
                best_f = f_values[0]
                best_x = simplex[0].copy()
                last_improvement_call = calls
                best_f_at_last_improvement = best_f
                
            # Check for convergence or stagnation
            spread_x = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            spread_f = np.max(f_values) - np.min(f_values)
            
            # Detect stagnation: no improvement for many calls
            stalled = (calls - last_improvement_call > 50) and (best_f - best_f_at_last_improvement < 1e-8)
            
            if np.max(spread_x) < 1e-7 or spread_f < 1e-12 or stalled:
                # Try to restart with a different step size if budget allows
                if calls < budget - 15:
                    # Determine new step size based on current spread
                    current_spread = np.max(spread_x)
                    if current_spread < 1e-6:
                        new_step = 0.3  # Very fine refinement
                    elif current_spread < 0.1:
                        new_step = 0.5
                    else:
                        new_step = 1.0
                    
                    center = simplex[0].copy()
                    new_simplex = [center]
                    new_f_values = [f_values[0]]
                    for i in range(dim):
                        if calls >= budget or time.time() > deadline:
                            break
                        xi = center.copy()
                        xi[i] += new_step
                        fi = safe_call(xi)
                        if fi is None:
                            break
                        new_simplex.append(xi)
                        new_f_values.append(fi)
                    if len(new_simplex) == dim + 1:
                        simplex = np.array(new_simplex)
                        f_values = np.array(new_f_values)
                        last_improvement_call = calls
                        continue
                break

            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = safe_call(xr)
            if fr is None:
                break
            
            if fr < f_values[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = safe_call(xe)
                if fe is None:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
            elif fr < f_values[-2]:
                # Reflection accepted
                simplex[-1] = xr
                f_values[-1] = fr
            else:
                # Contraction
                if fr < f_values[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5.0, 5.0)
                fc = safe_call(xc)
                if fc is None:
                    break
                if fc <= min(fr, f_values[-1]):
                    simplex[-1] = xc
                    f_values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        if calls >= budget or time.time() > deadline:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fi = safe_call(simplex[i])
                        if fi is None:
                            break
                        f_values[i] = fi
                    if fi is None:
                        break

        # Phase 3: Final high-precision polish using coordinate descent
        # Use remaining budget for fine-tuning
        if calls < budget and time.time() < deadline:
            x_polish = best_x.copy()
            f_polish = best_f
            step = 0.1
            
            # Coordinate descent with adaptive step size
            for coord in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                    
                # Try positive and negative directions
                for direction in [1, -1]:
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    x_trial = x_polish.copy()
                    x_trial[coord] += direction * step
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    f_trial = safe_call(x_trial)
                    
                    if f_trial is not None and f_trial < f_polish:
                        x_polish = x_trial
                        f_polish = f_trial
                        # Try to continue in same direction
                        for _ in range(3):
                            if calls >= budget or time.time() > deadline:
                                break
                            x_trial2 = x_polish.copy()
                            x_trial2[coord] += direction * step
                            x_trial2 = np.clip(x_trial2, -5.0, 5.0)
                            f_trial2 = safe_call(x_trial2)
                            if f_trial2 is not None and f_trial2 < f_polish:
                                x_polish = x_trial2
                                f_polish = f_trial2
                            else:
                                break
                    else:
                        break
                
                # Halve step size for next coordinate
                step *= 0.5
            
            # Update best if polish improved
            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish

        return best_x, best_f
