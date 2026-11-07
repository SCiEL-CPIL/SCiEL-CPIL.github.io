"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer: (1+1)-ES for global exploration + Nelder-Mead for local ridge tracking.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                f = float(func(x))
            except:
                return None
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Exploration using (1+1)-ES
        # This is robust and fast. We allocate ~30% of the budget or until a plateau is reached.
        # (1+1)-ES is good at finding the general basin of attraction.
        n_es_calls = max(50, budget // 3)
        
        # Initialize
        x = rng.uniform(-4, 4, dim)
        fx = safe_call(x)
        if fx is None:
            return best_x, best_f
            
        sigma = 1.0  # Initial step size
        
        for _ in range(n_es_calls):
            if calls >= budget or time.time() > deadline:
                break
                
            # Generate trial
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = safe_call(y)
            if fy is None:
                break
                
            # Update step size based on success (1/5 rule approximation)
            if fy < fx:
                x = y
                fx = fy
                # Increase step size slightly, capped
                sigma = min(sigma * 1.2, 2.0)
            else:
                # Decrease step size
                sigma *= 0.85
                if sigma < 0.05:
                    # If step size is too small, we might be stuck in a local minimum or very close to optimum.
                    # Break out to let NM take over if we've used enough budget, or continue if NM isn't ready yet.
                    # For now, just keep going but with tiny steps.
                    pass

        # Phase 2: Local Refinement using Nelder-Mead
        # NM is excellent for sharp ridges (like f13) because the simplex stretches along the valley.
        # We start NM from the best point found by ES.
        
        if best_x is None:
            return best_x, best_f
            
        center = best_x.copy()
        
        # Initialize Simplex
        # Use a small initial step size since we are already in a local basin
        init_step = 0.2 
        simplex = [center.copy()]
        f_values = [best_f]
        
        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            xi = center.copy()
            xi[i] += init_step
            fi = safe_call(xi)
            if fi is None:
                break
            simplex.append(xi)
            f_values.append(fi)
            
        if calls >= budget or len(simplex) < dim + 1:
            return best_x, best_f
            
        simplex = np.array(simplex)
        f_values = np.array(f_values)
        
        # NM Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_shrink = 0.5
        
        # Main NM Loop
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Sort simplex
            idx = np.argsort(f_values)
            simplex = simplex[idx]
            f_values = f_values[idx]
            
            # Check for convergence
            spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            if np.max(spread) < 1e-8 or np.max(f_values) - np.min(f_values) < 1e-12:
                # Stagnated. Try a small random perturbation to restart NM?
                # Or just break. Given the budget, breaking is safer to avoid wasting calls.
                # However, if we have lots of budget left, a restart might help.
                # Let's try a single restart if we are stuck and have budget.
                if calls < budget - 10:
                    # Perturb the best point
                    new_center = np.clip(simplex[0] + rng.normal(0, 0.5, dim), -5, 5)
                    new_f = safe_call(new_center)
                    if new_f is None:
                        break
                    if new_f < f_values[0]:
                        # Found something better, restart simplex around it
                        simplex = [new_center.copy()]
                        f_values = [new_f]
                        for i in range(dim):
                            if calls >= budget:
                                break
                            xi = new_center.copy()
                            xi[i] += init_step
                            fi = safe_call(xi)
                            if fi is None:
                                break
                            simplex.append(xi)
                            f_values.append(fi)
                        if len(simplex) < dim + 1:
                            break
                        simplex = np.array(simplex)
                        f_values = np.array(f_values)
                        continue
                break
            
            # Centroid
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = safe_call(xr)
            if fr is None:
                break
            
            if fr < f_values[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
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
                xc = np.clip(xc, -5, 5)
                fc = safe_call(xc)
                if fc is None:
                    break
                if fc <= min(fr, f_values[-1]):
                    simplex[-1] = xc
                    f_values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fi = safe_call(simplex[i])
                        if fi is None:
                            break
                        f_values[i] = fi
                    if fi is None:
                        break

        return best_x, best_f

# EVOLVE-BLOCK-END
