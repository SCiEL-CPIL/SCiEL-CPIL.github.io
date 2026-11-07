"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead Simplex Optimizer with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        best_x = None
        best_f = float('inf')
        
        calls = 0
        rng = np.random.default_rng(42)
        
        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            calls += 1
            try:
                return func(x)
            except:
                return None

        # Initialize simplex
        # Start with a reasonably large simplex to explore the space
        # Base point near center
        center = np.zeros(dim)
        simplex = []
        f_values = []
        
        # First vertex at center
        x0 = center.copy()
        f0 = safe_call(x0)
        if f0 is None:
            return best_x, best_f
        simplex.append(x0)
        f_values.append(f0)
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()
            
        # Other vertices along axes with initial step size
        step = 2.0
        for i in range(dim):
            xi = center.copy()
            xi[i] += step
            fi = safe_call(xi)
            if fi is None:
                break
            simplex.append(xi)
            f_values.append(fi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
                
        if calls >= budget or len(simplex) < dim + 1:
            return best_x, best_f
            
        simplex = np.array(simplex)
        f_values = np.array(f_values)
        
        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Sort simplex by function value
            idx = np.argsort(f_values)
            simplex = simplex[idx]
            f_values = f_values[idx]
            
            # Update global best
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = simplex[0].copy()
                
            # Check for convergence/stagnation
            # If simplex is too small or values are very close
            spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            if np.max(spread) < 1e-6 or np.max(f_values) - np.min(f_values) < 1e-10:
                # Restart with a larger step or random restart
                # Shrink the simplex towards the best point and then expand?
                # Or just restart with a new random point near best
                # Let's try expanding the simplex from the best point
                center = simplex[0].copy()
                simplex_new = [center]
                f_values_new = [f_values[0]]
                step = 1.0 # Smaller step for refinement
                for i in range(dim):
                    xi = center.copy()
                    xi[i] += step
                    fi = safe_call(xi)
                    if fi is None:
                        break
                    simplex_new.append(xi)
                    f_values_new.append(fi)
                if len(simplex_new) < dim + 1:
                    break
                simplex = np.array(simplex_new)
                f_values = np.array(f_values_new)
                continue

            # Standard Nelder-Mead steps
            # Centroid of all points except the worst
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
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fi = safe_call(simplex[i])
                        if fi is None:
                            break
                        f_values[i] = fi
                    if fi is None:
                        break

        return best_x, best_f

# EVOLVE-BLOCK-END
