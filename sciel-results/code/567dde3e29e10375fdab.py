"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts for BBOB f5 (linear slope)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        # Initialize simplex at center with a reasonable step size
        # For linear functions, starting at center is often good if the slope is random.
        # However, f5 is "Linear slope". The minimum is in [-4, 4]^5.
        # A simplex size of 1.0 or 2.0 is a good start.
        simplex = [x.copy() + 1.0 * np.eye(self.dim, i) for i in range(self.dim)]
        simplex.append(x.copy() - 1.0 * np.ones(self.dim))
        
        # Evaluate initial simplex
        fvals = []
        for i in range(self.dim + 1):
            if calls >= self.budget:
                break
            if time.time() > deadline - 1.0:
                break
            fx_i = func(simplex[i])
            fvals.append(fx_i)
            calls += 1
            if fx_i < best_fx:
                best_fx = fx_i
                best_x = simplex[i].copy()
        
        # If we ran out of budget or time during initialization, return best
        if len(fvals) < self.dim + 1:
            return best_x, best_fx
            
        # Main Nelder-Mead loop
        while calls < self.budget and time.time() < deadline - 1.0:
            # Sort simplex by f-values
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check for convergence
            if abs(fvals[0] - fvals[-1]) < 1e-12:
                break
                
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            if calls >= self.budget or time.time() > deadline - 1.0:
                break
            f_r = func(x_r)
            calls += 1
            if f_r < best_fx:
                best_fx = f_r
                best_x = x_r.copy()
                
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5, 5)
                if calls >= self.budget or time.time() > deadline - 1.0:
                    break
                f_e = func(x_e)
                calls += 1
                if f_e < best_fx:
                    best_fx = f_e
                    best_x = x_e.copy()
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Reflection is good enough
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - simplex[-1])
                    
                x_c = np.clip(x_c, -5, 5)
                if calls >= self.budget or time.time() > deadline - 1.0:
                    break
                f_c = func(x_c)
                calls += 1
                if f_c < best_fx:
                    best_fx = f_c
                    best_x = x_c.copy()
                    
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, self.dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        if calls >= self.budget or time.time() > deadline - 1.0:
                            break
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if fvals[i] < best_fx:
                            best_fx = fvals[i]
                            best_x = simplex[i].copy()
                    if calls >= self.budget or time.time() > deadline - 1.0:
                        break

        # Fill remaining budget with local random search around best_x
        # This ensures we use all 500 calls, which helps AOCC if we are already close
        # but maybe stuck in a tiny local minimum (unlikely for linear, but good practice)
        # Or just to satisfy the "repeat last n_i" logic if we finish early?
        # The scoring says: "repeating the last n_i if fewer calls are made".
        # So making fewer calls is fine, but we have budget.
        # For a linear function, once we find the minimum (or boundary), we are done.
        # However, if we are not at the global minimum, local search might help.
        # Given the "Linear slope" nature, Nelder-Mead should converge very fast.
        # The remaining budget can be used to verify or just idle.
        # Let's do a simple local random search to be safe.
        step = 0.1
        while calls < self.budget and time.time() < deadline - 1.0:
            x_new = best_x + step * np.random.randn(self.dim)
            x_new = np.clip(x_new, -5, 5)
            f_new = func(x_new)
            calls += 1
            if f_new < best_fx:
                best_fx = f_new
                best_x = x_new.copy()
                step *= 1.2
            else:
                step *= 0.9
            if step < 1e-6:
                # Stagnated, break to save time? Or just keep calling?
                # If we break, we return early. The score uses the last n_i.
                # If we are at the optimum, n_i is minimal.
                # Breaking is fine.
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
