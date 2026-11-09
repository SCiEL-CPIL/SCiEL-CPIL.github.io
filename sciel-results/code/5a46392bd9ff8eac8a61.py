"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Deterministic Nelder-Mead simplex optimizer for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        call_count = 0
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # Initialize simplex: origin and axis offsets
        simplex = [np.zeros(dim)]
        for d in range(dim):
            v = np.zeros(dim)
            v[d] = 4.0
            simplex.append(v)
            
        vals = [wrapped_func(x) for x in simplex]
        best_val = min(vals)
        best_x = simplex[np.argmin(vals)].copy()

        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        while call_count < budget and time.time() < deadline:
            # Sort simplex by fitness
            order = np.argsort(vals)
            simplex = [simplex[i] for i in order]
            vals = [vals[i] for i in order]
            
            if vals[0] < best_val:
                best_val = vals[0]
                best_x = simplex[0].copy()

            # If the simplex has converged, we could break, but NM usually 
            # just keeps going or we can break if range is tiny.
            if vals[-1] - vals[0] < 1e-12:
                break

            x_worst = simplex[-1]
            f_worst = vals[-1]
            
            # Centroid of all but worst
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_reflect = x_centroid + alpha * (x_centroid - x_worst)
            f_reflect = wrapped_func(x_reflect)
            
            if f_reflect < vals[0]:
                # Expansion
                x_expand = x_centroid + gamma * (x_centroid - x_worst)
                f_expand = wrapped_func(x_expand)
                
                if f_expand < f_reflect:
                    simplex[-1] = x_expand
                    vals[-1] = f_expand
                else:
                    simplex[-1] = x_reflect
                    vals[-1] = f_reflect
            elif f_reflect < vals[-2]:
                # Accept reflection
                simplex[-1] = x_reflect
                vals[-1] = f_reflect
            else:
                # Contraction
                # Outside contraction if f_worst < f_reflect, else inside
                if f_reflect < f_worst:
                    # Outside contraction
                    x_contract = x_centroid + 0.5 * (x_reflect - x_centroid)
                    f_contract = wrapped_func(x_contract)
                    
                    if f_contract <= f_reflect:
                        simplex[-1] = x_contract
                        vals[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            vals[i] = wrapped_func(simplex[i])
                else:
                    # Inside contraction
                    x_contract = x_centroid + 0.5 * (x_worst - x_centroid)
                    f_contract = wrapped_func(x_contract)
                    
                    if f_contract < f_worst:
                        simplex[-1] = x_contract
                        vals[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            vals[i] = wrapped_func(simplex[i])

        return best_x, best_val

# EVOLVE-BLOCK-END
