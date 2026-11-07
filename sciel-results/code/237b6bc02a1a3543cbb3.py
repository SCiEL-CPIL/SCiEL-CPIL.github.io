import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Leave 10 seconds margin for safety
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        # Multi-start Nelder-Mead strategy
        # Partition budget into 10 runs of 50 calls each
        n_starts = 10
        calls_per_start = budget // n_starts
        
        best_x = None
        best_f = float('inf')
        total_calls = 0
        
        # Deterministic starting points within [-4, 4]^dim
        # Use a simple deterministic pattern to ensure coverage
        starts = []
        for i in range(n_starts):
            # Create a deterministic point based on index
            # Use a simple hash-like function to spread points
            point = np.zeros(dim)
            for d in range(dim):
                # Alternate between -4 and 4 with some variation
                val = -4.0 + 8.0 * ((i * (d + 1) + d) % n_starts) / n_starts
                point[d] = val
            starts.append(point)
        
        # Add origin as a potential good start
        starts.append(np.zeros(dim))
        
        for start_idx, start_point in enumerate(starts):
            if total_calls >= budget or not time_ok():
                break
            
            # Initialize Nelder-Mead simplex
            # Start with a simplex around start_point
            initial_step = 0.5
            simplex = [start_point.copy()]
            for d in range(dim):
                point = start_point.copy()
                point[d] += initial_step
                simplex.append(point)
            
            # Evaluate initial simplex
            f_simplex = []
            for point in simplex:
                if total_calls >= budget or not time_ok():
                    break
                f_val = func(point)
                total_calls += 1
                f_simplex.append(f_val)
                if f_val < best_f:
                    best_f = f_val
                    best_x = point.copy()
            
            if len(f_simplex) < dim + 1:
                continue
            
            # Nelder-Mead iterations
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5  # shrink
            
            for iter_nm in range(calls_per_start - (dim + 1)):
                if total_calls >= budget or not time_ok():
                    break
                
                # Sort simplex by fitness
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                worst_point = simplex[-1]
                reflect_point = centroid + alpha * (centroid - worst_point)
                
                if total_calls >= budget or not time_ok():
                    break
                f_reflect = func(reflect_point)
                total_calls += 1
                if f_reflect < best_f:
                    best_f = f_reflect
                    best_x = reflect_point.copy()
                
                # Check if reflection is good
                if f_simplex[0] <= f_reflect < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = reflect_point
                    f_simplex[-1] = f_reflect
                elif f_reflect < f_simplex[0]:
                    # Try expansion
                    expand_point = centroid + gamma * (reflect_point - centroid)
                    if total_calls >= budget or not time_ok():
                        break
                    f_expand = func(expand_point)
                    total_calls += 1
                    if f_expand < best_f:
                        best_f = f_expand
                        best_x = expand_point.copy()
                    
                    if f_expand < f_reflect:
                        simplex[-1] = expand_point
                        f_simplex[-1] = f_expand
                    else:
                        simplex[-1] = reflect_point
                        f_simplex[-1] = f_reflect
                else:
                    # Try contraction
                    if f_reflect < f_simplex[-1]:
                        # Outside contraction
                        contract_point = centroid + rho * (reflect_point - centroid)
                    else:
                        # Inside contraction
                        contract_point = centroid - rho * (worst_point - centroid)
                    
                    if total_calls >= budget or not time_ok():
                        break
                    f_contract = func(contract_point)
                    total_calls += 1
                    if f_contract < best_f:
                        best_f = f_contract
                        best_x = contract_point.copy()
                    
                    if f_contract < min(f_reflect, f_simplex[-1]):
                        simplex[-1] = contract_point
                        f_simplex[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            if total_calls >= budget or not time_ok():
                                break
                            f_shrink = func(simplex[i])
                            total_calls += 1
                            f_simplex[i] = f_shrink
                            if f_shrink < best_f:
                                best_f = f_shrink
                                best_x = simplex[i].copy()
                        break
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
        
        return best_x, best_f
