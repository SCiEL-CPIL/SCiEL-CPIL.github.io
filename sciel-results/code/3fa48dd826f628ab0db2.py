import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Parallel Best-First Nelder-Mead
        # Initialize 5 independent simplexes using Latin Hypercube-like sampling
        n_simplexes = 5
        simplex_size = dim + 1  # Number of vertices in each simplex
        simplex_list = []  # List of dicts: {'vertices': [...], 'values': [...], 'active': bool}
        
        # Generate LHS-like points for initialization
        # For 5 points in 5D, we can permute indices to ensure spread
        # Or just use random but stratified
        init_points = []
        # Ensure we cover the space somewhat
        # Point 0: Center
        init_points.append(np.zeros(dim))
        # Points 1-4: Stratified random
        # Split box into 2^5 = 32 cells? No, just 5 points.
        # Let's use a simple Latin Hypercube design for 5 points in 5D.
        # Actually, just 4 random points + center is fine for 5 simplexes?
        # We need 5 distinct starting points.
        
        rng = np.random.default_rng(42)
        
        # Generate 5 starting points
        # 1. Origin
        starts = [np.zeros(dim)]
        # 2-5. Random points in [-4, 4]
        for _ in range(4):
            starts.append(rng.uniform(-4.0, 4.0, dim))
            
        # Initialize simplexes
        base_alpha = 0.5  # Initial step size for simplex vertices
        
        for s_idx in range(n_simplexes):
            x0 = starts[s_idx].copy()
            # Evaluate center
            f0 = func(x0)
            calls += 1
            record(x0, f0)
            
            vertices = [x0.copy()]
            values = [f0]
            
            # Generate other vertices by perturbing one axis at a time
            for i in range(dim):
                if not time_ok() or calls >= budget:
                    # If we run out of budget during init, skip remaining simplexes
                    break
                xi = x0.copy()
                # Alternate direction for diversity
                direction = 1.0 if (s_idx + i) % 2 == 0 else -1.0
                xi[i] += direction * base_alpha
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                record(xi, fi)
                vertices.append(xi)
                values.append(fi)
                
            if len(vertices) == simplex_size:
                simplex_list.append({
                    'vertices': vertices,
                    'values': values,
                    'stagnation': 0
                })
            
            if not time_ok() or calls >= budget:
                break

        if not simplex_list:
            return best_x, best_f

        # Nelder-Mead parameters
        gamma = 1.0
        rho = 2.0
        sigma_nm = 0.5
        shrink = 0.5
        
        # Best-First Queue Logic
        # We don't use a complex priority queue, but we iterate through active simplexes
        # and pick the one with the lowest current best value to prioritize refinement.
        # However, to ensure we don't starve a simplex that might be finding a new basin,
        # we can round-robin but bias towards the best.
        # Given the "Parallel Best-First" directive, we will evaluate the vertex
        # with the highest potential improvement.
        
        # A simpler, robust approach for "Best-First":
        # In each iteration, pick the simplex with the lowest current minimum function value.
        # Perform one NM step on it.
        # If a simplex stagnates (no improvement in X steps), mark it inactive or restart it?
        # For now, let's just let them run. If one is stuck, its best value won't decrease,
        # so others will be preferred.
        
        active_indices = list(range(len(simplex_list)))
        
        # Track global best to detect if we should restart a stuck simplex?
        # The directive says "queue of 5 independent simplexes... each iteration evaluates... 
        # to ensure the most promising basin is always refined".
        
        while calls < budget and time_ok() and active_indices:
            # Find the index of the active simplex with the best (lowest) current minimum
            best_idx = -1
            best_val = np.inf
            for idx in active_indices:
                if idx >= len(simplex_list): continue
                s_min = np.min(simplex_list[idx]['values'])
                if s_min < best_val:
                    best_val = s_min
                    best_idx = idx
            
            if best_idx == -1:
                break
                
            s = simplex_list[best_idx]
            vertices = s['vertices']
            values = s['values']
            
            # Sort simplex by function value
            indices = np.argsort(values)
            vertices = [vertices[i] for i in indices]
            values = [values[i] for i in indices]
            
            x_best = vertices[0]
            f_best = values[0]
            x_worst = vertices[-1]
            f_worst = values[-1]
            x_second_worst = vertices[-2]
            f_second_worst = values[-2]
            
            # Centroid of all but worst
            centroid = np.mean(vertices[:-1], axis=0)
            
            # Reflection
            x_r = centroid + gamma * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            
            if not time_ok() or calls >= budget:
                break
            f_r = func(x_r)
            calls += 1
            record(x_r, f_r)
            
            # Determine if this step improved the simplex best
            improved_simplex = f_r < f_best or (f_r < f_worst and f_r < f_second_worst)
            
            if f_r < f_best:
                # Expansion
                x_e = centroid + rho * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                if not time_ok() or calls >= budget:
                    break
                f_e = func(x_e)
                calls += 1
                record(x_e, f_e)
                if f_e < f_r:
                    vertices[-1] = x_e
                    values[-1] = f_e
                else:
                    vertices[-1] = x_r
                    values[-1] = f_r
                s['stagnation'] = 0
            elif f_r < f_second_worst:
                vertices[-1] = x_r
                values[-1] = f_r
                s['stagnation'] = 0
            else:
                # Contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c = centroid + sigma_nm * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - sigma_nm * (x_worst - centroid)
                x_c = np.clip(x_c, -5, 5)
                
                if not time_ok() or calls >= budget:
                    break
                f_c = func(x_c)
                calls += 1
                record(x_c, f_c)
                
                if f_c < min(f_r, f_worst):
                    vertices[-1] = x_c
                    values[-1] = f_c
                    s['stagnation'] = 0
                else:
                    # Shrink
                    for i in range(1, len(vertices)):
                        if not time_ok() or calls >= budget:
                            break
                        vertices[i] = x_best + shrink * (vertices[i] - x_best)
                        vertices[i] = np.clip(vertices[i], -5, 5)
                        values[i] = func(vertices[i])
                        calls += 1
                        record(vertices[i], values[i])
                    # Shrinkage implies stagnation
                    s['stagnation'] += 1
                    
                    # If a simplex shrinks too much or stagnates for too long, 
                    # we might want to restart it from the global best to explore elsewhere?
                    # Or just let it die. 
                    # Let's mark it inactive if it shrinks too many times.
                    if s['stagnation'] > 5:
                        if best_idx in active_indices:
                            active_indices.remove(best_idx)
                        continue # Skip to next iteration to pick another simplex

            # Update stored simplex
            s['vertices'] = vertices
            s['values'] = values

        return best_x, best_f
