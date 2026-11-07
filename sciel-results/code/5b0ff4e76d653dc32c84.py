import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(123)
        N = self.dim
        bounds_low = -5.0
        bounds_high = 5.0
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def update_best(x, f):
            nonlocal best_x, best_f, calls
            if f < best_f:
                best_f = f
                best_x = x.copy()
            calls += 1

        # Phase 1: Random Search
        # Use a portion of the budget to find a good starting point.
        # 20 points is a good balance.
        n_random = 20
        for _ in range(n_random):
            if calls >= self.budget:
                break
            x = rng.uniform(bounds_low, bounds_high, N)
            f = func(x)
            update_best(x, f)
            
        if best_x is None:
            # Should not happen if budget > 0
            return np.zeros(N), np.inf

        # Phase 2: Nelder-Mead
        # Initialize simplex around best_x
        # Simplex size: 0.5 * (bounds_high - bounds_low) / N? Or just a fixed size like 1.0 or 2.0.
        # Given the box is [-5, 5], size 2.0 is reasonable.
        simplex_size = 2.0
        
        # Create initial simplex
        simplex = []
        simplex.append(best_x.copy())
        for i in range(N):
            v = best_x.copy()
            v[i] += simplex_size
            v[i] = np.clip(v[i], bounds_low, bounds_high)
            simplex.append(v)
            
        # Evaluate initial simplex
        for i in range(N + 1):
            if calls >= self.budget:
                break
            f = func(simplex[i])
            update_best(simplex[i], f)
            # Store fitness in parallel list for NM logic
            # We need to keep track of fitness for each vertex
        # We need a list of (x, f) for the simplex
        simplex_data = []
        for i in range(N + 1):
            # Re-evaluate? No, we already did. But we need to map them.
            # Actually, the loop above updated global best, but we need local fitness for NM steps.
            # It's safer to store the fitness of the current simplex vertices.
            pass
        
        # Let's restructure NM to manage state cleanly.
        # We will maintain a list of vertices and their fitness.
        
        # Re-initialize simplex_data from the evaluations above? 
        # The loop above didn't store per-vertex fitness in a way that's easy to access if we just called update_best.
        # Let's do Phase 2 initialization more carefully.
        
        # Clear previous simplex logic if we want to be clean.
        # Actually, let's just implement a standard NM loop that evaluates points as needed.
        
        # Current simplex vertices and their fitness
        vertices = []
        fitnesses = []
        
        # Add the best point found in Phase 1
        vertices.append(best_x.copy())
        fitnesses.append(best_f)
        
        # Add other vertices
        for i in range(N):
            if calls >= self.budget:
                break
            v = best_x.copy()
            # Step size in direction i
            step = 1.0 
            v[i] += step
            v[i] = np.clip(v[i], bounds_low, bounds_high)
            # Avoid duplicates
            if any(np.allclose(v, u) for u in vertices):
                v[i] -= 2 * step # Try opposite direction
                v[i] = np.clip(v[i], bounds_low, bounds_high)
                if any(np.allclose(v, u) for u in vertices):
                    v = rng.uniform(bounds_low, bounds_high, N)
            
            f = func(v)
            update_best(v, f)
            vertices.append(v)
            fitnesses.append(f)
            
        if len(vertices) < N + 1:
            # Not enough budget or failed to build simplex
            return best_x, best_f
            
        # Nelder-Mead Parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink
        
        while calls + 1 <= self.budget:
            # Sort vertices by fitness
            sorted_indices = np.argsort(fitnesses)
            vertices = [vertices[i] for i in sorted_indices]
            fitnesses = [fitnesses[i] for i in sorted_indices]
            
            x_worst = vertices[-1]
            f_worst = fitnesses[-1]
            x_best = vertices[0]
            f_best = fitnesses[0]
            
            # Centroid of all but worst
            centroid = np.mean(vertices[:-1], axis=0)
            
            # 1. Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, bounds_low, bounds_high)
            
            if calls >= self.budget:
                break
            f_r = func(x_r)
            update_best(x_r, f_r)
            
            if f_r < f_best:
                # 2. Expansion
                if calls >= self.budget:
                    break
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, bounds_low, bounds_high)
                f_e = func(x_e)
                update_best(x_e, f_e)
                
                if f_e < f_r:
                    # Replace worst with expanded
                    vertices[-1] = x_e
                    fitnesses[-1] = f_e
                else:
                    # Replace worst with reflected
                    vertices[-1] = x_r
                    fitnesses[-1] = f_r
            elif f_r < fitnesses[-2]:
                # 3. Accept Reflection
                vertices[-1] = x_r
                fitnesses[-1] = f_r
            else:
                # 4. Contraction
                # Outside or Inside contraction
                if f_r < f_worst:
                    # Inside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Outside contraction
                    x_c = centroid - rho * (centroid - x_worst)
                
                x_c = np.clip(x_c, bounds_low, bounds_high)
                
                if calls >= self.budget:
                    break
                f_c = func(x_c)
                update_best(x_c, f_c)
                
                if f_c < min(f_r, f_worst):
                    vertices[-1] = x_c
                    fitnesses[-1] = f_c
                else:
                    # 5. Shrink
                    # Shrink all points except best towards best
                    # This requires N evaluations
                    # Check budget
                    if calls + N > self.budget:
                        break
                    for i in range(1, N + 1):
                        if calls >= self.budget:
                            break
                        v = vertices[i]
                        new_v = x_best + sigma * (v - x_best)
                        new_v = np.clip(new_v, bounds_low, bounds_high)
                        f_s = func(new_v)
                        update_best(new_v, f_s)
                        vertices[i] = new_v
                        fitnesses[i] = f_s
                        
        return best_x, best_f
