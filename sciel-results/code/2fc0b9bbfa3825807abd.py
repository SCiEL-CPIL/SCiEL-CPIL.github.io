import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = np.zeros(n)
        best_f = np.inf

        def do_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        
        # Initialize 5 Nelder-Mead simplexes at different locations
        # Locations: Center, and 4 corners of the box
        init_points = [
            np.zeros(n),
            np.array([5.0] * n),
            np.array([-5.0] * n),
            np.array([5.0] * n),
            np.array([-5.0] * n)
        ]
        
        # Add some randomness to avoid symmetry issues
        for i in range(len(init_points)):
            init_points[i] = init_points[i] + rng.standard_normal(n) * 0.1
            init_points[i] = np.clip(init_points[i], -5.0, 5.0)

        # Simplex class for Nelder-Mead
        class Simplex:
            def __init__(self, initial_point, idx):
                self.points = [initial_point.copy()]
                self.fvals = [do_call(initial_point)]
                self.idx = idx
                # Create initial simplex by perturbing each dimension
                for i in range(n):
                    new_point = initial_point.copy()
                    # Perturb dimension i by a small amount
                    # Use a step size relative to the box size
                    step = 0.5 if i % 2 == 0 else -0.5
                    new_point[i] += step
                    new_point = np.clip(new_point, -5.0, 5.0)
                    self.points.append(new_point)
                    self.fvals.append(do_call(new_point))
                self.sort()

            def sort(self):
                # Sort points by fitness
                order = np.argsort(self.fvals)
                self.points = [self.points[i] for i in order]
                self.fvals = [self.fvals[i] for i in order]

            def step(self):
                if calls_made >= total_budget or not time_ok():
                    return False
                
                # 1. Centroid of all points except the worst
                centroid = np.mean(self.points[:-1], axis=0)
                
                # 2. Reflection
                worst = self.points[-1]
                alpha = 1.0
                reflected = centroid + alpha * (centroid - worst)
                reflected = np.clip(reflected, -5.0, 5.0)
                f_reflected = do_call(reflected)
                
                if f_reflected < self.fvals[-2]:
                    # 3. Expansion
                    gamma = 2.0
                    expanded = centroid + gamma * (reflected - centroid)
                    expanded = np.clip(expanded, -5.0, 5.0)
                    f_expanded = do_call(expanded)
                    
                    if f_expanded < f_reflected:
                        new_point = expanded
                        new_fval = f_expanded
                    else:
                        new_point = reflected
                        new_fval = f_reflected
                else:
                    if f_reflected < self.fvals[-1]:
                        new_point = reflected
                        new_fval = f_reflected
                    else:
                        # 4. Contraction
                        rho = 0.5
                        contracted = centroid + rho * (worst - centroid)
                        contracted = np.clip(contracted, -5.0, 5.0)
                        f_contracted = do_call(contracted)
                        
                        if f_contracted < self.fvals[-1]:
                            new_point = contracted
                            new_fval = f_contracted
                        else:
                            # 5. Shrinkage
                            for i in range(1, n + 1):
                                self.points[i] = self.points[0] + 0.5 * (self.points[i] - self.points[0])
                                self.points[i] = np.clip(self.points[i], -5.0, 5.0)
                                self.fvals[i] = do_call(self.points[i])
                            self.sort()
                            return True
                            
                # Replace worst point
                self.points[-1] = new_point
                self.fvals[-1] = new_fval
                self.sort()
                return True

        # Create simplexes
        simplexes = []
        for i, p in enumerate(init_points):
            if calls_made >= total_budget or not time_ok():
                break
            simplexes.append(Simplex(p, i))

        # Run the ensemble
        # We will cycle through the simplexes.
        # To balance exploration, we can run each simplex for a few steps.
        # Given 500 budget and 5 simplexes, we have ~100 steps per simplex if we were serial.
        # But we are interleaving.
        
        # Strategy: Interleave steps.
        # Keep track of consecutive failures per simplex to detect stagnation?
        # For now, just round-robin.
        
        while calls_made < total_budget and time_ok():
            for simplex in simplexes:
                if calls_made >= total_budget or not time_ok():
                    break
                # Perform one Nelder-Mead step
                simplex.step()
                
                # Optional: If a simplex is stuck (small change in best f), restart it?
                # For simplicity, we rely on the initial diversity and NM's local search power.
                # However, if a simplex converges to a local minimum, it might waste budget.
                # We can check if the spread of the simplex is very small.
                # If spread < 1e-4, we might restart it at a random location or best global.
                # Let's add a simple restart logic if the simplex is too small.
                spread = np.linalg.norm(np.array(simplex.points) - simplex.points[0])
                if spread < 1e-4:
                    # Restart this simplex at the current global best with some noise
                    # Or a random point? Random point is better for escaping basins.
                    new_center = rng.uniform(-5, 5, size=n)
                    # Re-initialize this simplex
                    simplex.points = [new_center.copy()]
                    simplex.fvals = [do_call(new_center)]
                    for i in range(n):
                        new_point = new_center.copy()
                        step = 0.5 if i % 2 == 0 else -0.5
                        new_point[i] += step
                        new_point = np.clip(new_point, -5.0, 5.0)
                        simplex.points.append(new_point)
                        simplex.fvals.append(do_call(new_point))
                    simplex.sort()

        return best_x, best_f
