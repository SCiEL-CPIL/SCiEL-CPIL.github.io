import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        N = self.budget

        def time_check():
            return time.time() - self.start_time < 280

        def safe_eval(x):
            if self.calls >= N or not time_check():
                return np.inf
            self.calls += 1
            try:
                val = func(x)
                return float(val)
            except Exception:
                return np.inf

        best_x = np.zeros(d)
        best_f = np.inf
        no_improve_count = 0
        max_no_improve = 20 # Restart if no improvement for 20 iterations
        
        # Initial point
        x0 = np.zeros(d)
        f0 = safe_eval(x0)
        best_f = f0
        best_x = x0.copy()
        
        # We will run multiple Nelder-Mead runs from different starting points
        # Start points: 0, and a few random points in the box
        start_points = [np.zeros(d)]
        # Add a few random starts to avoid local minima
        np.random.seed(42) # Fixed seed for reproducibility in optimizer logic? No, let it be random or use a fixed one.
        # The task says "hidden seeded shift", so the function is deterministic.
        # We should probably use a fixed seed for our random starts to be consistent?
        # Or just use np.random.
        
        for _ in range(5):
            start_points.append(np.random.uniform(-4, 4, d))
            
        for sp in start_points:
            if self.calls >= N or not time_check():
                break
                
            # Run Nelder-Mead from sp
            x_current = sp.copy()
            f_current = safe_eval(x_current)
            
            if f_current < best_f:
                best_f = f_current
                best_x = x_current.copy()
                no_improve_count = 0
            else:
                no_improve_count = 0 # Reset for new start? Or keep global?
                # Let's keep global no_improve to stop if we are stuck.
                # But if we restart, we hope for improvement.
                
            # Simplex initialization
            delta = 0.5
            simplex = [x_current]
            f_vals = [f_current]
            
            for i in range(d):
                x_i = x_current.copy()
                x_i[i] += delta
                f_i = safe_eval(x_i)
                simplex.append(x_i)
                f_vals.append(f_i)
                
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
            if no_improve_count > max_no_improve:
                continue # Try next start point
                
            # Nelder-Mead Loop
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            tol = 1e-8
            
            while self.calls < N and time_check():
                # Sort
                combined = list(zip(f_vals, simplex))
                combined.sort(key=lambda x: x[0])
                f_vals = [c[0] for c in combined]
                simplex = [c[1] for c in combined]
                
                x_worst = simplex[-1]
                f_worst = f_vals[-1]
                x_best = simplex[0]
                f_best = f_vals[0]
                
                if abs(f_worst - f_best) < tol:
                    break
                    
                centroid = np.sum(simplex[:-1], axis=0) / d
                
                x_r = centroid + alpha * (centroid - x_worst)
                f_r = safe_eval(x_r)
                
                improved = False
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                    no_improve_count = 0
                    improved = True
                else:
                    no_improve_count += 1
                    
                if f_r < f_best:
                    x_e = centroid + gamma * (centroid - x_worst)
                    f_e = safe_eval(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                        if f_e < best_f:
                            best_f = f_e
                            best_x = x_e.copy()
                            no_improve_count = 0
                            improved = True
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    x_c = centroid + rho * (x_worst - centroid)
                    f_c = safe_eval(x_c)
                    
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()
                            no_improve_count = 0
                            improved = True
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            f_vals[i] = safe_eval(simplex[i])
                            if f_vals[i] < best_f:
                                best_f = f_vals[i]
                                best_x = simplex[i].copy()
                                no_improve_count = 0
                                improved = True
                            
                if no_improve_count > max_no_improve:
                    break # Stagnated, break inner loop
                    
        return best_x, best_f
