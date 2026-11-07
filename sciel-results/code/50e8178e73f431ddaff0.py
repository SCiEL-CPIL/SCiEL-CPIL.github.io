import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        calls = 0

        def count_call(x):
            nonlocal calls
            calls += 1
            return func(x)

        # Line search along direction d
        def line_search(x, d, f_val, remaining_budget):
            nonlocal calls
            # 1. Coarse random search
            norm_d = np.linalg.norm(d)
            if norm_d < 1e-10:
                return x, f_val, 0
            
            d_unit = d / norm_d
            candidates = [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0]
            
            best_alpha = 0.0
            best_f = f_val
            best_x = x
            local_calls = 0
            
            # Sample up to 8 candidates
            for alpha in candidates:
                if calls + 1 > self.budget:
                    break
                x_trial = np.clip(x + alpha * d_unit, -5, 5)
                f_trial = count_call(x_trial)
                local_calls += 1
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial
                    best_alpha = alpha
            
            # 2. Refinement (Golden Section) if improved
            if best_alpha != 0.0 and calls + 10 <= self.budget:
                # Find interval [a, b]
                a = best_alpha - 0.5
                b = best_alpha + 0.5
                
                x_a = np.clip(x + a * d_unit, -5, 5)
                x_b = np.clip(x + b * d_unit, -5, 5)
                f_a = count_call(x_a)
                f_b = count_call(x_b)
                local_calls += 2
                
                # Expand if needed
                for _ in range(2):
                    if f_a < best_f or f_b < best_f:
                        a -= 1.0
                        b += 1.0
                        x_a = np.clip(x + a * d_unit, -5, 5)
                        x_b = np.clip(x + b * d_unit, -5, 5)
                        f_a = count_call(x_a)
                        f_b = count_call(x_b)
                        local_calls += 2
                    else:
                        break
                
                # Golden Section
                gr = (np.sqrt(5) - 1) / 2
                x1 = np.clip(x + (b - gr * (b - a)) * d_unit, -5, 5)
                x2 = np.clip(x + (a + gr * (b - a)) * d_unit, -5, 5)
                f1 = count_call(x1)
                f2 = count_call(x2)
                local_calls += 2
                
                for _ in range(4):
                    if f1 < f2:
                        b = x2
                        f_b = f2
                        x2 = x1
                        f2 = f1
                        x1 = np.clip(x + (b - gr * (b - a)) * d_unit, -5, 5)
                        f1 = count_call(x1)
                    else:
                        a = x1
                        f_a = f1
                        x1 = x2
                        f1 = f2
                        x2 = np.clip(x + (a + gr * (b - a)) * d_unit, -5, 5)
                        f2 = count_call(x2)
                    local_calls += 1
                    if abs(b - a) < 0.05:
                        break
                
                # Final check at midpoint
                x_mid = np.clip(x + ((a + b) / 2) * d_unit, -5, 5)
                f_mid = count_call(x_mid)
                local_calls += 1
                
                if f_mid < best_f:
                    best_x = x_mid
                    best_f = f_mid
            
            return best_x, best_f, local_calls

        # Main Loop: Powell's Method
        x = np.zeros(self.dim)
        f_x = count_call(x)
        
        D = np.eye(self.dim)
        
        while calls < self.budget:
            x_old = x.copy()
            f_old = f_x
            
            max_improvement = 0
            worst_dir_idx = 0
            
            for i in range(self.dim):
                if calls + 15 > self.budget: # Reserve some budget
                    break
                
                d = D[i]
                x, f_x, used = line_search(x, d, f_x, self.budget - calls)
                
                improvement = f_old - f_x
                if improvement > max_improvement:
                    max_improvement = improvement
                    worst_dir_idx = i
            
            # Update Direction
            if max_improvement > 1e-6:
                # Replace worst direction with the vector of movement
                new_dir = x - x_old
                # Keep the set of directions roughly orthogonal? 
                # Powell's standard update: shift directions and replace the worst one.
                D = np.roll(D, -1, axis=1) # Shift columns? No, D is (dim, dim)
                # Standard Powell:
                # d_1, ..., d_n
                # After cycle, if d_k was worst, new set is d_{k+1}, ..., d_n, d_{new}
                # Let's just replace D[worst_dir_idx] with new_dir
                D[worst_dir_idx] = new_dir
                
                # Renormalize?
                # Not strictly necessary for Powell, but helps with numerical stability
                # We'll let the line search handle scaling via d_unit
            
            else:
                # Stagnation: Restart
                if calls + 50 > self.budget:
                    break
                
                # Random Restart
                x = rng.uniform(-5, 5, self.dim)
                f_x = count_call(x)
                D = np.eye(self.dim)
        
        return x, f_x
