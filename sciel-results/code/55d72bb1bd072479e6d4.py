import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin from 330s limit
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Initialize population with diverse points using stratified sampling
        pop_size = 50  # Larger population for better exploration
        points = []
        values = []
        
        # Center point
        x_center = np.zeros(self.dim)
        f_center = func(x_center)
        points.append(x_center)
        values.append(f_center)
        
        # Generate initial population with stratified sampling for better coverage
        for _ in range(pop_size - 1):
            # Use a mix of random and structured points
            if rng.random() < 0.3:
                # Structured: sample from a grid-like pattern
                x_struct = rng.uniform(-4, 4, self.dim)
                # Add some randomness
                x_struct += rng.normal(0, 0.5, self.dim)
                x_rand = np.clip(x_struct, -5, 5)
            else:
                # Random sampling
                x_rand = rng.uniform(-5, 5, self.dim)
            
            f_rand = func(x_rand)
            points.append(x_rand)
            values.append(f_rand)
        
        calls_used = len(points)
        best_f = min(values)
        best_x = points[np.argmin(values)]
        
        # Phase 2: Differential Evolution with adaptive parameters
        # DE parameters
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        
        # Track convergence to adapt parameters
        prev_best_f = best_f
        stagnation_count = 0
        
        while calls_used < self.budget - 20:  # Reserve calls for final refinement
            if time.time() > deadline - 10:
                break
            
            # Sort population by fitness
            sorted_indices = np.argsort(values)
            sorted_points = [points[i] for i in sorted_indices]
            sorted_values = [values[i] for i in sorted_indices]
            
            # Update best
            if sorted_values[0] < best_f:
                best_f = sorted_values[0]
                best_x = sorted_points[0]
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Adapt parameters based on stagnation
            if stagnation_count > 10:
                F = max(0.4, F * 0.9)  # Reduce mutation strength
                CR = min(0.95, CR * 1.05)  # Increase crossover rate
                if stagnation_count > 30:
                    # Restart some individuals
                    n_restart = pop_size // 5
                    for _ in range(n_restart):
                        idx = rng.integers(pop_size)
                        points[idx] = rng.uniform(-5, 5, self.dim)
                        values[idx] = func(points[idx])
                        calls_used += 1
                        if calls_used >= self.budget - 20:
                            break
                    stagnation_count = 0
            
            # Generate offspring for each individual
            new_points = []
            new_values = []
            
            for i in range(pop_size):
                if calls_used >= self.budget - 20:
                    break
                if time.time() > deadline - 10:
                    break
                
                # Select three distinct random indices different from i
                while True:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                    if len({r1, r2, r3, i}) == 4:
                        break
                
                # DE/rand/1 mutation
                mutant = points[r1] + F * (points[r2] - points[r3])
                
                # Binary crossover
                trial = np.copy(points[i])
                j_rand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_used += 1
                
                # Selection: keep better of parent and trial
                if f_trial < values[i]:
                    new_points.append(trial)
                    new_values.append(f_trial)
                else:
                    new_points.append(points[i])
                    new_values.append(values[i])
            
            if new_points:
                points = new_points
                values = new_values
        
        # Phase 3: Local refinement from best point
        remaining_budget = self.budget - calls_used
        if remaining_budget > 10 and time.time() < deadline - 5:
            try:
                res = minimize(
                    func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxfev': remaining_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        return best_x, best_f
