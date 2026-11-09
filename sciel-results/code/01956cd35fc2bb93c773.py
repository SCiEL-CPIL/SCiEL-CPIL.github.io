"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-start Adaptive Simulated Annealing (ASA) optimizer for BBOB f19 (Griewank-Rosenbrock).
    Strategy:
    1. Use diverse starting points (center + random).
    2. Run ASA from each starting point with temperature-controlled perturbations.
    3. ASA uses deterministic search direction based on temperature and step size.
    4. Final polish pass with fine-tuned ASA on best found point.
    5. Early progress is maximized by quickly converging on good basins.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, keep 30s safety margin
        deadline = start_time + 270.0
        
        dim = self.dim
        budget = self.budget
        
        global_best_x = None
        global_best_f = float('inf')
        calls_made = 0
        
        def call_func(x):
            nonlocal calls_made, global_best_x, global_best_f
            if calls_made >= budget:
                return global_best_f
            val = func(x)
            calls_made += 1
            if val < global_best_f:
                global_best_f = val
                global_best_x = x.copy()
            return val

        def check_time():
            return time.time() > deadline or calls_made >= budget

        # --- Phase 1: Generate diverse starting points ---
        starts = []
        
        # Center
        starts.append(np.zeros(dim))
        
        # Random points for better coverage
        num_random_starts = 15
        for _ in range(num_random_starts):
            x_rand = np.random.uniform(-5, 5, dim)
            starts.append(x_rand)
        
        # Shuffle starts
        np.random.shuffle(starts)
        
        # --- Phase 2: ASA from each start ---
        asa_results = []  # Store (best_f, best_x) from each ASA run
        
        for start_idx, start_x in enumerate(starts):
            if check_time():
                break
            
            # ASA parameters
            # Initial temperature, final temperature, cooling rate
            T_init = 10.0
            T_final = 0.01
            alpha_cool = 0.95  # Cooling rate per iteration
            
            # Initial step size (will be adapted)
            step_size = 1.0
            
            # Current and best solution
            x_current = start_x.copy()
            f_current = call_func(x_current)
            
            x_best = x_current.copy()
            f_best = f_current
            
            # ASA loop
            T = T_init
            max_iter = 80  # Iterations per ASA run
            
            for iter_i in range(max_iter):
                if check_time():
                    break
                
                # Generate random direction
                direction = np.random.randn(dim)
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction /= direction_norm
                
                # ASA step: move in direction scaled by temperature and step size
                # The step magnitude is proportional to T / (T + step_size)
                step_mag = step_size * T / (T + step_size)
                
                x_candidate = x_current + step_mag * direction
                x_candidate = np.clip(x_candidate, -5, 5)
                
                f_candidate = call_func(x_candidate)
                
                if check_time():
                    break
                
                # Metropolis acceptance criterion
                if f_candidate < f_current:
                    # Always accept if better
                    x_current = x_candidate.copy()
                    f_current = f_candidate
                    
                    # Update best
                    if f_candidate < f_best:
                        f_best = f_candidate
                        x_best = x_current.copy()
                else:
                    # Accept with probability exp(-(f_candidate - f_current) / T)
                    delta_f = f_candidate - f_current
                    if delta_f < 1e-10:
                        accept_prob = 1.0
                    else:
                        accept_prob = np.exp(-delta_f / max(T, 1e-10))
                    
                    if np.random.random() < accept_prob:
                        x_current = x_candidate.copy()
                        f_current = f_candidate
                
                # Cool down
                T *= alpha_cool
                
                # Adapt step size based on progress
                # If we found a better solution recently, increase step size slightly
                # Otherwise, decrease it
                if f_current < f_best * 0.99:  # Significant improvement
                    step_size = min(step_size * 1.1, 2.0)
                else:
                    step_size = max(step_size * 0.95, 0.1)
                
                # Check for convergence
                if T < T_final:
                    break
            
            asa_results.append((f_best, x_best.copy()))
        
        # --- Phase 3: Final polish on the best found point ---
        if asa_results and not check_time():
            # Find the best result
            best_idx = np.argmin([r[0] for r in asa_results])
            best_x = asa_results[best_idx][1]
            best_f = asa_results[best_idx][0]
            
            # Run a fine-tuned ASA polish
            T_init_polish = 0.5
            T_final_polish = 0.001
            alpha_cool_polish = 0.98
            step_size_polish = 0.1
            
            x_current = best_x.copy()
            f_current = call_func(x_current)
            
            x_best = x_current.copy()
            f_best = f_current
            
            T = T_init_polish
            max_polish_iter = 150
            
            for iter_i in range(max_polish_iter):
                if check_time():
                    break
                
                # Generate random direction
                direction = np.random.randn(dim)
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction /= direction_norm
                
                # ASA step
                step_mag = step_size_polish * T / (T + step_size_polish)
                
                x_candidate = x_current + step_mag * direction
                x_candidate = np.clip(x_candidate, -5, 5)
                
                f_candidate = call_func(x_candidate)
                
                if check_time():
                    break
                
                # Metropolis acceptance
                if f_candidate < f_current:
                    x_current = x_candidate.copy()
                    f_current = f_candidate
                    
                    if f_candidate < f_best:
                        f_best = f_candidate
                        x_best = x_current.copy()
                else:
                    delta_f = f_candidate - f_current
                    if delta_f < 1e-10:
                        accept_prob = 1.0
                    else:
                        accept_prob = np.exp(-delta_f / max(T, 1e-10))
                    
                    if np.random.random() < accept_prob:
                        x_current = x_candidate.copy()
                        f_current = f_candidate
                
                # Cool down
                T *= alpha_cool_polish
                
                # Adapt step size
                if f_current < f_best * 0.99:
                    step_size_polish = min(step_size_polish * 1.05, 0.5)
                else:
                    step_size_polish = max(step_size_polish * 0.98, 0.01)
                
                # Check for convergence
                if T < T_final_polish:
                    break
        
        # Ensure global_best is set
        if global_best_x is None:
            global_best_x = np.zeros(dim)
            global_best_f = func(global_best_x)
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
