"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        for i in range(N):
            if xs[i] - rs[i] < -tol or xs[i] + rs[i] > 1 + tol:
                return False
            if ys[i] - rs[i] < -tol or ys[i] + rs[i] > 1 + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True
    
    def objective_value(x):
        """Negative of sum of radii if valid, else large penalty."""
        if not is_valid(x):
            return 1e10
        return -np.sum(x[:, 2])
    
    def simulated_annealing(x_init, max_iters=50000, T_start=0.05, T_end=0.001, seed=42):
        """Simulated annealing with pairwise repulsion forces."""
        rng = np.random.RandomState(seed)
        x = x_init.copy()
        best_x = x.copy()
        best_val = objective_value(x)
        
        # Precompute initial
        T = T_start
        
        for it in range(max_iters):
            # Check time
            if time.time() > deadline - 3:
                break
            
            # Temperature schedule
            T = T_start * (T_end / T_start) ** (it / max_iters)
            
            # Generate perturbation: move random circles
            n_moves = rng.randint(1, 4)
            x_new = x.copy()
            for _ in range(n_moves):
                idx = rng.randint(N)
                # Move position
                dx = rng.normal(0, T * 0.5)
                dy = rng.normal(0, T * 0.5)
                x_new[idx, 0] += dx
                x_new[idx, 1] += dy
                # Optionally adjust radius
                if rng.random() < 0.3:
                    dr = rng.normal(0, T * 0.3)
                    x_new[idx, 2] = max(0.001, min(0.5, x_new[idx, 2] + dr))
            
            # Clamp to bounds
            for i in range(N):
                r = x_new[i, 2]
                x_new[i, 0] = max(r, min(1 - r, x_new[i, 0]))
                x_new[i, 1] = max(r, min(1 - r, x_new[i, 1]))
            
            val_new = objective_value(x_new)
            val_curr = objective_value(x)
            
            # Accept if better or with Boltzmann probability
            if val_new < val_curr or (val_curr < 1e9 and rng.random() < np.exp(-(val_new - val_curr) / T)):
                x = x_new
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
            
            # Periodic cooling boost: occasionally kick with larger moves
            if it % 5000 == 0 and it > 0:
                # Try a larger random perturbation
                for _ in range(3):
                    idx = rng.randint(N)
                    x[idx, 0] += rng.normal(0, 0.02)
                    x[idx, 1] += rng.normal(0, 0.02)
                    for i in range(N):
                        r = x[i, 2]
                        x[i, 0] = max(r, min(1 - r, x[i, 0]))
                        x[i, 1] = max(r, min(1 - r, x[i, 1]))
                    val_kick = objective_value(x)
                    if val_kick < best_val:
                        best_val = val_kick
                        best_x = x.copy()
        
        return best_x, best_val
    
    def initial_grid():
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    def random_init(seed=0):
        rng = np.random.RandomState(seed)
        x = np.zeros((N, 3))
        # Place circles with small radii randomly
        for i in range(N):
            r = rng.uniform(0.01, 0.05)
            x[i, 0] = rng.uniform(r, 1 - r)
            x[i, 1] = rng.uniform(r, 1 - r)
            x[i, 2] = r
        return x
    
    # Strategy: multiple SA restarts with different initializations
    best_overall = None
    best_score = -np.inf
    
    # Start with grid initialization
    x_grid = initial_grid()
    x_best, val = simulated_annealing(x_grid, max_iters=30000, T_start=0.04, T_end=0.0005, seed=1)
    if val < best_score:
        best_score = val
        best_overall = x_best.copy()
    
    # Multiple random restarts
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        seed = 100 + restart * 7
        x_rand = random_init(seed)
        # Give random init slightly larger radii to start
        x_rand[:, 2] = np.clip(x_rand[:, 2] * 1.5, 0.01, 0.1)
        # Clamp positions
        for i in range(N):
            r = x_rand[i, 2]
            x_rand[i, 0] = max(r, min(1 - r, x_rand[i, 0]))
            x_rand[i, 1] = max(r, min(1 - r, x_rand[i, 1]))
        
        # Vary SA parameters per restart
        T_s = 0.03 + 0.01 * (restart % 3)
        T_e = 0.0003 + 0.0001 * (restart % 2)
        max_it = 25000 + 5000 * (restart % 2)
        
        x_best_r, val_r = simulated_annealing(x_rand, max_iters=max_it, T_start=T_s, T_end=T_e, seed=seed + 1000)
        if val_r < best_score:
            best_score = val_r
            best_overall = x_best_r.copy()
    
    # Final local refinement: use gradient-free coordinate descent on best solution
    if best_overall is not None and time.time() < deadline - 5:
        x_ref = best_overall.copy()
        # Coordinate descent: try adjusting each circle's position/radius
        for sweep in range(50):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                # Try small perturbations in x, y, r
                for dim in range(3):
                    for delta in [0.001, -0.001, 0.002, -0.002, 0.0005, -0.0005]:
                        x_try = x_ref.copy()
                        x_try[i, dim] += delta
                        # Clamp
                        r = x_try[i, 2]
                        if r < 0.001 or r > 0.5:
                            continue
                        x_try[i, 0] = max(r, min(1 - r, x_try[i, 0]))
                        x_try[i, 1] = max(r, min(1 - r, x_try[i, 1]))
                        x_try[i, 2] = max(0.001, min(0.5, x_try[i, 2]))
                        
                        if is_valid(x_try):
                            val_try = -np.sum(x_try[:, 2])
                            if val_try < -np.sum(x_ref[:, 2]):
                                x_ref = x_try
                                improved = True
                                break
            if not improved:
                break
        
        if -np.sum(x_ref[:, 2]) > -np.sum(best_overall[:, 2]):
            best_overall = x_ref
    
    # Ensure validity of final result
    if best_overall is None or not is_valid(best_overall):
        return initial_grid()
    
    return best_overall

# EVOLVE-BLOCK-END
