import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 110.0  # Safety margin from 120s limit

def construct():
    start_time = time.time()
    
    # Baseline: Grid packing
    side = int(np.ceil(np.sqrt(N)))
    grid_x = []
    grid_y = []
    grid_r = []
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            grid_x.append(x)
            grid_y.append(y)
            grid_r.append(r)
            idx += 1
            
    best_x = np.array(grid_x)
    best_y = np.array(grid_y)
    best_r = np.array(grid_r)
    best_val = np.sum(best_r)
    
    def check_validity(x, y, r, tol=1e-9):
        # Check boundary
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
            return False
        if np.any(r < -tol):
            return False
        
        # Check non-overlap
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        
        # Ignore self
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)
        
        if np.any(dist < min_dist - tol):
            return False
        return True

    def compute_energy(x, y, r):
        """Compute total energy: sum of radii + penalty for overlaps and boundary violations"""
        energy = -np.sum(r)  # We want to maximize sum(r), so minimize -sum(r)
        
        # Boundary penalty
        boundary_violations = np.maximum(0, r - x) + np.maximum(0, r - (1 - x)) + \
                              np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
        energy += 100.0 * np.sum(boundary_violations**2)
        
        # Overlap penalty
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)
        
        violations = np.maximum(0, min_dist - dist)
        energy += 100.0 * np.sum(violations**2)
        
        return energy

    def simulated_annealing(x0, y0, r0, max_steps=5000, T_initial=0.05, T_final=1e-6, cooling_rate=0.999):
        """Simulated annealing to optimize circle positions"""
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        
        current_energy = compute_energy(x, y, r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_energy = current_energy
        
        T = T_initial
        rng = np.random.RandomState(123)
        
        for step in range(max_steps):
            if time.time() > DEADLINE - 5:
                break
            
            # Generate random perturbation
            idx = rng.randint(N)
            step_size = 0.01 * T / T_initial
            
            # Perturb position
            dx = rng.normal(0, step_size)
            dy = rng.normal(0, step_size)
            
            # Perturb radius slightly
            dr = rng.normal(0, step_size * 0.5)
            
            x_new = x.copy()
            y_new = y.copy()
            r_new = r.copy()
            
            x_new[idx] = np.clip(x_new[idx] + dx, 0, 1)
            y_new[idx] = np.clip(y_new[idx] + dy, 0, 1)
            r_new[idx] = np.clip(r_new[idx] + dr, 0.01, 0.3)
            
            new_energy = compute_energy(x_new, y_new, r_new)
            
            # Accept or reject
            delta = new_energy - current_energy
            if delta < 0 or rng.random() < np.exp(-delta / T):
                x, y, r = x_new, y_new, r_new
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            
            # Cool down
            T *= cooling_rate
            T = max(T, T_final)
        
        return best_x, best_y, best_r, -best_energy  # Return negative energy as value

    # Run simulated annealing with multiple restarts
    rng = np.random.RandomState(42)
    
    # Initial guess: spread circles in a grid-like pattern with some randomness
    n_per_side = int(np.ceil(np.sqrt(N)))
    for restart in range(8):
        if time.time() > DEADLINE - 10:
            break
        
        # Create initial positions
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        idx = 0
        for i in range(n_per_side):
            for j in range(n_per_side):
                if idx >= N:
                    break
                x0[idx] = (i + 0.5 + rng.uniform(-0.1, 0.1)) / n_per_side
                y0[idx] = (j + 0.5 + rng.uniform(-0.1, 0.1)) / n_per_side
                r0[idx] = 0.5 / n_per_side * rng.uniform(0.8, 1.0)
                idx += 1
        
        # Run simulated annealing
        x_sa, y_sa, r_sa, val_sa = simulated_annealing(x0, y0, r0)
        
        # Check validity
        if check_validity(x_sa, y_sa, r_sa):
            if val_sa > best_val:
                best_x, best_y, best_r, best_val = x_sa, y_sa, r_sa, val_sa
        else:
            # Try to repair by scaling radii
            for scale in [0.99, 0.98, 0.95, 0.90, 0.85]:
                r_test = r_sa * scale
                if check_validity(x_sa, y_sa, r_test):
                    val_test = np.sum(r_test)
                    if val_test > best_val:
                        best_x, best_y, best_r, best_val = x_sa, y_sa, r_test, val_test
                    break
    
    # Final safety check
    if not check_validity(best_x, best_y, best_r):
        # Aggressive repair
        for scale in np.linspace(0.99, 0.80, 10):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            # Fallback to seed
            best_x = np.array(grid_x)
            best_y = np.array(grid_y)
            best_r = np.array(grid_r)

    return np.column_stack((best_x, best_y, best_r))
