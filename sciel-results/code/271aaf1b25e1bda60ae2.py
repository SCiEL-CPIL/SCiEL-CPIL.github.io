"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import random
import math

def construct():
    """
    Pack 26 circles in the unit square maximizing sum of radii.
    Uses a vectorized Simulated Annealing approach.
    """
    n = N
    
    def is_valid(x, y, r, tol=1e-9):
        # Vectorized boundary check
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
            return False
        
        # Vectorized overlap check
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        r_sum = r[:, None] + r[None, :]
        
        # Check upper triangle
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        if np.any(dist[mask] < r_sum[mask] - tol):
            return False
        return True

    def compute_energy(x, y, r, w_overlap=10.0, w_bound=10.0):
        e = -np.sum(r)
        
        # Overlap penalty
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        r_sum = r[:, None] + r[None, :]
        
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        overlaps = np.maximum(0, r_sum - dist)
        e += w_overlap * np.sum(overlaps[mask] ** 2)
        
        # Boundary penalty
        left_viol = np.maximum(0, r - x)
        right_viol = np.maximum(0, x - (1 - r))
        bottom_viol = np.maximum(0, r - y)
        top_viol = np.maximum(0, y - (1 - r))
        
        e += w_bound * (np.sum(left_viol**2) + np.sum(right_viol**2) + 
                    np.sum(bottom_viol**2) + np.sum(top_viol**2))
                    
        return e

    best_value = 0.0
    best_x = None
    best_y = None
    best_r = None
    
    n_restarts = 15
    max_iterations = 8000
    
    for restart in range(n_restarts):
        # Initialize with random positions and small radii
        x = np.random.uniform(0.1, 0.9, n)
        y = np.random.uniform(0.1, 0.9, n)
        r = np.full(n, 0.03)
        
        current_energy = compute_energy(x, y, r)
        
        local_best_val = 0.0
        local_best_x = x.copy()
        local_best_y = y.copy()
        local_best_r = r.copy()
        
        if is_valid(x, y, r):
            local_best_val = np.sum(r)
            
        T = 0.5
        T_min = 1e-5
        cooling_rate = 0.9995
        
        # Adaptive steps
        step_r = 0.01
        step_pos = 0.02
        
        for it in range(max_iterations):
            # Generate move
            move_type = random.random()
            
            if move_type < 0.4:
                # Change radius
                idx = random.randint(0, n-1)
                old_r = r[idx]
                delta = random.uniform(-0.005, 0.015)
                new_r = max(0.01, old_r + delta)
                r[idx] = new_r
            else:
                # Change position
                idx = random.randint(0, n-1)
                old_x = x[idx]
                old_y = y[idx]
                dx = random.uniform(-step_pos, step_pos)
                dy = random.uniform(-step_pos, step_pos)
                x[idx] = old_x + dx
                y[idx] = old_y + dy
            
            new_energy = compute_energy(x, y, r)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                current_energy = new_energy
                
                # Check validity for best tracking
                if is_valid(x, y, r, tol=1e-8):
                    val = np.sum(r)
                    if val > local_best_val:
                        local_best_val = val
                        local_best_x = x.copy()
                        local_best_y = y.copy()
                        local_best_r = r.copy()
            else:
                # Revert
                if move_type < 0.4:
                    r[idx] = old_r
                else:
                    x[idx] = old_x
                    y[idx] = old_y
            
            T *= cooling_rate
            if T < T_min:
                break
        
        # Local Polish: Try to increase radii of the smallest circles
        # Sort by radius
        order = np.argsort(local_best_r)
        for idx in order:
            # Try to increase radius
            for _ in range(100):
                old_r = local_best_r[idx]
                new_r = old_r + 0.001
                if new_r > 0.5: break
                local_best_r[idx] = new_r
                if is_valid(local_best_x, local_best_y, local_best_r, tol=1e-8):
                    local_best_val = np.sum(local_best_r)
                else:
                    local_best_r[idx] = old_r
                    break

        if local_best_val > best_value:
            best_value = local_best_val
            best_x = local_best_x
            best_y = local_best_y
            best_r = local_best_r

    if best_x is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    cr = 0.5 / side - 1e-4
                    cells.append([cx, cy, cr])
        best_x = np.array([c[0] for c in cells[:N]])
        best_y = np.array([c[1] for c in cells[:N]])
        best_r = np.array([c[2] for c in cells[:N]])

    return np.column_stack([best_x, best_y, best_r])
# EVOLVE-BLOCK-END
