"""
Circle packing in perimeter-4 rectangle: 21 circles.
Uses a force-directed relaxation with radius growth:
1. Initialize circles in a grid pattern.
2. Use repulsive forces to resolve overlaps and keep circles inside the rectangle.
3. Iteratively grow radii while maintaining validity.
4. Optimize aspect ratio.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def check_validity(x, y, r, w, tol=1e-9):
        h = 2.0 - w
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum * r_sum - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    def relax(x, y, r, w, n_iter=2000, dt=0.001):
        """
        Force-directed relaxation to resolve overlaps and boundary violations.
        Returns updated x, y.
        """
        h = 2.0 - w
        x = x.copy()
        y = y.copy()
        
        # Adaptive step size
        for it in range(n_iter):
            if time.time() > deadline - 5.0:
                break
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Repulsive forces between circles
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d2 = dx * dx + dy * dy
                    d = np.sqrt(d2) if d2 > 1e-12 else 1e-6
                    r_sum = r[i] + r[j]
                    if d < r_sum:
                        # Overlap: push apart
                        overlap = r_sum - d
                        # Force proportional to overlap
                        f = overlap * 10.0
                        if d > 1e-12:
                            fx[i] += f * dx / d
                            fy[i] += f * dy / d
                            fx[j] -= f * dx / d
                            fy[j] -= f * dy / d
                        else:
                            # Random small push
                            fx[i] += np.random.randn() * 0.01
                            fy[i] += np.random.randn() * 0.01
                            fx[j] -= np.random.randn() * 0.01
                            fy[j] -= np.random.randn() * 0.01
            
            # Boundary forces
            for i in range(N):
                # Left wall
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 10.0
                # Right wall
                if x[i] > w - r[i]:
                    fx[i] -= (x[i] - (w - r[i])) * 10.0
                # Bottom wall
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 10.0
                # Top wall
                if y[i] > h - r[i]:
                    fy[i] -= (y[i] - (h - r[i])) * 10.0
            
            # Update positions
            x += dt * fx
            y += dt * fy
            
            # Clamp to boundaries (hard constraint)
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
            
            # Check if converged (no overlaps and no boundary violations)
            max_violation = 0.0
            for i in range(N):
                max_violation = max(max_violation, r[i] - x[i], x[i] - (w - r[i]), r[i] - y[i], y[i] - (h - r[i]))
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    r_sum = r[i] + r[j]
                    if d < r_sum:
                        max_violation = max(max_violation, r_sum - d)
            
            if max_violation < 1e-8:
                break
            
            # Decay step size if not converging
            if it > 500 and it % 100 == 0:
                dt *= 0.95
        
        return x, y

    def grow_and_relax(x, y, r, w, growth_steps=50, growth_factor=0.01):
        """
        Iteratively grow radii and relax to maintain validity.
        Returns best x, y, r found.
        """
        best_sum_r = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        for step in range(growth_steps):
            if time.time() > deadline - 5.0:
                break
            
            # Try to grow all radii uniformly
            new_r = r * (1.0 + growth_factor)
            
            # Check if new radii are feasible after relaxation
            new_x, new_y = relax(x, y, new_r, w, n_iter=1000)
            
            # Check validity
            valid, val = check_validity(new_x, new_y, new_r, w)
            if valid and val > best_sum_r:
                best_sum_r = val
                best_x, best_y, best_r = new_x.copy(), new_y.copy(), new_r.copy()
                x, y, r = new_x, new_y, new_r
            else:
                # Try smaller growth
                growth_factor *= 0.5
                if growth_factor < 1e-5:
                    break
        
        return best_x, best_y, best_r, best_sum_r

    def optimize_aspect_ratio(x, y, r, w_init, n_trials=10):
        """
        Try different aspect ratios to find a better packing.
        """
        best_val = 0.0
        best_result = None
        
        for trial in range(n_trials):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb aspect ratio
            w = w_init * (1.0 + np.random.randn() * 0.1)
            w = np.clip(w, 0.5, 1.5)
            
            # Scale positions to new rectangle
            h_old = 2.0 - w_init
            h_new = 2.0 - w
            x_new = x * (w / w_init)
            y_new = y * (h_new / h_old)
            
            # Relax to resolve overlaps
            x_new, y_new = relax(x_new, y_new, r, w, n_iter=1000)
            
            # Check validity
            valid, val = check_validity(x_new, y_new, r, w)
            if valid and val > best_val:
                best_val = val
                best_result = (x_new, y_new, r.copy(), w)
        
        return best_result, best_val

    # Initialize: 3x7 grid in a 1x1 square
    cols, rows = 3, 7
    w = 1.0
    h = 1.0
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    cell_w = w / cols
    cell_h = h / rows
    r_init = min(cell_w, cell_h) * 0.45  # Start with slightly smaller circles
    
    for i in range(N):
        row, col = divmod(i, cols)
        x[i] = (col + 0.5) * cell_w
        y[i] = (row + 0.5) * cell_h
        r[i] = r_init
    
    # Relax initial configuration
    x, y = relax(x, y, r, w, n_iter=500)
    
    # Grow radii
    x, y, r, val = grow_and_relax(x, y, r, w, growth_steps=100, growth_factor=0.02)
    
    # Optimize aspect ratio
    result, val2 = optimize_aspect_ratio(x, y, r, w, n_trials=20)
    if result is not None:
        x, y, r, w = result
        val = val2
    
    # Final validity check
    valid, final_val = check_validity(x, y, r, w)
    if not valid:
        # Fallback: shrink radii slightly
        r = r * 0.95
        x, y = relax(x, y, r, w, n_iter=500)
        valid, final_val = check_validity(x, y, r, w)
        if not valid:
            # Very conservative fallback
            r = r * 0.9
            x, y = relax(x, y, r, w, n_iter=500)
    
    # Shrink slightly to ensure strict inequality
    r_final = r * (1 - 1e-9)
    return np.column_stack((x, y, r_final))

if __name__ == "__main__":
    res = construct()
    print(res.shape)
