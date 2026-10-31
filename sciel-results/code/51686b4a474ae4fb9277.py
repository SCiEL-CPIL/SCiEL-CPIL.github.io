"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """Pack 21 circles in a perimeter-4 rectangle, maximizing sum of radii using physics-based relaxation."""
    N_CIRCLES = 21
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Rectangle parameters: perimeter = 4, so 2*(w+h) = 4 => w + h = 2
    # We'll optimize over aspect ratio
    
    def compute_energy(xs, ys, rs, w):
        """Compute total energy (penalties) for the configuration."""
        h = 2.0 - w
        if w <= 0.01 or h <= 0.01:
            return 1e10
        
        energy = 0.0
        n = len(xs)
        
        # Wall constraints
        for i in range(n):
            x, y, r = xs[i], ys[i], rs[i]
            if x < r:
                energy += 100.0 * (r - x) ** 2
            if x > w - r:
                energy += 100.0 * (x - (w - r)) ** 2
            if y < r:
                energy += 100.0 * (r - y) ** 2
            if y > h - r:
                energy += 100.0 * (y - (h - r)) ** 2
        
        # Circle-circle overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                min_dist = rs[i] + rs[j]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    overlap = min_dist - dist
                    energy += 100.0 * overlap * overlap
        
        return energy
    
    def is_valid(xs, ys, rs, w, tol=1e-6):
        """Check if configuration is valid (no overlaps, all inside rectangle)."""
        h = 2.0 - w
        if w <= 0.01 or h <= 0.01:
            return False
        
        n = len(xs)
        for i in range(n):
            x, y, r = xs[i], ys[i], rs[i]
            if r <= 0:
                return False
            if x < r - tol or x > w - r + tol:
                return False
            if y < r - tol or y > h - r + tol:
                return False
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True
    
    def get_value(xs, ys, rs, w):
        """Get the objective value (sum of radii) if valid, else None."""
        if not is_valid(xs, ys, rs, w):
            return None
        return np.sum(rs)
    
    # Try multiple aspect ratios and use simulated annealing
    best_value = 0.0
    best_circles = None
    
    np.random.seed(42)
    
    # Aspect ratios to try: w from 0.5 to 1.5 (h = 2 - w)
    aspect_ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    for w in aspect_ratios:
        if time.time() - start_time > time_limit - 10:
            break
        
        h = 2.0 - w
        
        # Initialize with grid-like placement
        # Use a 5x5 grid with 21 circles (last row has 1)
        cols = 5
        rows = 5
        xs = np.zeros(N_CIRCLES)
        ys = np.zeros(N_CIRCLES)
        rs = np.zeros(N_CIRCLES)
        
        for i in range(N_CIRCLES):
            col = i % cols
            row = i // cols
            xs[i] = (col + 0.5) / cols * w
            ys[i] = (row + 0.5) / rows * h
            # Initial radius: fit in cell with margin
            cell_w = w / cols
            cell_h = h / rows
            rs[i] = min(cell_w, cell_h) * 0.48
        
        # Simulated annealing
        temperature = 0.1
        cooling_rate = 0.999
        iterations = 20000
        step_size = 0.05
        
        current_energy = compute_energy(xs, ys, rs, w)
        best_local_energy = current_energy
        best_local_xs = xs.copy()
        best_local_ys = ys.copy()
        best_local_rs = rs.copy()
        
        for it in range(iterations):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Perturb random circle
            idx = np.random.randint(N_CIRCLES)
            
            # Decide what to perturb: position or radius
            if np.random.random() < 0.7:
                # Perturb position
                dx = np.random.normal(0, step_size)
                dy = np.random.normal(0, step_size)
                xs_new = xs.copy()
                ys_new = ys.copy()
                rs_new = rs.copy()
                xs_new[idx] = np.clip(xs[idx] + dx, 0.01, w - 0.01)
                ys_new[idx] = np.clip(ys[idx] + dy, 0.01, h - 0.01)
            else:
                # Perturb radius
                dr = np.random.normal(0, step_size * 0.5)
                rs_new = rs.copy()
                rs_new[idx] = np.clip(rs[idx] + dr, 0.01, min(w, h) / 2 - 0.01)
                xs_new = xs.copy()
                ys_new = ys.copy()
            
            new_energy = compute_energy(xs_new, ys_new, rs_new, w)
            delta = new_energy - current_energy
            
            if delta < 0 or np.random.random() < np.exp(-delta / max(temperature, 1e-10)):
                xs, ys, rs = xs_new, ys_new, rs_new
                current_energy = new_energy
                
                if current_energy < best_local_energy:
                    best_local_energy = current_energy
                    best_local_xs = xs.copy()
                    best_local_ys = ys.copy()
                    best_local_rs = rs.copy()
            
            temperature *= cooling_rate
            step_size *= 0.9999
        
        # Check if best local solution is valid
        val = get_value(best_local_xs, best_local_ys, best_local_rs, w)
        if val is not None and val > best_value:
            best_value = val
            best_circles = np.column_stack([best_local_xs, best_local_ys, best_local_rs])
    
    # If no valid solution found, return a simple fallback
    if best_circles is None:
        # Simple 5x5 grid with 21 circles
        cols = 5
        rows = 5
        w = 1.0
        h = 1.0
        circles = []
        for i in range(N_CIRCLES):
            col = i % cols
            row = i // cols
            x = (col + 0.5) / cols * w
            y = (row + 0.5) / rows * h
            r = min(w / cols, h / rows) * 0.45
            circles.append([x, y, r])
        best_circles = np.array(circles)
    
    return best_circles
# EVOLVE-BLOCK-END
