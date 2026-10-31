"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def construct():
    """Pack 21 circles in a perimeter-4 rectangle to maximize sum of radii."""
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit

    best_val = 0.0
    best_result = None

    # Perimeter 4 => 2*(w+h) = 4 => w+h = 2
    # We test a range of aspect ratios w/h
    # Square (1.0) is a good starting point, but rectangular might allow better packing
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33]

    # Helper function to generate hexagonal-like initial positions
    def get_hex_init(w, h, n, seed=0):
        """
        Generates initial positions for n circles in a w x h rectangle.
        Uses a hexagonal lattice pattern scaled to fit.
        """
        np.random.seed(seed)
        # Estimate number of columns
        # Area of rectangle = w*h
        # Approximate circle radius r ~ sqrt(w*h / (n * pi * 1.1)) (1.1 for packing efficiency)
        # But we just want a good spread.
        
        # Try to fit a hex grid
        # Hex grid spacing: dx = 2r, dy = sqrt(3)r
        # Let's just use a jittered grid for robustness and simplicity, 
        # but with a bias towards hexagonal spacing if possible.
        
        # Simple robust init: Grid with jitter
        # Choose cols and rows to approximate aspect ratio
        cols = int(np.round(np.sqrt(n * w / h)))
        if cols < 1: cols = 1
        rows = int(np.ceil(n / cols))
        
        # Adjust if grid is too small
        while cols * rows < n:
            if w > h:
                cols += 1
            else:
                rows += 1
        
        positions = []
        idx = 0
        for r_idx in range(rows):
            for c_idx in range(cols):
                if idx >= n:
                    break
                # Center of cell
                x = (c_idx + 0.5) * w / cols
                y = (r_idx + 0.5) * h / rows
                
                # Add small jitter to avoid symmetry locks
                jitter_x = np.random.uniform(-0.1, 0.1) * (w / cols)
                jitter_y = np.random.uniform(-0.1, 0.1) * (h / rows)
                
                x += jitter_x
                y += jitter_y
                
                # Clamp to stay inside with a small margin
                x = np.clip(x, 0.05, w - 0.05)
                y = np.clip(y, 0.05, h - 0.05)
                
                positions.append([x, y])
                idx += 1
        
        return np.array(positions[:n])

    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        # w/h = ar => w = ar*h => ar*h + h = 2 => h = 2/(1+ar), w = 2*ar/(1+ar)
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Try a few restarts with different seeds for initialization
        max_restarts = 3
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            positions = get_hex_init(w, h, N, seed=restart*10+int(ar*100))
            
            # Initial radii: estimate based on min distance to neighbors and walls
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
                    min_dist = min(min_dist, d)
            for i in range(N):
                min_dist = min(min_dist, positions[i,0], w-positions[i,0], positions[i,1], h-positions[i,1])
            
            # Start with radii slightly smaller than half the min distance
            r0 = 0.4 * min_dist
            radii = np.full(N, r0)
            
            # Flatten variables: [x0, y0, r0, x1, y1, r1, ...]
            x0 = np.zeros(3*N)
            for i in range(N):
                x0[3*i] = positions[i, 0]
                x0[3*i+1] = positions[i, 1]
                x0[3*i+2] = radii[i]
            
            # Objective: Maximize sum of radii => Minimize -sum(radii)
            def objective(v):
                # v[2::3] are the radii
                return -np.sum(v[2::3])
            
            # Constraints
            # 1. Inside rectangle:
            #    x - r >= 0
            #    w - x - r >= 0
            #    y - r >= 0
            #    h - y - r >= 0
            def cons_inside(v):
                x = v[0::3]
                y = v[1::3]
                r = v[2::3]
                return np.concatenate([
                    x - r,
                    w - x - r,
                    y - r,
                    h - y - r
                ])
            
            # 2. Non-overlap:
            #    dist(i,j) - r_i - r_j >= 0
            #    dist = sqrt((xi-xj)^2 + (yi-yj)^2)
            #    To avoid sqrt in constraint for speed, we can square both sides if positive,
            #    but SLSQP handles non-linear constraints. Sqrt is fine for N=21 (210 pairs).
            def cons_non_overlap(v):
                x = v[0::3]
                y = v[1::3]
                r = v[2::3]
                cons = []
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        cons.append(dist - r[i] - r[j])
                return np.array(cons)
            
            # 3. Positive radius
            def cons_positive_r(v):
                return v[2::3]
            
            constraints = [
                {'type': 'ineq', 'fun': cons_inside},
                {'type': 'ineq', 'fun': cons_non_overlap},
                {'type': 'ineq', 'fun': cons_positive_r}
            ]
            
            # Run SLSQP
            try:
                result = minimize(
                    objective, x0, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 150, 'ftol': 1e-8}
                )
                
                if result.success or (result.fun < 0 and result.fun < -0.1): # Check if it improved
                    v = result.x
                    x_res = v[0::3]
                    y_res = v[1::3]
                    r_res = v[2::3]
                    
                    # Validation
                    valid = True
                    if np.any(r_res <= 0):
                        valid = False
                    else:
                        # Check boundaries
                        if np.any(x_res < r_res - 1e-5) or np.any(x_res > w - r_res + 1e-5):
                            valid = False
                        if np.any(y_res < r_res - 1e-5) or np.any(y_res > h - r_res + 1e-5):
                            valid = False
                        else:
                            # Check overlaps
                            for i in range(N):
                                for j in range(i+1, N):
                                    dist = np.sqrt((x_res[i]-x_res[j])**2 + (y_res[i]-y_res[j])**2)
                                    if dist < r_res[i] + r_res[j] - 1e-5:
                                        valid = False
                                        break
                                if not valid:
                                    break
                    
                    if valid:
                        total_r = np.sum(r_res)
                        if total_r > best_val:
                            best_val = total_r
                            best_result = np.column_stack((x_res, y_res, r_res)).copy()
                            
            except Exception:
                continue

    # Fallback: Simple grid if optimization failed
    if best_result is None:
        # 5x5 grid in 1x1 square (perimeter 4)
        # 21 circles.
        # Radius approx 0.1
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.01 # Slightly smaller to ensure no overlap
                    cells.append([x, y, r])
        best_result = np.array(cells[:N])
        best_val = np.sum(best_result[:, 2])

    return best_result

# EVOLVE-BLOCK-END
