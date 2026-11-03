"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist_sq = dx * dx + dy * dy
            min_dist = circles[i][2] + circles[j][2]
            if dist_sq < (min_dist - tol) ** 2:
                return False
    return True

def _total_radius(circles):
    return sum(c[2] for c in circles)

def _relax_circles(circles, w, h, iterations=2000, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    Uses vectorized overlap resolution for speed.
    """
    n = len(circles)
    pos = np.array([c[:2] for c in circles], dtype=np.float64)
    rad = np.array([c[2] for c in circles], dtype=np.float64)
    
    for _ in range(iterations):
        moved = False
        
        # 1. Resolve boundary violations
        # Push circles inside if they are outside
        pos[:, 0] = np.clip(pos[:, 0], rad, w - rad)
        pos[:, 1] = np.clip(pos[:, 1], rad, h - rad)
        
        # Check if any were actually moved (approximate check for convergence)
        # We'll rely on the overlap loop to determine 'moved' status more strictly
        
        # 2. Resolve overlaps
        # Vectorized pairwise distance calculation
        # diff[i, j] = pos[j] - pos[i]
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        min_dist = rad[:, np.newaxis] + rad[np.newaxis, :]
        
        # Identify overlaps where dist < min_dist
        # We need to update positions iteratively
        # To keep it simple and fast, we do a few passes of pairwise correction
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = rad[i] + rad[j]
                
                if d < min_d:
                    if d < 1e-9:
                        # Random push if coincident
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_d - d) / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                    else:
                        push = (min_d - d) / 2.0
                        ux = dx / d
                        uy = dy / d
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[j, 0] += ux * push
                        pos[j, 1] += uy * push
                    moved = True
        
        # Re-clamp to boundary after overlap resolution
        pos[:, 0] = np.clip(pos[:, 0], rad, w - rad)
        pos[:, 1] = np.clip(pos[:, 1], rad, h - rad)
        
        if not moved:
            break
    
    new_circles = [(pos[i, 0], pos[i, 1], rad[i]) for i in range(n)]
    
    # Final validation check
    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles
    
    return circles

def _optimize_radii_positions(circles, w, h, deadline):
    """
    Optimize radii and positions using SLSQP with hard constraints.
    """
    if time.time() > deadline - 5.0:
        return None
        
    n = len(circles)
    x0 = np.array([v for c in circles for v in c], dtype=np.float64)
    
    # Objective: maximize sum of radii => minimize negative sum
    def objective(x):
        if time.time() > deadline:
            return 0.0
        r = x[2::3]
        return -np.sum(r)
    
    # Constraints
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        # x >= r  => x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        # x + r <= w => w - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        # y >= r  => y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        # y + r <= h => h - y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        # r > 0 => r >= 1e-6
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6})
        
    # Overlap constraints: dist >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq', 
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w)) # x
        bounds.append((1e-6, h)) # y
        bounds.append((1e-6, min(w, h)/2)) # r
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-9}
        )
        
        if res.success:
            new_circles = [(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)]
            if _validate(new_circles, w, h, tol=1e-5):
                return new_circles
    except Exception:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_circles = None
    best_score = 0.0
    
    # Aspect ratios: w+h=2
    # We try a few aspect ratios. Square is often good, but slight deviations can help.
    aspect_ratios = [
        (1.0, 1.0),
        (0.95, 1.05),
        (1.05, 0.95),
        (0.9, 1.1),
        (1.1, 0.9),
    ]
    
    for w, h in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
            
        # Initialize with a grid of small circles
        # 21 circles: 5x5 grid minus 4 corners? Or 4x5 + 1?
        # Let's use a 5x5 grid, remove 4 corners to get 21.
        # Or just a 5x5 grid of 25 and remove 4? 
        # Simpler: 5x5 grid, take first 21.
        # Actually, a 4x5 grid is 20, plus 1 in the middle?
        # Let's stick to 5x5 grid, take 21 positions.
        # Positions: (0,0) to (4,4). Remove (0,0), (0,4), (4,0), (4,4) -> 21 circles.
        # This creates a hole in the middle? No, it's just the corners.
        # This shape is roughly rectangular with cut corners.
        
        cols, rows = 5, 5
        circles = []
        for i in range(rows):
            for j in range(cols):
                # Skip corners for better packing?
                # Actually, let's just use all 25 and pick 21?
                # No, the task says pack 21 circles.
                # Let's use a 5x5 grid and remove the 4 corners to allow larger circles in the middle?
                # Or just use the first 21 in row-major order?
                # Row-major: (0,0)..(0,4), (1,0)..(1,4), (2,0)..(2,4), (3,0)..(3,4), (4,0)..(4,2)
                # This is L-shaped.
                
                # Better: 4x5 grid = 20 circles. Add 1 in the center of the 4x5?
                # Let's use a 5x5 grid and remove 4 corners.
                if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                    continue
                cx = (j + 0.5) / cols * w
                cy = (i + 0.5) / rows * h
                # Small initial radius
                r = 0.05
                circles.append((cx, cy, r))
        
        # Ensure we have exactly 21
        if len(circles) != N:
            # Fallback to simple grid if logic failed
            circles = []
            for i in range(rows):
                for j in range(cols):
                    if len(circles) >= N:
                        break
                    cx = (j + 0.5) / cols * w
                    cy = (i + 0.5) / rows * h
                    circles.append((cx, cy, 0.05))
                    if len(circles) == N:
                        break
                if len(circles) == N:
                    break

        # Relax to ensure validity
        valid_circles = _relax_circles(circles, w, h, iterations=500)
        
        if _validate(valid_circles, w, h):
            score = _total_radius(valid_circles)
            if score > best_score:
                best_score = score
                best_circles = valid_circles
            
            # Optimize with SLSQP
            if time.time() < deadline - 15.0:
                optimized = _optimize_radii_positions(valid_circles, w, h, deadline)
                if optimized is not None:
                    opt_score = _total_radius(optimized)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_circles = optimized
    
    if best_circles is not None:
        return np.array(best_circles)
    
    # Fallback
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.05) 
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
