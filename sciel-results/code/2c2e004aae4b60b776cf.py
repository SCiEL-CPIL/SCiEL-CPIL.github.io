"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _check_validity(circles, W, H):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r < 1e-9:
            return False
        if x - r < -1e-9 or y - r < -1e-9 or x + r > W + 1e-9 or y + r > H + 1e-9:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist_sq = dx * dx + dy * dy
            min_dist = circles[i][2] + circles[j][2]
            if dist_sq < min_dist * min_dist - 1e-9:
                return False
    return True


def _objective_slsqp(params, W, H):
    """Objective for SLSQP: Minimize -sum(r)"""
    circles = params.reshape(N, 3)
    r = circles[:, 2]
    return -np.sum(r)


def _constraints_slsqp(W, H):
    """Constraints for SLSQP"""
    cons = []
    # Boundary constraints: x - r >= 0, y - r >= 0, x + r <= W, y + r <= H
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: W - p[3*i] - p[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: H - p[3*i+1] - p[3*i+2]})
    
    # Overlap constraints: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i] - p[3*j])**2 + (p[3*i+1] - p[3*j+1])**2 - (p[3*i+2] + p[3*j+2])**2
            })
            
    # Radius positive
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+2]})

    return cons


def _local_optimize_slsqp(circles, W, H, max_iter=1000):
    """Use SLSQP to locally optimize the circle positions and radii."""
    params = circles.flatten()
    constraints = _constraints_slsqp(W, H)
    
    # Bounds for variables
    # x in [0, W], y in [0, H], r in [0, min(W,H)/2]
    r_max = min(W, H) / 2.0
    bounds = []
    for i in range(N):
        bounds.append((0, W))
        bounds.append((0, H))
        bounds.append((1e-6, r_max))

    try:
        result = minimize(
            _objective_slsqp, params, args=(W, H),
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-12}
        )
        if result.success:
            return result.x.reshape(N, 3)
    except Exception:
        pass
    return circles


def _local_optimize_nm(circles, W, H, max_iter=2000):
    """Use Nelder-Mead with penalty for final polish."""
    def obj(params):
        c = params.reshape(N, 3)
        x, y, r = c[:,0], c[:,1], c[:,2]
        pen = 0.0
        # Boundary
        pen += 100.0 * np.sum(np.maximum(0, r - x))
        pen += 100.0 * np.sum(np.maximum(0, r - y))
        pen += 100.0 * np.sum(np.maximum(0, x + r - W))
        pen += 100.0 * np.sum(np.maximum(0, y + r - H))
        # Overlap
        for i in range(N):
            for j in range(i+1, N):
                d2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                d_min = r[i] + r[j]
                if d2 < d_min**2:
                    pen += 10.0 * (d_min - np.sqrt(max(d2, 0)))**2
        return -np.sum(r) + pen

    params = circles.flatten()
    try:
        result = minimize(
            obj, params,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        return result.x.reshape(N, 3)
    except Exception:
        return circles


def _grid_initialization(W, H):
    """Create a grid-based initial configuration."""
    # Try to find a grid that fits well
    # For N=21, 5x5 grid is 25 spots, we take 21.
    # Or 4x6 = 24 spots.
    # Let's try a few grid shapes and pick the one with largest min radius.
    best_circles = None
    best_r = 0
    
    for rows in range(3, 8):
        for cols in range(3, 8):
            if rows * cols < N:
                continue
            # Calculate max radius for this grid
            # Circles must fit in W and H
            # Distance between centers: W/cols, H/rows
            # Max r is min(W/cols, H/rows) / 2
            r = min(W / cols, H / rows) / 2.0
            if r > best_r:
                best_r = r
                circles = []
                count = 0
                for i in range(rows):
                    for j in range(cols):
                        if count >= N:
                            break
                        cx = (j + 0.5) * W / cols
                        cy = (i + 0.5) * H / rows
                        circles.append([cx, cy, r])
                        count += 1
                    if count >= N:
                        break
                best_circles = np.array(circles)
                
    return best_circles


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None

    # Perimeter = 2(W+H) = 4 => W+H = 2
    # Aspect ratio a = W/H => W = 2a/(a+1), H = 2/(a+1)
    # Optimal aspect ratio is likely close to 1 or related to sqrt(N)
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    
    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)
        
        # 1. Grid Initialization + SLSQP + NM Polish
        circles = _grid_initialization(W, H)
        if circles is not None:
            # SLSQP
            circ_slsqp = _local_optimize_slsqp(circles, W, H, max_iter=500)
            # NM Polish
            circ_nm = _local_optimize_nm(circ_slsqp, W, H, max_iter=1000)
            
            if _check_validity(circ_nm, W, H):
                score = np.sum(circ_nm[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = circ_nm.copy()

        # 2. Random Restarts
        n_restarts = 5
        for _ in range(n_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Random initialization
            circles = []
            for _ in range(N):
                r = rng.uniform(0.05, min(W, H)/4)
                x = rng.uniform(r, W - r)
                y = rng.uniform(r, H - r)
                circles.append([x, y, r])
            circles = np.array(circles)
            
            # SLSQP
            circ_slsqp = _local_optimize_slsqp(circles, W, H, max_iter=300)
            # NM Polish
            circ_nm = _local_optimize_nm(circ_slsqp, W, H, max_iter=1000)
            
            if _check_validity(circ_nm, W, H):
                score = np.sum(circ_nm[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = circ_nm.copy()

    # Fallback if no valid solution found
    if best_result is None:
        # Simple grid in square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        # Ensure it fits in 1x1 (perimeter 4)
        # The grid above is in [0,1]x[0,1] which has perimeter 4.
        # But radii might be too large if they overlap.
        # Let's just return a very safe small circle config.
        r_safe = 0.01
        x_safe = np.linspace(r_safe, 1-r_safe, N)
        y_safe = np.linspace(r_safe, 1-r_safe, N)
        # This is just a line, not a packing. 
        # Better fallback:
        cells = []
        r = 0.05
        for i in range(5):
            for j in range(5):
                if len(cells) < N:
                    cells.append([0.1 + i*0.2, 0.1 + j*0.2, r])
        best_result = np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
