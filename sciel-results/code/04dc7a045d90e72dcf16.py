"""Circle packing in unit square n=32: maximize sum of radii."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog, minimize

DEADLINE = 140.0  # safety margin before 150s kill

def _validate(circles):
    """Check circles are inside [0,1]^2 with disjoint interiors."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < 0:
            return False
        if x - r < -1e-9 or y - r < -1e-9:
            return False
        if x + r > 1 + 1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            d = np.hypot(xi - xj, yi - yj)
            if d < ri + rj - 1e-9:
                return False
    return True

def _sum_radii(circles):
    return float(np.sum(circles[:, 2]))

def _optimize_radii_for_centers(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to non-overlap and boundary."""
    n = len(centers)
    # Variables: r_0, ..., r_{n-1}
    # Maximize sum(r) -> minimize -sum(r)
    c_obj = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # Boundary: r_i <= x_i, r_i <= y_i, r_i <= 1-x_i, r_i <= 1-y_i
    # Pairwise: r_i + r_j <= d_ij
    
    A_rows = []
    b_rows = []
    
    for i in range(n):
        x, y = centers[i]
        # r_i <= x_i
        row = np.zeros(n); row[i] = 1.0
        A_rows.append(row); b_rows.append(x)
        # r_i <= y_i
        row = np.zeros(n); row[i] = 1.0
        A_rows.append(row); b_rows.append(y)
        # r_i <= 1-x_i
        row = np.zeros(n); row[i] = 1.0
        A_rows.append(row); b_rows.append(1.0 - x)
        # r_i <= 1-y_i
        row = np.zeros(n); row[i] = 1.0
        A_rows.append(row); b_rows.append(1.0 - y)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            row = np.zeros(n); row[i] = 1.0; row[j] = 1.0
            A_rows.append(row); b_rows.append(d)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    bounds = [(0, None)] * n
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success and res.x is not None:
        radii = np.clip(res.x, 0, None)
        # Re-check feasibility with the computed radii
        circles = np.column_stack([centers, radii])
        if _validate(circles):
            return circles
        # If validation fails due to numerical issues, shrink slightly
        for scale in [0.999, 0.998, 0.995, 0.99, 0.98, 0.95]:
            radii2 = radii * scale
            circles = np.column_stack([centers, radii2])
            if _validate(circles):
                return circles
    return None

def _hexagonal_init():
    """Hexagonal packing initial centers."""
    # For 32 circles, try a hexagonal arrangement
    # Hexagonal lattice with spacing s: centers at (i*s + (j%2)*s/2, j*s*sqrt(3)/2)
    # We want to fit as many as possible in unit square
    # Try to find a good spacing and position
    
    best = None
    best_score = -1
    
    for s in np.linspace(0.12, 0.25, 60):
        # Generate hex grid points
        centers = []
        j = 0
        while True:
            i = 0
            while True:
                x = i * s + (j % 2) * s / 2
                y = j * s * np.sqrt(3) / 2
                if x > 1.0 or y > 1.0:
                    break
                centers.append((x, y))
                i += 1
            if len(centers) >= N:
                break
            j += 1
            if j > 20:
                break
        
        if len(centers) >= N:
            # Take first N, but try to center them
            # Compute bounding box and shift
            pts = np.array(centers[:N])
            xmin, ymin = pts.min(axis=0)
            xmax, ymax = pts.max(axis=0)
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            # Shift to center in unit square
            shifted = pts - np.array([cx, cy]) + np.array([0.5, 0.5])
            
            # Check if all in [0,1]^2 (with some margin for radii)
            if shifted.min() >= -0.01 and shifted.max() <= 1.01:
                # Estimate min radius
                min_r = min(
                    shifted[:, 0].min(),
                    shifted[:, 1].min(),
                    (1 - shifted[:, 0]).min(),
                    (1 - shifted[:, 1]).min()
                )
                # Min pairwise distance
                min_d = float('inf')
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.hypot(shifted[i,0]-shifted[j,0], shifted[i,1]-shifted[j,1])
                        min_d = min(min_d, d)
                est_score = min(min_r, min_d/2) * N
                if est_score > best_score:
                    best_score = est_score
                    best = shifted.copy()
    
    if best is not None:
        return best
    return None

def _grid_init():
    """Grid initial centers."""
    side = int(np.ceil(np.sqrt(N)))
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) < N:
                centers.append(((i + 0.5) / side, (j + 0.5) / side))
    return np.array(centers[:N])

def _random_init():
    """Random initial centers."""
    return np.random.uniform(0.05, 0.95, size=(N, 2))

def _perturb(centers, scale=0.05):
    """Perturb centers randomly."""
    new_centers = centers + np.random.normal(0, scale, size=centers.shape)
    # Clip to stay within [0.01, 0.99]
    new_centers = np.clip(new_centers, 0.01, 0.99)
    return new_centers

def _local_optimize(centers, max_iter=50, tol=1e-6):
    """Local optimization: alternate between center optimization and radius LP."""
    current_centers = centers.copy()
    current_circles = _optimize_radii_for_centers(current_centers)
    if current_circles is None:
        return None
    current_score = _sum_radii(current_circles)
    
    for it in range(max_iter):
        # Optimize centers using gradient-based method
        # Objective: maximize sum of radii (from LP)
        # Since LP is piecewise linear, we use a smoothed approximation
        
        # Simpler approach: use the current radii as fixed, and move centers
        # to increase the minimum constraint slack
        radii = current_circles[:, 2]
        
        # Compute constraint slacks
        # For each pair, slack = d_ij - r_i - r_j
        # For boundaries, slack = x_i - r_i, etc.
        
        # Use gradient ascent on a barrier function
        def barrier_neg(x_flat):
            x = x_flat.reshape(-1, 2)
            # Boundary penalties
            pen = 0.0
            for i in range(N):
                pen += max(0, radii[i] - x[i, 0]) ** 2
                pen += max(0, radii[i] - x[i, 1]) ** 2
                pen += max(0, radii[i] - (1 - x[i, 0])) ** 2
                pen += max(0, radii[i] - (1 - x[i, 1])) ** 2
            # Pairwise penalties
            for i in range(N):
                for j in range(i+1, N):
                    d = np.hypot(x[i,0]-x[j,0], x[i,1]-x[j,1])
                    pen += max(0, radii[i] + radii[j] - d) ** 2
            # Also reward moving apart
            # Actually, we want to minimize penalty, which is equivalent to maximizing feasibility
            return pen
        
        # But this doesn't directly maximize sum of radii. 
        # Better: re-solve LP after moving centers slightly.
        
        # Try small random moves and accept if LP score improves
        improved = False
        for _ in range(20):
            new_centers = _perturb(current_centers, scale=0.02)
            new_circles = _optimize_radii_for_centers(new_centers)
            if new_circles is not None:
                new_score = _sum_radii(new_circles)
                if new_score > current_score + tol:
                    current_centers = new_centers
                    current_circles = new_circles
                    current_score = new_score
                    improved = True
                    break
        
        if not improved:
            break
    
    return current_circles

def _simulated_annealing(start_centers, iterations=500, t0=0.1, t_min=0.001, cooling=0.995):
    """Simulated annealing on centers with LP radius optimization."""
    current_centers = start_centers.copy()
    current_circles = _optimize_radii_for_centers(current_centers)
    if current_circles is None:
        return None
    current_score = _sum_radii(current_circles)
    
    best_centers = current_centers.copy()
    best_circles = current_circles.copy()
    best_score = current_score
    
    t = t0
    for it in range(iterations):
        new_centers = _perturb(current_centers, scale=t * 0.5)
        new_circles = _optimize_radii_for_centers(new_centers)
        if new_circles is None:
            continue
        new_score = _sum_radii(new_circles)
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / max(t, 1e-9)):
            current_centers = new_centers
            current_circles = new_circles
            current_score = new_score
            if current_score > best_score:
                best_centers = current_centers.copy()
                best_circles = current_circles.copy()
                best_score = current_score
        t *= cooling
        if t < t_min:
            break
        if time.time() > DEADLINE - 10:
            break
    
    return best_circles

def construct():
    start_time = time.time()
    
    # Try multiple initializations
    inits = []
    
    # Grid
    inits.append(_grid_init())
    
    # Hexagonal
    hex_init = _hexagonal_init()
    if hex_init is not None:
        inits.append(hex_init)
    
    # Random
    for _ in range(3):
        inits.append(_random_init())
    
    best_circles = None
    best_score = -1
    
    for init_centers in inits:
        if time.time() > DEADLINE - 20:
            break
        
        # First, optimize radii for this center configuration
        circles = _optimize_radii_for_centers(init_centers)
        if circles is None:
            continue
        score = _sum_radii(circles)
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
        
        # Then do local optimization
        if time.time() > DEADLINE - 15:
            break
        local_result = _local_optimize(init_centers, max_iter=30)
        if local_result is not None:
            local_score = _sum_radii(local_result)
            if local_score > best_score:
                best_score = local_score
                best_circles = local_result.copy()
    
    # Simulated annealing from best center
    if best_circles is not None and time.time() < DEADLINE - 30:
        start_centers = best_circles[:, :2]
        sa_result = _simulated_annealing(start_centers, iterations=300, t0=0.05, t_min=0.001, cooling=0.995)
        if sa_result is not None:
            sa_score = _sum_radii(sa_result)
            if sa_score > best_score:
                best_score = sa_score
                best_circles = sa_result.copy()
    
    if best_circles is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
