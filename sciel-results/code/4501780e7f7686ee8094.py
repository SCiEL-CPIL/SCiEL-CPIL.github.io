"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay
import time


def _elapsed(start):
    return time.time() - start


def _remaining(start, deadline):
    return max(0.0, deadline - time.time())


def _is_valid(xy, r, eps=1e-9):
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -eps or 1.0 - x - ri < -eps or y - ri < -eps or 1.0 - y - ri < -eps:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            if np.sqrt(dx * dx + dy * dy) < r[i] + r[j] - eps:
                return False
    return True


def _score(xy, r):
    if not _is_valid(xy, r):
        return 0.0
    return float(np.sum(r))


def _hex_packing_init(n):
    """Hexagonal packing of n circles in unit square (equal radii)."""
    positions = []
    r = 0.5 / (1 + np.sqrt(3))  # hex circle radius for unit spacing
    # Actually for hex packing in square, use a simpler approach
    # Place circles in hexagonal grid
    dx = 2 * r
    dy = r * np.sqrt(3)
    row = 0
    while len(positions) < n:
        for col in range(int(1.0 / dx) + 2):
            if len(positions) >= n:
                break
            x = dx * (col + (0.5 if row % 2 else 0)) + r
            y = dy * row + r
            if x + r <= 1.0 + 1e-9 and y + r <= 1.0 + 1e-9 and x - r >= -1e-9 and y - r >= -1e-9:
                positions.append([x, y, r])
        row += 1
    # Pad if needed
    while len(positions) < n:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:n])


def _structured_init(n, seed=0):
    """Structured grid init with slight jitter."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(n):
        gx = i % side
        gy = i // side
        x = (gx + 0.5) / side + rng.uniform(-0.02, 0.02)
        y = (gy + 0.5) / side + rng.uniform(-0.02, 0.02)
        r = 0.4 / side
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        positions.append([x, y, r])
    return np.array(positions[:n])


def _random_init(n, seed=0):
    """Random init with small radii."""
    rng = np.random.RandomState(seed)
    positions = []
    for i in range(n):
        r = rng.uniform(0.01, 0.05)
        x = rng.uniform(r, 1 - r)
        y = rng.uniform(r, 1 - r)
        positions.append([x, y, r])
    return np.array(positions[:n])


def _grid_init(n, side=6):
    """Simple grid init."""
    positions = []
    for i in range(n):
        gx = i % side
        gy = i // side
        x = (gx + 0.5) / side
        y = (gy + 0.5) / side
        r = 0.4 / side
        positions.append([x, y, r])
    return np.array(positions[:n])


def _build_contact_graph(xy, r, eps=0.005):
    """
    Build a contact graph: edges where dist(i,j) <= r[i]+r[j]+eps.
    Returns list of (i, j) edges.
    """
    n = len(xy)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist <= r[i] + r[j] + eps:
                edges.append((i, j))
    return edges


def _truss_objective(params, n, edges, wall_edges):
    """
    Minimize negative sum of radii subject to contact slacks being non-negative.
    We use a penalty on the minimum slack to encourage all contacts to be tight.
    
    params: flat array [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    edges: list of (i, j) for circle-circle contacts
    wall_edges: list of (i, wall) where wall in {0,1,2,3} = left, right, bottom, top
    """
    params = np.asarray(params)
    xy = params[0::3].reshape(-1, 2)
    r = np.maximum(params[1::3], 0.001)
    
    # Compute slacks for circle-circle contacts
    slacks = []
    for (i, j) in edges:
        dx = xy[i, 0] - xy[j, 0]
        dy = xy[i, 1] - xy[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        slack = dist - (r[i] + r[j])
        slacks.append(slack)
    
    # Compute slacks for wall contacts
    for (i, wall) in wall_edges:
        if wall == 0:  # left
            slacks.append(xy[i, 0] - r[i])
        elif wall == 1:  # right
            slacks.append(1.0 - xy[i, 0] - r[i])
        elif wall == 2:  # bottom
            slacks.append(xy[i, 1] - r[i])
        elif wall == 3:  # top
            slacks.append(1.0 - xy[i, 1] - r[i])
    
    if len(slacks) == 0:
        return -np.sum(r)
    
    slacks = np.array(slacks)
    min_slack = np.min(slacks)
    
    # Objective: maximize sum of radii, penalize negative min slack
    # We want min_slack >= 0, so penalize max(0, -min_slack)^2
    penalty = 1000.0 * max(0.0, -min_slack) ** 2
    # Also penalize any individual negative slack
    neg_slack_penalty = 100.0 * np.sum(np.maximum(0.0, -slacks)) ** 2
    
    total_radius = np.sum(r)
    return -total_radius + penalty + neg_slack_penalty


def _truss_optimize(xy, r, deadline, max_iter=200):
    """
    Optimize using the truss/contact-graph model.
    Iteratively rebuild contact graph and optimize.
    """
    n = len(xy)
    best_total = _score(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    
    for iteration in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Build contact graph
        edges = _build_contact_graph(xy, r, eps=0.01)
        
        # Build wall contact graph
        wall_edges = []
        for i in range(n):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            if x - ri < 0.01:
                wall_edges.append((i, 0))
            if 1.0 - x - ri < 0.01:
                wall_edges.append((i, 1))
            if y - ri < 0.01:
                wall_edges.append((i, 2))
            if 1.0 - y - ri < 0.01:
                wall_edges.append((i, 3))
        
        if len(edges) == 0 and len(wall_edges) == 0:
            break
        
        # Prepare optimization
        params = np.zeros(3 * n)
        params[0::3] = xy[:, 0]
        params[1::3] = xy[:, 1]
        params[2::3] = r
        
        bounds = []
        for i in range(n):
            bounds.append((0.001, 0.999))
            bounds.append((0.001, 0.999))
            bounds.append((0.001, 0.4))
        
        try:
            result = minimize(
                _truss_objective,
                params,
                args=(n, edges, wall_edges),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            opt_params = result.x
            opt_xy = opt_params[0::3].reshape(-1, 2)
            opt_r = np.maximum(opt_params[1::3], 0.001)
            
            # Validate
            total = _score(opt_xy, opt_r)
            if total > best_total:
                best_total = total
                best_xy = opt_xy.copy()
                best_r = opt_r.copy()
                xy = opt_xy
                r = opt_r
            else:
                # No improvement, try with slightly different contact tolerance
                break
                
        except Exception:
            break
    
    return best_xy, best_r, best_total


def _grow_radii(xy, r, deadline, max_iter=50):
    """
    Try to grow radii while maintaining validity.
    Use a simple iterative approach: try increasing each radius slightly.
    """
    n = len(xy)
    best_total = _score(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    
    for iteration in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline - 2.0:
                break
            
            # Try to increase radius of circle i
            old_r = r[i]
            new_r = old_r * 1.01 + 0.001
            r[i] = new_r
            
            # Check validity
            if _is_valid(xy, r):
                total = _score(xy, r)
                if total > best_total:
                    best_total = total
                    best_xy = xy.copy()
                    best_r = r.copy()
                    improved = True
                else:
                    r[i] = old_r
            else:
                r[i] = old_r
        
        if not improved:
            break
    
    return best_xy, best_r, best_total


def _voronoi_init(n, seed=0):
    """
    Initialize using Voronoi-based largest empty circle approach.
    Start with a few points, iteratively add the largest empty circle.
    """
    rng = np.random.RandomState(seed)
    points = []
    radii = []
    
    # Start with a random point
    x0 = rng.uniform(0.2, 0.8)
    y0 = rng.uniform(0.2, 0.8)
    r0 = 0.05
    points.append([x0, y0])
    radii.append(r0)
    
    for _ in range(n - 1):
        # Compute Voronoi of current points
        if len(points) < 3:
            # Add a random point
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            r = 0.03
            points.append([x, y])
            radii.append(r)
            continue
        
        pts = np.array(points)
        try:
            vor = Voronoi(pts)
        except Exception:
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            r = 0.03
            points.append([x, y])
            radii.append(r)
            continue
        
        # Find the vertex with maximum distance to nearest point, but also check walls
        best_r = 0.0
        best_x, best_y = 0.5, 0.5
        
        # Check Voronoi vertices
        for v in vor.vertices:
            vx, vy = v
            if vx < -0.01 or vx > 1.01 or vy < -0.01 or vy > 1.01:
                continue
            # Distance to nearest existing point
            dists = np.sqrt((pts[:, 0] - vx)**2 + (pts[:, 1] - vy)**2)
            min_dist = np.min(dists)
            # Also check wall distance
            wall_dist = min(vx, 1.0 - vx, vy, 1.0 - vy)
            candidate_r = min(min_dist, wall_dist) * 0.95  # 5% margin
            if candidate_r > best_r:
                best_r = candidate_r
                best_x, best_y = vx, vy
        
        # Also check boundary midpoints
        for bx, by in [(0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5)]:
            dists = np.sqrt((pts[:, 0] - bx)**2 + (pts[:, 1] - by)**2)
            min_dist = np.min(dists)
            wall_dist = min(bx, 1.0 - bx, by, 1.0 - by)
            candidate_r = min(min_dist, wall_dist) * 0.95
            if candidate_r > best_r:
                best_r = candidate_r
                best_x, best_y = bx, by
        
        if best_r > 0.005:
            points.append([best_x, best_y])
            radii.append(best_r)
        else:
            # Fallback: random
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            r = 0.02
            points.append([x, y])
            radii.append(r)
    
    xy = np.array(points[:n])
    r = np.array(radii[:n])
    
    # Ensure validity by reducing radii if needed
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j]:
                # Reduce both radii
                excess = r[i] + r[j] - dist
                r[i] = max(0.001, r[i] - excess / 2)
                r[j] = max(0.001, r[j] - excess / 2)
    
    # Ensure inside walls
    for i in range(n):
        r[i] = min(r[i], xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
        r[i] = max(r[i], 0.001)
    
    return xy, r


def _soft_objective(params, n, penalty_weight=500.0):
    """
    Soft penalty objective for L-BFGS-B.
    """
    params = np.asarray(params)
    xy = params[0::3].reshape(-1, 2)
    r = np.maximum(params[1::3], 0.001)
    
    wall_violation = 0.0
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        wall_violation += max(0.0, ri - x)**2
        wall_violation += max(0.0, ri - (1.0 - x))**2
        wall_violation += max(0.0, ri - y)**2
        wall_violation += max(0.0, ri - (1.0 - y))**2
    
    overlap_violation = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_violation += (min_dist - dist)**2
    
    penalty = penalty_weight * (wall_violation + overlap_violation)
    total_radius = np.sum(r)
    return -total_radius + penalty


def _gradient_optimize(xy, r, deadline, max_iter=200, penalty_weight=500.0):
    """
    L-BFGS-B with soft penalties.
    """
    n = len(xy)
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r
    
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.4))
    
    best_total = _score(xy, r)
    best_params = params.copy()
    
    try:
        result = minimize(
            _soft_objective,
            params,
            args=(n, penalty_weight),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        opt_params = result.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[1::3], 0.001)
        
        total = _score(opt_xy, opt_r)
        if total > best_total:
            best_total = total
            best_params = opt_params.copy()
            
    except Exception:
        pass
    
    final_xy = best_params[0::3].reshape(-1, 2)
    final_r = np.maximum(best_params[1::3], 0.001)
    return final_xy, final_r, best_total


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_xy = None
    best_r = None
    best_total = 0.0
    
    # Try multiple initializations
    inits = []
    
    # 1. Hex packing
    hex_pos = _hex_packing_init(N)
    inits.append((hex_pos[:, :2], hex_pos[:, 2].copy(), "hex"))
    
    # 2. Structured grid
    for seed in [0, 1, 2]:
        grid_pos = _structured_init(N, seed=seed)
        inits.append((grid_pos[:, :2], grid_pos[:, 2].copy(), f"grid_{seed}"))
    
    # 3. Voronoi-based
    vor_xy, vor_r = _voronoi_init(N, seed=0)
    inits.append((vor_xy, vor_r, "voronoi"))
    
    # 4. Random
    for seed in [10, 20]:
        rand_pos = _random_init(N, seed=seed)
        inits.append((rand_pos[:, :2], rand_pos[:, 2].copy(), f"rand_{seed}"))
    
    # 5. Simple grid
    grid_pos = _grid_init(N, side=6)
    inits.append((grid_pos[:, :2], grid_pos[:, 2].copy(), "simple_grid"))
    
    for xy_init, r_init, name in inits:
        if time.time() > deadline - 20.0:
            break
        
        # Ensure initial validity
        total_init = _score(xy_init, r_init)
        if total_init <= 0:
            continue
        
        # Step 1: Truss optimization (the new core method)
        if time.time() < deadline - 10.0:
            t_xy, t_r, t_total = _truss_optimize(xy_init, r_init, deadline - 10.0, max_iter=100)
            if t_total > best_total:
                best_total = t_total
                best_xy = t_xy
                best_r = t_r
        
        # Step 2: Gradient optimization (as backup/polish)
        if time.time() < deadline - 5.0 and best_total > 0:
            g_xy, g_r, g_total = _gradient_optimize(
                best_xy if best_xy is not None else xy_init,
                best_r if best_r is not None else r_init,
                deadline - 5.0,
                max_iter=100,
                penalty_weight=500.0
            )
            if g_total > best_total:
                best_total = g_total
                best_xy = g_xy
                best_r = g_r
    
    # If no valid result, fallback
    if best_xy is None or best_total <= 0:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result
    
    # Final: grow radii if time permits
    if time.time() < deadline - 3.0:
        p_xy, p_r, p_total = _grow_radii(best_xy, best_r, deadline - 2.0, max_iter=30)
        if p_total > best_total:
            best_total = p_total
            best_xy = p_xy
            best_r = p_r
    
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _score(final_xy, final_r), _is_valid(final_xy, final_r)
    if not _is_valid(final_xy, final_r):
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
    
    return result
# EVOLVE-BLOCK-END
