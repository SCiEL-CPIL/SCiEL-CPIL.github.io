import numpy as np
from scipy.spatial import Voronoi
import time

N = 32

def _voronoi_inradii(xy, r):
    """
    For each circle center, compute the maximum radius such that the circle
    stays within the unit square AND does not overlap any other circle,
    using Voronoi cells. The inradius of the Voronoi cell (clipped to the
    square) gives the max radius for that center.
    
    Returns an array of max_radii of shape (n,).
    """
    n = len(xy)
    max_r = np.zeros(n)
    
    # Boundary distances: max radius limited by square
    for i in range(n):
        d_left = xy[i, 0]
        d_right = 1.0 - xy[i, 0]
        d_bottom = xy[i, 1]
        d_top = 1.0 - xy[i, 1]
        max_r[i] = min(d_left, d_right, d_bottom, d_top)
    
    # Voronoi-based: for each circle, the nearest other circle center distance / 2
    # gives a local packing constraint, but Voronoi inradius is more accurate.
    # Build Voronoi of the centers, then for each cell find the minimum distance
    # from the center to any edge of the cell (clipped to [0,1]^2).
    
    # Add boundary points to ensure Voronoi cells are bounded
    # We'll use a simpler approach: for each i, max_r[i] = min over j!=i of dist(i,j)/2,
    # and also boundary distance. This is the "nearest neighbor" approximation.
    # For better accuracy, use actual Voronoi.
    
    try:
        vor = Voronoi(xy)
        for i in range(n):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if len(region) == 0:
                continue
            vertices = vor.vertices[region]
            # For each edge of the Voronoi cell, compute distance from xy[i] to that edge
            # But simpler: the inradius is the minimum distance from xy[i] to any
            # line segment forming the cell boundary.
            # Even simpler: the minimum distance to any other Voronoi vertex is an upper bound,
            # but the true inradius is the min distance to any edge.
            # Let's compute distance from xy[i] to each edge (line segment between consecutive vertices).
            min_dist = np.inf
            for k in range(len(vertices)):
                v1 = vertices[k]
                v2 = vertices[(k+1) % len(vertices)]
                # Distance from point to line segment
                d = _point_to_segment_dist(xy[i], v1, v2)
                if d < min_dist:
                    min_dist = d
            # Also check against square boundaries
            min_dist = min(min_dist, max_r[i])
            max_r[i] = min_dist
    except Exception:
        # Fallback: nearest neighbor
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = np.linalg.norm(xy[i] - xy[j])
                max_r[i] = min(max_r[i], d / 2.0)
    
    return max_r


def _point_to_segment_dist(p, a, b):
    """Distance from point p to line segment ab."""
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-15)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(p - closest)


def _is_valid(xy, r, tol=1e-9):
    n = len(r)
    if np.any(r < -tol):
        return False
    if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
        return False
    if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
        return False
    dx = xy[:, 0][:, None] - xy[:, 0][None, :]
    dy = xy[:, 1][:, None] - xy[:, 1][None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    iu = np.triu_indices(n, k=1)
    if np.any(dist[iu] < min_dist[iu] - tol):
        return False
    return True


def _gradient_step(xy, r, lr=0.005, max_iter=2000):
    """
    Gradient ascent on sum(r) with Voronoi-based radius constraints.
    We alternate:
      1. For current centers, compute max allowable radius for each circle.
      2. Set r = min(current r grown, max_r) — i.e., grow toward max_r.
      3. Compute gradient of sum(r) w.r.t. centers (via how max_r depends on centers).
      4. Update centers to increase the sum of max_r values.
    
    Simplified: use a penalty-based approach.
    Objective: maximize sum(r_i) subject to:
      - r_i <= boundary_dist(i)
      - r_i + r_j <= dist(i,j) for all i<j
    
    We use projected gradient on (xy, r) jointly.
    """
    n = len(xy)
    xy = xy.copy()
    r = r.copy()
    
    best_sum = np.sum(r)
    best_xy = xy.copy()
    best_r = r.copy()
    
    for it in range(max_iter):
        # Compute current max radii given centers
        max_r = _voronoi_inradii(xy, r)
        
        # Gradient of objective w.r.t. centers:
        # d(sum r_i)/d(xy_k) = sum over i of d(max_r_i)/d(xy_k)
        # Approximate via finite differences on a few dimensions, or use analytic.
        # For efficiency, use a simple heuristic: move centers apart if too close.
        
        # Alternative: treat it as a smooth optimization.
        # Use subgradient of the "growth" direction.
        # For each circle, the gradient of max_r_i w.r.t. its own center points
        # toward the center of the largest inscribed circle in its Voronoi cell.
        # This is complex. Instead, use a momentum-based push-apart.
        
        # Force-based but more structured:
        fxy = np.zeros_like(xy)
        
        # Pairwise repulsion when close
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[j, 0] - xy[i, 0]
                dy = xy[j, 1] - xy[i, 1]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = r[i] + r[j]
                if dist < min_dist + 0.01 and dist > 1e-10:
                    # Push apart proportional to how close they are
                    overlap = max(0, min_dist + 0.01 - dist)
                    f_vec = np.array([dx, dy]) / dist
                    fxy[i] -= f_vec * overlap * 2.0
                    fxy[j] += f_vec * overlap * 2.0
        
        # Boundary forces
        for i in range(n):
            if xy[i, 0] < r[i] + 0.005:
                fxy[i, 0] += (r[i] + 0.005 - xy[i, 0]) * 5.0
            if xy[i, 0] > 1 - r[i] - 0.005:
                fxy[i, 0] -= (xy[i, 0] - (1 - r[i] - 0.005)) * 5.0
            if xy[i, 1] < r[i] + 0.005:
                fxy[i, 1] += (r[i] + 0.005 - xy[i, 1]) * 5.0
            if xy[i, 1] > 1 - r[i] - 0.005:
                fxy[i, 1] -= (xy[i, 1] - (1 - r[i] - 0.005)) * 5.0
        
        # Update centers
        xy += lr * fxy
        xy = np.clip(xy, 0.01, 0.99)
        
        # Update radii: grow toward max_r
        # Use a growth rate that depends on how much room there is
        for i in range(n):
            if r[i] < max_r[i]:
                # Grow by a fraction of the gap
                growth = (max_r[i] - r[i]) * 0.1
                r[i] += growth
            else:
                # Slight shrink to allow repositioning
                r[i] *= 0.999
        
        # Clamp
        r = np.clip(r, 0.001, 0.5)
        
        # Check validity
        if _is_valid(xy, r):
            s = np.sum(r)
            if s > best_sum:
                best_sum = s
                best_xy = xy.copy()
                best_r = r.copy()
        else:
            # Shrink to validity
            for scale in np.linspace(1.0, 0.5, 20):
                r_test = r * scale
                if _is_valid(xy, r_test):
                    r = r_test
                    break
        
        # Adaptive learning rate
        if it % 200 == 0:
            lr *= 0.9
    
    return best_xy, best_r


def _initial_config(restart, rng):
    n = N
    if restart == 0:
        # 6x6 grid
        side = 6
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        for i in range(n):
            row = i // side
            col = i % side
            xy[i] = [(col + 0.5) / side, (row + 0.5) / side]
            r[i] = 0.5 / side - 0.005
    elif restart == 1:
        # 8x4 grid
        side_x = 8
        side_y = 4
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        for i in range(n):
            row = i // side_x
            col = i % side_x
            xy[i] = [(col + 0.5) / side_x, (row + 0.5) / side_y]
            r[i] = min(0.5 / side_x, 0.5 / side_y) - 0.005
    elif restart == 2:
        # Hexagonal-ish
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        cols = 6
        for i in range(n):
            row = i // cols
            col = i % cols
            x = (col + 0.5) / cols + (1.0 / (2 * cols) if row % 2 == 1 else 0)
            y = (row + 0.5) / 6
            xy[i] = [x, y]
            r[i] = 0.075
    elif restart == 3:
        # Random
        xy = rng.uniform(0.1, 0.9, size=(n, 2))
        r = rng.uniform(0.04, 0.08, size=n)
    elif restart == 4:
        # 4x8
        side_x = 4
        side_y = 8
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        for i in range(n):
            row = i // side_x
            col = i % side_x
            xy[i] = [(col + 0.5) / side_x, (row + 0.5) / side_y]
            r[i] = min(0.5 / side_x, 0.5 / side_y) - 0.005
    else:
        # Random with bias toward edges
        xy = rng.uniform(0.05, 0.95, size=(n, 2))
        r = rng.uniform(0.05, 0.1, size=n)
    
    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_score = -np.inf
    
    rng = np.random.RandomState(123)
    
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline - 3:
            break
        
        xy, r = _initial_config(restart % 5, rng)
        
        # Run gradient optimization
        xy_opt, r_opt = _gradient_step(xy, r, lr=0.003, max_iter=1500)
        
        if not _is_valid(xy_opt, r_opt):
            for scale in np.linspace(1.0, 0.3, 30):
                r_test = r_opt * scale
                if _is_valid(xy_opt, r_test):
                    r_opt = r_test
                    break
            else:
                continue
        
        score = np.sum(r_opt)
        if score > best_score:
            best_score = score
            best_result = (xy_opt.copy(), r_opt.copy())
        
        # Periodic refinement of best
        if best_result is not None and restart % 5 == 4:
            if time.time() < deadline - 15:
                xy_ref, r_ref = _gradient_step(best_result[0], best_result[1], lr=0.001, max_iter=1000)
                if not _is_valid(xy_ref, r_ref):
                    for scale in np.linspace(1.0, 0.3, 30):
                        r_test = r_ref * scale
                        if _is_valid(xy_ref, r_test):
                            r_ref = r_test
                            break
                    else:
                        continue
                ref_score = np.sum(r_ref)
                if ref_score > best_score:
                    best_score = ref_score
                    best_result = (xy_ref.copy(), r_ref.copy())
    
    if best_result is None:
        # Fallback
        side = 6
        xy = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            xy[i] = [(col + 0.5) / side, (row + 0.5) / side]
            r[i] = 0.5 / side - 0.005
        return np.column_stack([xy, r])
    
    xy, r = best_result
    if not _is_valid(xy, r):
        for scale in np.linspace(1.0, 0.1, 50):
            r_test = r * scale
            if _is_valid(xy, r_test):
                r = r_test
                break
    
    return np.column_stack([xy, r])
