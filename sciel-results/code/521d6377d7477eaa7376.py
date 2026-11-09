"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side length, and rotation."""
    angle = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees + rotation
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angles = base_angles + angle
    verts = np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])
    return verts

def _signed_distance_point_to_polygon(px, py, verts):
    """
    Compute signed distance from point (px,py) to a convex polygon (CCW).
    Positive if inside, negative if outside.
    Uses distance to edges with sign based on cross product.
    """
    n = len(verts)
    min_dist = np.inf
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx = x2 - x1
        dy = y2 - y1
        # Distance from point to line segment
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = np.clip(t, 0, 1)
        cx = x1 + t * dx
        cy = y1 + t * dy
        dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        # Determine sign: cross product to check which side
        cross = dx * (py - y1) - dy * (px - x1)
        if cross < 0:  # outside (assuming CCW polygon, outside is clockwise)
            dist = -dist
        if abs(dist) < abs(min_dist):
            min_dist = dist
    return min_dist

def _sdf_overlap_penalty(verts1, verts2, sigma=0.05):
    """
    Compute a smooth penalty for overlap between two convex polygons.
    We sample points on each polygon's edges and compute SDF to the other.
    Use soft-min of negative SDF values (penalty when inside).
    """
    # Sample points on edges of polygon 1
    n1 = len(verts1)
    n2 = len(verts2)
    
    # Get points on edges of poly1
    pts1 = []
    for i in range(n1):
        x1, y1 = verts1[i]
        x2, y2 = verts1[(i + 1) % n1]
        for t in np.linspace(0, 1, 5):
            pts1.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    pts1 = np.array(pts1)
    
    # Compute SDF of each point to polygon 2
    sdf_vals = np.array([_signed_distance_point_to_polygon(p[0], p[1], verts2) for p in pts1])
    
    # Penalty: sum of soft-min of -sdf (penalize when inside, i.e., sdf > 0)
    # Actually, we want penalty when points are inside poly2 (sdf > 0)
    # Use softplus-like: penalty = sum of max(0, sdf) smoothed
    # Better: penalty = sum of log(1 + exp(sdf/sigma)) * sigma - sigma * n
    # This is a smooth approximation of sum of max(0, sdf)
    penalty = np.sum(sigma * np.log1p(np.exp(sdf_vals / sigma)) - sdf_vals) - len(sdf_vals) * sigma
    # Wait, that's not right. Let me think again.
    # We want a smooth function that is 0 when sdf < 0 (outside) and ~sdf when sdf > 0 (inside)
    # softplus(x) = log(1+exp(x)) approximates max(0,x)
    # So penalty = sum of softplus(sdf_vals)
    penalty = np.sum(np.log1p(np.exp(sdf_vals)))
    return penalty

def _sdf_containment_penalty(verts_inner, outer_center, outer_side, outer_angle):
    """
    Penalty for inner hexagon vertices being outside the outer hexagon.
    """
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    n = len(verts_inner)
    penalty = 0.0
    for i in range(n):
        px, py = verts_inner[i]
        sdf = _signed_distance_point_to_polygon(px, py, outer_verts)
        # Penalty when outside (sdf < 0)
        if sdf < 0:
            penalty += np.log1p(np.exp(-sdf))
    return penalty

def _compute_objective(params, outer_center, outer_side, outer_angle, sigma=0.05):
    """
    Compute the objective: sum of overlap penalties + containment penalties.
    params: array of shape (3*N,) with [x0, y0, angle0, x1, y1, angle1, ...]
    """
    total_penalty = 0.0
    verts_list = []
    
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        angle = params[3*i+2] * 180.0 / np.pi  # convert to degrees
        verts = _hex_vertices(x, y, 1.0, angle)
        verts_list.append(verts)
        
        # Containment penalty
        total_penalty += _sdf_containment_penalty(verts, outer_center, outer_side, outer_angle)
    
    # Overlap penalties between all pairs
    for i in range(N):
        for j in range(i+1, N):
            total_penalty += _sdf_overlap_penalty(verts_list[i], verts_list[j], sigma)
    
    return total_penalty

def _compute_objective_grad(params, outer_center, outer_side, outer_angle, sigma=0.05, eps=1e-5):
    """Compute gradient via finite differences (expensive but reliable)."""
    grad = np.zeros_like(params)
    base_val = _compute_objective(params, outer_center, outer_side, outer_angle, sigma)
    for i in range(len(params)):
        params[i] += eps
        val = _compute_objective(params, outer_center, outer_side, outer_angle, sigma)
        grad[i] = (val - base_val) / eps
        params[i] -= eps
    return grad

def _validate_packing(inner, center, side, angle):
    """Check if the packing is valid: all inner hexagons inside outer, no overlaps."""
    outer_verts = _hex_vertices(center[0], center[1], side, angle)
    
    inner_verts_list = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        inner_verts_list.append(verts)
        
        # Check containment
        for vx, vy in verts:
            sdf = _signed_distance_point_to_polygon(vx, vy, outer_verts)
            if sdf < -1e-6:
                return False
    
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            # Check if any vertex of i is inside j or vice versa
            for vx, vy in inner_verts_list[i]:
                sdf = _signed_distance_point_to_polygon(vx, vy, inner_verts_list[j])
                if sdf > 1e-6:
                    return False
            for vx, vy in inner_verts_list[j]:
                sdf = _signed_distance_point_to_polygon(vx, vy, inner_verts_list[i])
                if sdf > 1e-6:
                    return False
    return True

def _initial_params_lattice():
    """Hexagonal lattice initial configuration."""
    params = np.zeros(3 * N)
    # Arrange in a hexagonal pattern
    positions = []
    # Center hexagon
    positions.append((0, 0))
    # Ring 1: 6 hexagons
    for k in range(6):
        ang = k * np.pi / 3
        positions.append((2 * np.sqrt(3) * np.cos(ang), 2 * np.sqrt(3) * np.sin(ang)))
    # Ring 2: 4 more to reach 11
    for k in range(4):
        ang = k * np.pi / 2 + np.pi / 4
        positions.append((4 * np.sqrt(3) * np.cos(ang), 4 * np.sqrt(3) * np.sin(ang)))
    
    for i in range(N):
        params[3*i] = positions[i][0]
        params[3*i+1] = positions[i][1]
        params[3*i+2] = 0.0
    return params

def _initial_params_random(seed):
    """Random initial configuration."""
    rng = np.random.RandomState(seed)
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = rng.uniform(-5, 5)
        params[3*i+1] = rng.uniform(-5, 5)
        params[3*i+2] = rng.uniform(0, 2*np.pi)
    return params

def _initial_params_zigzag():
    """Zigzag row initial configuration."""
    params = np.zeros(3 * N)
    row1 = N // 2
    row2 = N - row1
    y1 = -1.5
    y2 = 1.5
    for i in range(row1):
        params[3*i] = (i - row1//2) * 2 * np.sqrt(3)
        params[3*i+1] = y1
        params[3*i+2] = 0.0
    for i in range(row2):
        params[3*(row1+i)] = (i - row2//2 + 0.5) * 2 * np.sqrt(3)
        params[3*(row1+i)+1] = y2
        params[3*(row1+i)+2] = 0.0
    return params

def construct():
    start_time = time.time()
    best_side = 40.0
    best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Try multiple initializations and optimize
    initializers = [
        _initial_params_lattice,
        _initial_params_zigzag,
        lambda: _initial_params_random(42),
        lambda: _initial_params_random(123),
        lambda: _initial_params_random(777),
    ]
    
    for init_idx, init_func in enumerate(initializers):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        try:
            params = init_func()
        except Exception:
            continue
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _compute_objective,
                params,
                args=(outer_center, best_side, outer_angle, 0.05),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6},
                callback=lambda x: None
            )
            
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Check if result is valid
            inner_result = []
            for i in range(N):
                inner_result.append([result.x[3*i], result.x[3*i+1], np.degrees(result.x[3*i+2])])
            
            if _validate_packing(inner_result, outer_center, best_side, outer_angle):
                # Now try to shrink the outer hexagon
                # Binary search on outer side length
                lo, hi = 3.0, best_side
                for _ in range(15):
                    if time.time() - start_time > DEADLINE - 3:
                        break
                    mid = (lo + hi) / 2
                    try:
                        res2 = minimize(
                            _compute_objective,
                            result.x,
                            args=(outer_center, mid, outer_angle, 0.05),
                            method='L-BFGS-B',
                            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
                        )
                        inner_mid = []
                        for i in range(N):
                            inner_mid.append([res2.x[3*i], res2.x[3*i+1], np.degrees(res2.x[3*i+2])])
                        if _validate_packing(inner_mid, outer_center, mid, outer_angle):
                            hi = mid
                            best_side = mid
                            best_inner = inner_mid
                            result.x = res2.x
                        else:
                            lo = mid
                    except Exception:
                        break
        except Exception:
            continue
    
    return best_inner, outer_center, best_side, outer_angle

# EVOLVE-BLOCK-END
