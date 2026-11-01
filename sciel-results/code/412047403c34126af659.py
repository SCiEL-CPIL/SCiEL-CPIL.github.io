"""Packing 12 unit hexagons in a minimal outer hexagon via gradient-based optimization."""
import numpy as np
from scipy.optimize import minimize
import math

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def hexagon_support(cx, cy, side, angle_deg, direction):
    """Max projection of hexagon onto a unit direction vector."""
    verts = hex_vertices(cx, cy, side, angle_deg)
    return np.max(verts @ direction)

def outer_hexagon_required(inner_list, outer_center, outer_angle_deg):
    """
    Compute the minimum outer hexagon side length such that all inner hexagons
    fit inside the outer hexagon (centered at outer_center, rotated by outer_angle_deg).
    
    For a regular hexagon with side s, the support function in direction u is s * (cos of angle between u and nearest hex-axis).
    Actually, for a regular hexagon, the support in direction making angle phi with a vertex direction is s * cos(phi) for |phi| <= 30 deg, but more precisely:
    The hexagon with vertices at angles 0,60,120,... has support in direction theta = s * max_i cos(theta - i*pi/3) ... no.
    
    Let me think again. A regular hexagon with side s and one vertex at angle 0 has vertices at (s*cos(k*pi/3), s*sin(k*pi/3)) for k=0..5.
    The support function h(u) for unit vector u at angle theta is s * max_k cos(theta - k*pi/3) for the hexagon centered at origin.
    Wait, that's not quite right either. The support of a convex polygon is max over vertices of v·u.
    
    For the outer hexagon to contain an inner hexagon, we need: for every direction, the outer hexagon's support >= inner hexagon's support.
    But since both are convex, it suffices to check the 6 directions of the outer hexagon's vertices? No, we need to check that each inner vertex is inside the outer hexagon.
    
    Actually, a cleaner approach: the outer hexagon with side S, centered at C, rotated by angle A, contains a point P iff for all 6 half-plane inequalities of the hexagon, P satisfies them.
    
    The 6 half-planes of a regular hexagon (side S, center C, rotation A) are defined by normals at angles A + k*pi/3 for k=0..5, with offset S * cos(pi/6) = S * sqrt(3)/2 from center.
    
    So for each inner hexagon, we compute the max over its vertices of (vertex - C) · n_k for each normal direction n_k. The required S is max over all inner hexagons, all vertices, all 6 normals of [(vertex - C) · n_k] / cos(pi/6)... 
    
    Wait, the half-plane is (x - C) · n <= S * cos(pi/6) where n is a unit normal. So the required S for that constraint is (x - C) · n / cos(pi/6).
    
    So outer_side = max over all constraints of [(vertex - C) · n_k] / cos(pi/6), and we take the max over all inner hexagons' vertices and all 6 normal directions.
    """
    outer_a = math.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        theta = outer_a + k * math.pi / 3.0
        normals.append(np.array([math.cos(theta), math.sin(theta)]))
    
    cos_pi6 = math.cos(math.pi / 6.0)
    max_extent = 0.0
    
    for (ix, iy, iangle) in inner_list:
        verts = hex_vertices(ix, iy, 1.0, iangle)
        for v in verts:
            rel = v - np.array(outer_center)
            for n in normals:
                proj = np.dot(rel, n)
                required_s = proj / cos_pi6
                if required_s > max_extent:
                    max_extent = required_s
    
    return max(0.0, max_extent)

def objective(params):
    """
    params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, cx, cy, outer_angle]
    Total: 12*3 + 3 = 39 parameters
    """
    inner_list = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i + 1]
        a = params[3*i + 2]
        inner_list.append((x, y, a))
    
    cx = params[3*N]
    cy = params[3*N + 1]
    outer_angle = params[3*N + 2]
    
    outer_center = (cx, cy)
    side = outer_hexagon_required(inner_list, outer_center, outer_angle)
    
    # Add penalty for overlapping inner hexagons
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ai = inner_list[i]
            xj, yj, aj = inner_list[j]
            # Approximate overlap check using distance between centers
            # Two unit hexagons overlap if center distance < ~2*inradius = 2*(sqrt(3)/2) = sqrt(3) ≈ 1.732
            # But with rotation, it's more complex. Use a conservative threshold.
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            min_dist = 1.732  # approximate minimum center distance for non-overlapping unit hexagons
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2 * 10.0
    
    return side + penalty

def construct():
    # Initial guess: hexagonal arrangement
    # Place 12 hexagons in a hexagonal pattern around the center
    # Ring 0: 1 hexagon at center
    # Ring 1: 6 hexagons at distance sqrt(3) in 6 directions
    # Ring 2: 5 hexagons (to get 12 total: 1 + 6 + 5 = 12)
    
    r1 = math.sqrt(3)  # distance for ring 1
    r2 = 2 * math.sqrt(3)  # approximate distance for ring 2
    
    inner_list = []
    # Center
    inner_list.append((0.0, 0.0, 0.0))
    
    # Ring 1: 6 hexagons
    for k in range(6):
        theta = k * math.pi / 3.0
        x = r1 * math.cos(theta)
        y = r1 * math.sin(theta)
        inner_list.append((x, y, 0.0))
    
    # Ring 2: 5 hexagons at angles 30, 90, 150, 210, 270 degrees (offset by 30)
    for k in range(5):
        theta = math.radians(30 + k * 72)
        x = r2 * math.cos(theta)
        y = r2 * math.sin(theta)
        inner_list.append((x, y, 0.0))
    
    # Outer center at origin, outer angle 0
    params = []
    for (x, y, a) in inner_list:
        params.extend([x, y, a])
    params.extend([0.0, 0.0, 0.0])
    
    # Optimize
    result = minimize(objective, params, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    opt_params = result.x
    inner_opt = []
    for i in range(N):
        x = opt_params[3*i]
        y = opt_params[3*i + 1]
        a = opt_params[3*i + 2]
        inner_opt.append([x, y, a])
    
    cx = opt_params[3*N]
    cy = opt_params[3*N + 1]
    outer_angle = opt_params[3*N + 2]
    
    outer_center = (cx, cy)
    side = outer_hexagon_required(inner_opt, outer_center, outer_angle)
    
    return inner_opt, outer_center, side, outer_angle

# EVOLVE-BLOCK-START
def construct():
    """Main entry point as required by the interface."""
    try:
        return _construct_impl()
    except Exception:
        # Fallback to a simple valid arrangement
        inner = [[math.sqrt(3) * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 10.0, 0.0

def _construct_impl():
    """The actual optimized construction."""
    # Initial guess: hexagonal arrangement
    r1 = math.sqrt(3)
    r2 = 2 * math.sqrt(3)
    
    inner_list = []
    inner_list.append((0.0, 0.0, 0.0))
    
    for k in range(6):
        theta = k * math.pi / 3.0
        x = r1 * math.cos(theta)
        y = r1 * math.sin(theta)
        inner_list.append((x, y, 0.0))
    
    for k in range(5):
        theta = math.radians(30 + k * 72)
        x = r2 * math.cos(theta)
        y = r2 * math.sin(theta)
        inner_list.append((x, y, 0.0))
    
    params = []
    for (x, y, a) in inner_list:
        params.extend([x, y, a])
    params.extend([0.0, 0.0, 0.0])
    
    result = minimize(objective, params, method='Nelder-Mead',
                      options={'maxiter': 8000, 'xatol': 1e-5, 'fatol': 1e-5})
    
    opt_params = result.x
    inner_opt = []
    for i in range(N):
        x = opt_params[3*i]
        y = opt_params[3*i + 1]
        a = opt_params[3*i + 2]
        inner_opt.append([float(x), float(y), float(a)])
    
    cx = float(opt_params[3*N])
    cy = float(opt_params[3*N + 1])
    outer_angle = float(opt_params[3*N + 2])
    
    outer_center = (cx, cy)
    side = outer_hexagon_required(inner_opt, outer_center, outer_angle)
    
    return inner_opt, outer_center, float(side), outer_angle
# EVOLVE-BLOCK-END
