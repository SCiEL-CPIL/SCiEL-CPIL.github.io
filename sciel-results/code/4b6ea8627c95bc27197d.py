"""Improved program: analytical objective + multi-start Powell + coordinate polish."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Precompute hexagon vertices for unit hexagon (circumradius = 1, inradius = sqrt(3)/2)
# A regular hexagon with side length 1 has vertices at distance 1 from center.
# Default orientation: flat-top or pointy-top? Let's use pointy-top (vertex at angle 0).
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees.
_HEX_VERTS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
# Inradius (apothem) of unit hexagon
_INRADIUS = math.sqrt(3) / 2.0

# For the outer hexagon, we need to compute the minimum side length such that all inner
# hexagons fit inside. The outer hexagon has center (cx, cy), side length S, and rotation phi.
# A regular hexagon with side length S has circumradius S and inradius S*sqrt(3)/2.
# The six supporting lines (normals) of the outer hexagon are at angles phi, phi+60, ..., phi+300.
# For each inner hexagon with center (xi, yi) and rotation theta, its vertices are at
# (xi + cos(theta + a_k), yi + sin(theta + a_k)) for k=0..5.
# The constraint is that for each of the 6 normals of the outer hexagon, the maximum projection
# of any inner hexagon vertex onto that normal must be <= inradius of outer = S * sqrt(3)/2.
# Also, the projection of the inner hexagon center plus its own inradius in the normal direction...
# Actually, the correct constraint: for a convex polygon inside another, it's enough to check
# that each vertex of the inner is inside the outer. But for hexagons, we can use the fact that
# the Minkowski sum / support function approach: the outer hexagon contains the inner hexagon
# iff for each of the 6 outward normals of the outer hexagon, the support function of the inner
# hexagon in that direction + projection of inner center <= inradius of outer.
# The support function of a unit hexagon (centered at origin) in direction u (unit vector) is
# the max over its vertices of dot(vertex, u). For a hexagon rotated by theta, this is
# max_k dot(R(theta) * v_k, u) = max_k dot(v_k, R(-theta) * u).
# Since the hexagon is symmetric, the support function in any direction depends only on the
# angle of the direction relative to the hexagon's orientation.

def _support_hex(theta, u_angle):
    """Support function of unit hexagon rotated by theta in direction u_angle (radians)."""
    # The support function of a regular hexagon with circumradius 1 in direction at angle alpha
    # relative to the hexagon's first vertex direction is:
    # max over k of cos(alpha - k*pi/3) for k=0..5
    # This equals the maximum of cos(alpha - k*pi/3) which is the same as the distance from
    # center to the supporting line in that direction.
    # For a regular hexagon, the support function h(alpha) = 1 / cos((alpha mod pi/3) - pi/6) ... 
    # Actually let's just compute it directly.
    rel = u_angle - theta
    # Normalize rel to [-pi/6, pi/6] relative to nearest vertex
    # The support function of a unit hexagon (vertices at distance 1) in direction alpha
    # is max_{k=0..5} cos(alpha - k*pi/3)
    # Due to symmetry, this is periodic with period pi/3.
    # Let beta = rel mod (pi/3), then h = max(cos(beta), cos(beta - pi/3))... 
    # Actually simpler: h(alpha) for unit hexagon = max over 6 vertices.
    # But we can derive: for alpha in [0, pi/6], the max is cos(alpha) (vertex 0 dominates)
    # for alpha in [pi/6, pi/2], the max is cos(alpha - pi/3)... 
    # Let's just use the formula: h(alpha) = 1 / cos(d) where d is the angular distance to the
    # nearest "flat" direction. The flats are at angles pi/6, pi/6+pi/3, etc.
    # Actually, the support function of a regular hexagon with inradius r and circumradius R:
    # h(alpha) = R * max_k cos(alpha - k*pi/3)
    # For unit hexagon, R=1.
    # The max of cos(alpha - k*pi/3) over k is achieved at the k closest to alpha.
    # Let k* = round(alpha / (pi/3)) mod 6
    # Then h = cos(alpha - k* * pi/3)
    k = int(round(rel / (math.pi / 3.0))) % 6
    return math.cos(rel - k * math.pi / 3.0)

def compute_outer_side(inner_positions, inner_angles_deg, outer_center, outer_rotation_deg):
    """Compute the minimum outer hexagon side length to contain all inner hexagons.
    inner_positions: (N, 2) array
    inner_angles_deg: (N,) array
    outer_center: (2,) array
    outer_rotation_deg: float
    Returns: outer side length S (or inf if infeasible, but it's always feasible for large S)
    """
    cx, cy = outer_center
    phi = math.radians(outer_rotation_deg)
    
    # The 6 outward normals of the outer hexagon are at angles phi + k*60 for k=0..5
    # For each normal direction, the constraint is:
    #   proj(outer_center, n) + max over inner hexagons of [proj(inner_center - outer_center, n) + support(inner_hex, n)] <= S * sqrt(3)/2
    # Wait, let me think again. The outer hexagon centered at (cx,cy) with rotation phi has
    # supporting lines: n_k . (x - c) = S * sqrt(3)/2 for each of 6 normals n_k.
    # So a point p is inside iff n_k . (p - c) <= S * sqrt(3)/2 for all k.
    # For an inner hexagon with center c_i and rotation theta_i, all its points p satisfy
    # n_k . (p - c) = n_k . (c_i - c) + n_k . (p - c_i) <= n_k . (c_i - c) + h_i(n_k)
    # where h_i is the support function of the unit hexagon at rotation theta_i.
    # So the constraint is: n_k . (c_i - c) + h_i(n_k) <= S * sqrt(3)/2 for all k, i.
    # Thus S >= (n_k . (c_i - c) + h_i(n_k)) / (sqrt(3)/2) for all k, i.
    # S_min = max over k, i of [n_k . (c_i - c) + h_i(n_k)] / (sqrt(3)/2)
    
    inradius_factor = math.sqrt(3) / 2.0
    max_val = 0.0
    
    for k in range(6):
        n_angle = phi + k * math.pi / 3.0
        nx = math.cos(n_angle)
        ny = math.sin(n_angle)
        
        for i in range(N):
            dx = inner_positions[i, 0] - cx
            dy = inner_positions[i, 1] - cy
            proj_center = nx * dx + ny * dy
            theta_i = math.radians(inner_angles_deg[i])
            h = _support_hex(theta_i, n_angle)
            val = proj_center + h
            if val > max_val:
                max_val = val
    
    S = max_val / inradius_factor
    # S must be at least 1 (side length of inner hexagons) but that's automatically satisfied
    return S

def cost_func(params):
    """Objective function: minimize outer side length.
    params: flat array of [x0, y0, a0, x1, y1, a1, ..., xN-1, yN-1, aN-1, cx, cy, phi]
    Total: 3*N + 3 parameters.
    """
    positions = params[:3*N].reshape(N, 2)
    angles = params[3*N:3*N+N]
    cx = params[3*N+N]
    cy = params[3*N+N+1]
    phi = params[3*N+N+2]
    
    return compute_outer_side(positions, angles, (cx, cy), phi)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_cost = float('inf')
    best_params = None
    
    # Multi-start Powell optimization
    # Initial configurations:
    starts = []
    
    # Start 1: Linear arrangement along x-axis (like seed but optimized)
    p1 = np.zeros(3*N + 3)
    for k in range(N):
        idx = k - N // 2
        p1[3*k] = 2.0 * idx  # spacing of 2 (touching)
        p1[3*k+1] = 0.0
        p1[3*k+2] = 0.0
    p1[3*N] = 0.0  # cx
    p1[3*N+1] = 0.0  # cy
    p1[3*N+2] = 0.0  # phi
    starts.append(p1)
    
    # Start 2: Hexagonal packing arrangement
    p2 = np.zeros(3*N + 3)
    # Place in a hexagonal grid pattern
    hex_positions = []
    # Center at origin, then surrounding
    hex_positions.append((0, 0))
    # 6 neighbors
    for k in range(6):
        angle = k * math.pi / 3
        hex_positions.append((2 * math.cos(angle), 2 * math.sin(angle)))
    # That's 7. Need 11 more... let's just use a simple pattern
    # Actually, let's use a different arrangement: two rows
    for k in range(N):
        if k < 6:
            p2[3*k] = k * 2.0
            p2[3*k+1] = 0.0
        else:
            p2[3*k] = (k-6) * 2.0 + 1.0
            p2[3*k+1] = math.sqrt(3)
        p2[3*k+2] = 0.0
    p2[3*N] = 0.0
    p2[3*N+1] = 0.0
    p2[3*N+2] = 0.0
    starts.append(p2)
    
    # Start 3: Random
    np.random.seed(42)
    p3 = np.random.randn(3*N + 3) * 5.0
    p3[3*N+2] = 0.0
    starts.append(p3)
    
    # Start 4: All at origin with random angles
    p4 = np.zeros(3*N + 3)
    p4[3*N+2] = 0.0
    for k in range(N):
        p4[3*k+2] = np.random.uniform(0, 60)
    starts.append(p4)
    
    for start_idx, x0 in enumerate(starts):
        if time.time() > deadline - 30.0:
            break
        
        # Powell method
        try:
            result = minimize(cost_func, x0, method='Powell', 
                            options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-10})
            if result.fun < best_cost:
                best_cost = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
        
        # Coordinate descent polish
        if best_params is not None:
            if time.time() > deadline - 20.0:
                break
            
            params = best_params.copy()
            n_params = len(params)
            step_sizes = [1.0, 0.1, 0.01, 0.001, 0.0001]
            improved = True
            for ss in step_sizes:
                if time.time() > deadline - 10.0:
                    break
                improved = False
                for _ in range(50):
                    if time.time() > deadline - 10.0:
                        break
                    for j in range(n_params):
                        old_val = params[j]
                        # Try small steps
                        for delta in [ss, -ss]:
                            params[j] = old_val + delta
                            c = cost_func(params)
                            if c < best_cost:
                                best_cost = c
                                best_params = params.copy()
                                improved = True
                            else:
                                params[j] = old_val
                    if not improved:
                        break
    
    if best_params is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract results
    positions = best_params[:3*N].reshape(N, 2)
    angles = best_params[3*N:3*N+N]
    cx = best_params[3*N+N]
    cy = best_params[3*N+N+1]
    phi = best_params[3*N+N+2]
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles[i])])
    
    return inner, (float(cx), float(cy)), float(best_cost), float(phi)

# EVOLVE-BLOCK-START
# (The construct function is defined above)
# EVOLVE-BLOCK-END
