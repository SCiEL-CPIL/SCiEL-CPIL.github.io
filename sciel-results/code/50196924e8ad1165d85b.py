"""Packing unit regular hexagons inside a regular hexagon using gradient-based optimization."""
import numpy as np
from scipy.optimize import minimize
import random

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, rot_deg):
    """Return 6 vertices of a regular hexagon (flat-top orientation at rot=0)."""
    theta = np.radians(rot_deg)
    angles = theta + np.pi / 6 + np.arange(6) * np.pi / 3
    r = side
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])

def _point_to_hexagon_dist(px, py, cx, cy, side, rot_deg):
    """
    Smooth approximate distance from a point (px,py) to the boundary of a hexagon.
    Uses the inradius-based radial distance as a smooth proxy.
    For a regular hexagon, the distance from center to edge (apothem) = side * sqrt(3)/2.
    We use |r - apothem| where r is radial distance, which is smooth and differentiable.
    """
    apothem = side * np.sqrt(3.0) / 2.0
    dx = px - cx
    dy = py - cy
    r = np.sqrt(dx * dx + dy * dy)
    # Smooth distance: positive when outside, negative when inside
    return r - apothem

def _hex_to_hex_clearance(vtxs_a, vtxs_b):
    """
    Minimum distance between two sets of vertices (proxy for hex-hex clearance).
    We compute min over all pairs of vertex-to-vertex distances, but for smoothness
    we use a soft-min. However, for the outer boundary constraint, we only need
    inner hex vertices vs outer boundary.
    """
    # For simplicity and differentiability, we'll handle outer boundary separately.
    # This function is a placeholder for inner-inner if needed.
    pass

def _outer_boundary_distance(px, py, ocx, ocy, oside, orot_deg):
    """
    Smooth distance from point (px,py) to the interior of the outer hexagon.
    Positive if inside, negative if outside. We want all inner vertices to be
    at least 0 inside (with margin for safety).
    
    For a regular hexagon centered at (ocx, ocy) with side length oside and rotation orot,
    the distance from center to each edge (apothem) is oside * sqrt(3)/2.
    
    We use the minimum of the 6 signed distances to the 6 edges.
    """
    apothem = oside * np.sqrt(3.0) / 2.0
    theta = np.radians(orot_deg)
    
    # Edge normals for a regular hexagon
    # For a flat-top hexagon (rot=0), the 6 edge outward normals are at angles 0, 60, 120, 180, 240, 300
    # Actually, for a regular hexagon with vertices at angles 30, 90, 150, 210, 270, 330 (pointy-top)
    # Let's use the same convention as _hex_vertices: vertices at theta + 30 + k*60 degrees
    # The edge normals (outward) are at theta + k*60 degrees for k=0..5
    # Wait, let me think more carefully.
    
    # A regular hexagon with "flat top" has vertices at angles 0, 60, 120, 180, 240, 300 from center
    # No, let's stick with our _hex_vertices definition:
    # vertices at theta + pi/6 + k*pi/3 = theta + 30 + k*60 degrees
    
    # The 6 edges connect consecutive vertices. The outward normal of each edge
    # points in the direction of the midpoint of the arc between two adjacent vertex angles.
    # Vertex k is at angle alpha_k = theta + 30 + k*60
    # Edge between vertex k and k+1 has outward normal at angle (alpha_k + alpha_{k+1})/2 = theta + 30 + k*60 + 30 = theta + 60 + k*60
    # So normals are at theta + 60 + k*60 for k=0..5, i.e., theta + k*60 + 60
    
    normal_angles = theta + np.pi / 3 + np.arange(6) * np.pi / 3
    
    # Distance from point to each edge line:
    # For an edge with outward normal n and the hexagon center at (ocx, ocy),
    # the edge line is: n . (p - c) = apothem
    # Signed distance (positive inside) = apothem - n . (p - c)
    
    dx = px - ocx
    dy = py - ocy
    
    min_dist = np.inf
    for k in range(6):
        nx = np.cos(normal_angles[k])
        ny = np.sin(normal_angles[k])
        # Distance from point to the edge line (signed, positive if on center side)
        dist = apothem - (nx * dx + ny * dy)
        if dist < min_dist:
            min_dist = dist
    
    return min_dist

def _objective(params, N):
    """
    Objective function: maximize the minimum clearance.
    We'll minimize the negative of the minimum clearance, or equivalently,
    find a configuration where all inner hex vertices are inside the outer hex.
    
    Actually, the task is to MINIMIZE the outer side length. So we should:
    - Fix the inner hexagon configuration (or optimize it too)
    - Minimize the outer side length such that all inner hexagons fit inside.
    
    Let me reformulate: We have inner hexagons at positions (x_i, y_i) with angles a_i.
    We want to find the minimum outer side length S such that all inner hexagons
    (with their 6 vertices) are inside the outer hexagon centered at (0,0) with side S.
    
    For a given inner configuration, the minimum S is determined by the inner hex vertex
    that is farthest from the center in terms of "hexagonal distance".
    
    Actually, the constraint is: for each inner hex vertex v, the distance from v to the
    outer hex boundary must be >= 0 (v is inside or on boundary).
    
    The minimum S for a given set of points is the smallest S such that all points are
    inside the outer hex of side S. This is a convex problem in S for fixed points.
    
    So the overall problem is: minimize S subject to all inner hex vertices being inside
    the outer hex of side S, and inner hexagons not overlapping.
    
    This is non-convex due to the overlap constraints and the inner configuration variables.
    
    Let's use a different approach: 
    - Parameterize: inner positions (x_i, y_i), inner angles (a_i), and outer side S.
    - Objective: minimize S.
    - Constraints: 
      1. All inner hex vertices are inside the outer hex (distance >= 0).
      2. Inner hexagons don't overlap (minimum distance between any two inner hex vertices
         from different hexagons >= some threshold, or use a smooth proxy).
    
    For differentiability, we'll use smooth approximations for the constraints.
    
    Actually, let's simplify: we'll optimize the inner configuration to maximize the
    minimum "hexagonal clearance" to the boundary, then compute the required S.
    
    Let me define:
    - For a given inner config, compute the minimum over all inner hex vertices of
      _outer_boundary_distance(v, 0, 0, S, 0) for a trial S.
    - We want this to be >= 0.
    - The minimum S is when the minimum distance is exactly 0.
    
    So for a given inner config, S_min = max over all inner vertices of the "hexagonal radius"
    of that vertex.
    
    The hexagonal radius of a point (x,y) is the smallest S such that (x,y) is inside
    the hexagon of side S centered at origin. This equals max over the 6 edge distances:
    S = max_k ( (n_k . (x,y)) / cos(0) ) ... let me think.
    
    For a hexagon of side S centered at origin with rotation 0, the apothem is S*sqrt(3)/2.
    The edge with normal at angle phi has equation: x*cos(phi) + y*sin(phi) = S*sqrt(3)/2.
    So for a point (x,y) to be inside, we need x*cos(phi) + y*sin(phi) <= S*sqrt(3)/2 for all 6 normals.
    Therefore S >= max_k ( (x*cos(phi_k) + y*sin(phi_k)) / (sqrt(3)/2) ) for all k where the value is positive.
    
    So S_min for a set of points = (2/sqrt(3)) * max over all points and all k of (x*cos(phi_k) + y*sin(phi_k)).
    
    This is a convex function of the point positions! So we can minimize S_min over the
    inner hexagon positions and angles, subject to non-overlap constraints.
    
    Let's implement this.
    """
    n = N
    # params layout: [x0, y0, a0, x1, y1, a1, ..., x_{n-1}, y_{n-1}, a_{n-1}]
    xs = params[0::3]
    ys = params[1::3]
    as_ = params[2::3]
    
    # Compute S_min for this configuration
    phi_k = np.arange(6) * np.pi / 3  # For rotation 0 outer hex
    cos_phi = np.cos(phi_k)
    sin_phi = np.sin(phi_k)
    
    max_val = 0.0
    for i in range(n):
        # Get vertices of inner hex i
        verts = _hex_vertices(xs[i], ys[i], 1.0, as_[i])
        for v in verts:
            px, py = v[0], v[1]
            vals = px * cos_phi + py * sin_phi
            max_val = max(max_val, np.max(vals))
    
    S_min = max_val * 2.0 / np.sqrt(3.0)
    
    # Add overlap penalty: for each pair of inner hexagons, compute a smooth
    # measure of overlap. We'll use the minimum distance between vertices of
    # different hexagons, and penalize if it's too small.
    # For a unit hexagon, the minimum distance between two non-overlapping hexagons
    # is 0 (they can touch). We want a smooth penalty that grows when they overlap.
    # 
    # Simple approach: for each pair, compute min vertex-vertex distance. If < threshold,
    # add a penalty. But min is not differentiable.
    # 
    # Alternative: use a soft-min. Or, since we're using L-BFGS-B, we can use a
    # penalty method with a smooth approximation.
    #
    # For now, let's just return S_min and handle overlaps via multi-start and
    # post-processing. This is a simplification.
    
    return S_min

def _hex_vertex_count(n):
    return n * 6

def construct():
    """
    Main entry point. Returns (inner, center, side, angle) where:
    - inner: list of [x, y, angle_degrees] for N unit hexagons
    - center: (x, y) of outer hexagon center
    - side: outer hexagon side length
    - angle: outer hexagon rotation in degrees
    """
    n = N
    
    # Initial configuration: hexagonal lattice arrangement
    # For a hexagonal packing of unit hexagons, the center-to-center distance
    # between adjacent hexagons is sqrt(3) (for flat-top orientation).
    
    # Let's start with a simple arrangement and then optimize.
    
    def _initial_config():
        """Generate initial hexagonal lattice arrangement."""
        # Hexagonal lattice: rows offset by half a hex width
        # For flat-top hexagons, horizontal spacing = 1.5, vertical spacing = sqrt(3)
        # Actually, for unit side length hexagons:
        # Width (flat-to-flat) = sqrt(3), height (point-to-point) = 2
        # For flat-top: width = 2, height = sqrt(3)
        # Horizontal distance between centers in same row = 1.5
        # Vertical distance between rows = sqrt(3) * 0.5 = sqrt(3)/2? 
        # Let me just use a simple grid and let optimization fix it.
        
        # Use a hexagonal close-packed arrangement
        # For N=11, we can do a hexagonal number arrangement or a rectangular grid.
        
        # Simple: place them in a roughly circular/hexagonal pattern
        # Start with a hexagonal lattice centered at origin
        
        xs = []
        ys = []
        as_ = []
        
        # Hexagonal lattice basis vectors
        # For flat-top hexagons, the lattice vectors are:
        # v1 = (1.5, 0), v2 = (0.75, sqrt(3)/2)
        a1 = 1.5
        a2x = 0.75
        a2y = np.sqrt(3.0) / 2.0
        
        # Generate hexagonal lattice points
        # For a hexagon of "radius" r (in lattice units), we have points
        # with |i| + |j| + |i+j| <= r or similar.
        
        # Let's just generate a diamond/hexagonal pattern
        max_r = 3  # This gives about 37 points, too many for N=11
        # For N=11, we need a smaller pattern.
        
        # Let's use a simple arrangement:
        # Center hexagon, surrounded by 6, then add 4 more.
        
        # Position 0: center
        xs.append(0.0)
        ys.append(0.0)
        as_.append(0.0)
        
        # 6 surrounding hexagons
        for k in range(6):
            angle = k * np.pi / 3
            r = np.sqrt(3.0)  # Distance to adjacent hex center
            xs.append(r * np.cos(angle))
            ys.append(r * np.sin(angle))
            as_.append(0.0)
        
        # 4 more hexagons: add to make 11
        # Add at radius 2*sqrt(3) in some directions
        for k in range(4):
            angle = k * np.pi / 2 + np.pi / 4
            r = 2.0 * np.sqrt(3.0)
            xs.append(r * np.cos(angle))
            ys.append(r * np.sin(angle))
            as_.append(0.0)
        
        return np.array(xs), np.array(ys), np.array(as_)
    
    def _compute_S_min(xs, ys, as_):
        """Compute minimum outer side length for given inner configuration."""
        phi_k = np.arange(6) * np.pi / 3
        cos_phi = np.cos(phi_k)
        sin_phi = np.sin(phi_k)
        
        max_val = 0.0
        for i in range(n):
            verts = _hex_vertices(xs[i], ys[i], 1.0, as_[i])
            for v in verts:
                px, py = v[0], v[1]
                vals = px * cos_phi + py * sin_phi
                max_val = max(max_val, np.max(vals))
        
        return max_val * 2.0 / np.sqrt(3.0)
    
    def _overlap_penalty(xs, ys, as_, threshold=0.05):
        """
        Compute a smooth penalty for overlapping inner hexagons.
        For each pair, compute the minimum distance between their vertices.
        If the min distance < threshold, add a penalty.
        """
        penalty = 0.0
        for i in range(n):
            verts_i = _hex_vertices(xs[i], ys[i], 1.0, as_[i])
            for j in range(i + 1, n):
                verts_j = _hex_vertices(xs[j], ys[j], 1.0, as_[j])
                # Compute min distance between any vertex of i and any vertex of j
                min_dist = np.inf
                for vi in verts_i:
                    for vj in verts_j:
                        d = np.sqrt((vi[0] - vj[0])**2 + (vi[1] - vj[1])**2)
                        if d < min_dist:
                            min_dist = d
                if min_dist < threshold:
                    # Quadratic penalty
                    penalty += (threshold - min_dist) ** 2
        return penalty
    
    def _objective_full(params):
        """Full objective: S_min + overlap penalty."""
        xs = params[0::3]
        ys = params[1::3]
        as_ = params[2::3]
        
        S_min = _compute_S_min(xs, ys, as_)
        penalty = _overlap_penalty(xs, ys, as_)
        
        return S_min + 100.0 * penalty
    
    # Multi-start optimization
    best_S = np.inf
    best_params = None
    
    # Initial configuration
    xs0, ys0, as0 = _initial_config()
    
    # Try multiple random perturbations
    np.random.seed(42)
    random.seed(42)
    
    n_starts = 10
    for start in range(n_starts):
        if start == 0:
            xs_s = xs0.copy()
            ys_s = ys0.copy()
            as_s = as0.copy()
        else:
            # Random perturbation
            xs_s = xs0 + np.random.randn(n) * 0.5
            ys_s = ys0 + np.random.randn(n) * 0.5
            as_s = as0 + np.random.randn(n) * 15.0  # degrees
        
        params0 = np.zeros(3 * n)
        params0[0::3] = xs_s
        params0[1::3] = ys_s
        params0[2::3] = as_s
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _objective_full,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.success or result.fun < best_S:
                S_val = _compute_S_min(result.x[0::3], result.x[1::3], result.x[2::3])
                pen_val = _overlap_penalty(result.x[0::3], result.x[1::3], result.x[2::3])
                if pen_val < 1e-4:  # Only accept if no significant overlap
                    if S_val < best_S:
                        best_S = S_val
                        best_params = result.x.copy()
        except Exception as e:
            pass
    
    # If optimization didn't find a good solution, use the initial configuration
    if best_params is None:
        xs_final = xs0
        ys_final = ys0
        as_final = as0
        side = _compute_S_min(xs_final, ys_final, as_final)
    else:
        xs_final = best_params[0::3]
        ys_final = best_params[1::3]
        as_final = best_params[2::3]
        side = _compute_S_min(xs_final, ys_final, as_final)
    
    # Build inner list
    inner = []
    for i in range(n):
        inner.append([float(xs_final[i]), float(ys_final[i]), float(as_final[i])])
    
    return inner, (0.0, 0.0), float(side), 0.0

# EVOLVE-BLOCK-END
