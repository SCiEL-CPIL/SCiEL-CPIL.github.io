"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import sys

N = 12
SQRT3 = np.sqrt(3.0)
APOTHEM = SQRT3 / 2.0

def construct():
    """
    Place 12 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Returns: (inner, center, side, angle)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Vertices of a unit regular hexagon centered at origin, rotated by angle_deg
    # Hexagon with flat top: vertices at angles 0, 60, 120, 180, 240, 300
    def hex_vertices(angle_deg):
        a = np.radians(angle_deg)
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        # Base vertices (unit hexagon, pointy-top orientation has vertices at 30, 90, 150, 210, 270, 330)
        # Actually let's use the standard: vertices at k*60 degrees for k=0..5
        # For a regular hexagon with side length 1, the vertices are at distance 1 from center
        # Let's use: vertex k at angle (60*k) degrees from center, radius 1
        base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
        cos_base = np.cos(base_angles)
        sin_base = np.sin(base_angles)
        # Rotate
        x = cos_base * cos_a - sin_base * sin_a
        y = cos_base * sin_a + sin_base * cos_a
        return np.column_stack([x, y])
    
    # Precompute base vertices for various angles (we'll rotate in function)
    # Actually, let's just compute on the fly since N is small
    
    def get_hex_corners(cx, cy, angle_deg):
        """Get 6 corners of unit hexagon centered at (cx, cy) with rotation angle_deg."""
        a = np.radians(angle_deg)
        base_angles = np.arange(6) * np.pi / 3.0 + a
        corners = np.column_stack([cx + np.cos(base_angles), cy + np.sin(base_angles)])
        return corners
    
    def compute_required_side(hexes):
        """
        Given list of (cx, cy, angle_deg), compute the minimum side length of a regular hexagon
        centered at (0,0) with rotation angle 0 that contains all hexes.
        
        The outer hexagon has side length s. Its vertices are at distance s from center,
        at angles 0, 60, 120, 180, 240, 300.
        
        For a point (x, y) to be inside a regular hexagon of side length s centered at origin
        with the given orientation, we need to check the 6 half-plane constraints.
        
        The hexagon can be described as the intersection of 6 half-planes.
        For a hexagon with "flat" sides (edges perpendicular to directions 30, 90, 150, 210, 270, 330),
        the constraints are:
          |x * cos(theta_k) + y * sin(theta_k)| <= s * cos(30 deg) = s * sqrt(3)/2
        where theta_k = 30 + 60*k degrees for k=0..5? 
        
        Let me think more carefully. A regular hexagon with side length s has:
        - Vertices at distance s from center
        - Apothem (distance from center to side midpoint) = s * sqrt(3)/2
        
        If we orient it so one side is horizontal at the top, the normal directions to the sides
        are at angles 0, 60, 120, 180, 240, 300 (pointing outward from center to side midpoints).
        
        Wait, let me use a cleaner formulation. A regular hexagon with side length s centered at origin,
        with one vertex pointing right (angle 0), has vertices at:
          (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
        
        The 6 edges have outward normals at angles 30, 90, 150, 210, 270, 330 degrees.
        The distance from center to each edge (apothem) is s * sqrt(3)/2.
        
        So a point (x,y) is inside if and only if:
          For each k=0..5: x*cos(theta_k) + y*sin(theta_k) <= s*sqrt(3)/2
        where theta_k = 30 + 60*k degrees.
        
        To contain a point, we need s >= (x*cos(theta_k) + y*sin(theta_k)) * 2/sqrt(3) for all k.
        Also s >= max over all points of this quantity.
        
        But we also need to handle the case where the point is on the other side, so we use abs:
        s >= max_k (|x*cos(theta_k) + y*sin(theta_k)| * 2/sqrt(3))
        
        Actually no, the constraint is one-sided for each normal direction. The hexagon is the intersection
        of 6 half-planes: n_k . p <= apothem for k=0..5. Since the normals point outward, and the hexagon
        is symmetric, we can write it as |n_k . p| <= apothem where n_k are unit vectors at angles 0, 60, 120, 180, 240, 300? 
        
        Let me just be explicit. The 6 outward normal directions (from center to side midpoints) for a hexagon
        with a vertex at angle 0 are at angles 30, 90, 150, 210, 270, 330.
        
        For each such direction theta_k, the constraint is:
          x * cos(theta_k) + y * sin(theta_k) <= s * sqrt(3) / 2
        
        This is equivalent to:
          s >= (x * cos(theta_k) + y * sin(theta_k)) * 2 / sqrt(3)
        
        Since the hexagon is convex and contains the origin, and the normals cover all directions,
        the maximum over all 6 constraints gives the required s.
        
        So for a set of points, s_required = max over all points p and all k of:
          (p.x * cos(theta_k) + p.y * sin(theta_k)) * 2 / sqrt(3)
        
        where theta_k = (30 + 60*k) degrees, k=0..5.
        """
        # Normal directions
        theta_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
        cos_thetas = np.cos(theta_angles)
        sin_thetas = np.sin(theta_angles)
        
        max_s = 0.0
        for cx, cy, ang in hexes:
            corners = get_hex_corners(cx, cy, ang)
            # For each corner, compute the required s
            for px, py in corners:
                # s must satisfy: px*cos(th) + py*sin(th) <= s * sqrt(3)/2
                # => s >= (px*cos(th) + py*sin(th)) * 2/sqrt(3)
                vals = (px * cos_thetas + py * sin_thetas) * 2.0 / SQRT3
                s_needed = np.max(vals)
                if s_needed > max_s:
                    max_s = s_needed
        return max_s
    
    def objective(params):
        """
        params: flat array of 3*N values: [cx_0, cy_0, ang_0, cx_1, cy_1, ang_1, ...]
        Returns the required side length.
        """
        hexes = []
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i + 1]
            ang = params[3*i + 2]
            hexes.append((cx, cy, ang))
        
        # Check for overlap (penalty if overlapping)
        # For two unit hexagons to be disjoint, the distance between centers must be at least 2
        # (since the maximum distance from center to any point in a unit hexagon is 1, so two hexagons
        # can be disjoint if center distance >= 2, but they might still overlap if closer)
        # Actually, the minimum distance between centers for two unit hexagons to be disjoint depends on orientation.
        # The worst case (most overlap-prone) is when they're aligned, requiring distance >= 2.
        # For a simple check, let's use distance >= 2 - small_epsilon as a necessary condition.
        # But this is not sufficient. Let's do a more careful check.
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = hexes[i][0] - hexes[j][0]
                dy = hexes[i][1] - hexes[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                # Two unit hexagons definitely overlap if dist < 1 (since each has inradius sqrt(3)/2 ~ 0.866,
                # so if centers are closer than 1, they definitely overlap). 
                # They definitely don't overlap if dist >= 2.
                # In between, it depends on orientation.
                if dist < 1.0:
                    overlap_penalty += (1.0 - dist) * 100.0  # Heavy penalty
                elif dist < 2.0:
                    # Check more carefully: the minimum distance between any point in hex_i and any point in hex_j
                    # This is expensive, so use a simpler bound: the sum of the "reach" in the direction between centers
                    # A simpler approach: if dist < 2, there might be overlap. Let's use a penalty that's zero at dist=2
                    # and increases as dist decreases.
                    # The inradius of a unit hexagon is sqrt(3)/2 ≈ 0.866. So the minimum center distance for
                    # non-overlap (in the worst orientation) is 2 * inradius = sqrt(3) ≈ 1.732? No, that's for circles.
                    # For hexagons, the maximum distance from center to any point is 1 (the vertex). So if dist < 2,
                    # vertices could overlap. Let's just use a soft penalty.
                    overlap_penalty += (2.0 - dist) * 50.0
        
        s_required = compute_required_side(hexes)
        return s_required + overlap_penalty
    
    # Initial guess: place hexagons in a honeycomb pattern
    # In a honeycomb, the centers are spaced by 2*apothem = sqrt(3) horizontally and ... 
    # Actually, for non-overlapping unit hexagons in a honeycomb, the center-to-center distance
    # in the three hex grid directions is 2 * apothem = sqrt(3) ≈ 1.732? No.
    # Two adjacent hexagons share an edge. The distance between their centers is 2 * apothem = sqrt(3).
    # Wait, apothem = sqrt(3)/2, so 2*apothem = sqrt(3) ≈ 1.732.
    
    # Let's place them in a triangular/honeycomb arrangement
    initial_params = np.zeros(3 * N)
    # Honeycomb: place in rows
    # Row offset: each row is shifted by sqrt(3)/2 horizontally, and separated by 1.5 vertically?
    # For unit hexagons (flat-top), the honeycomb spacing is:
    # Horizontal distance between adjacent centers: sqrt(3)
    # Vertical distance between rows: 1.5
    # Let's just use a simple grid for now and let optimization fix it.
    
    # Place in a 4x3 grid or similar
    positions = []
    # 12 hexagons: 4 columns, 3 rows
    for row in range(3):
        for col in range(4):
            cx = col * SQRT3 + (row % 2) * SQRT3 / 2
            cy = row * 1.5
            positions.append((cx - 1.5 * SQRT3, cy - 1.5, 0.0))
    
    for i, (cx, cy, ang) in enumerate(positions):
        initial_params[3*i] = cx
        initial_params[3*i + 1] = cy
        initial_params[3*i + 2] = ang
    
    best_result = None
    best_score = float('inf')
    
    # Try a few different initial configurations
    configs = [
        # Honeycomb-like
        lambda i: (
            (i % 4) * SQRT3 + (i // 4) % 2 * SQRT3 / 2 - 1.5 * SQRT3,
            (i // 4) * 1.5 - 1.5,
            0.0
        ),
        # Line
        lambda i: (i * 2.0 - 11.0, 0.0, 0.0),
        # Grid 4x3
        lambda i: (
            (i % 4) * 2.0 - 3.0,
            (i // 4) * 2.0 - 2.0,
            0.0
        ),
        # Clustered
        lambda i: (
            (i % 3 - 1) * 1.9,
            (i // 3 - 1) * 1.9,
            0.0
        ),
    ]
    
    for cfg_idx, config_fn in enumerate(configs):
        if time.time() > deadline:
            break
        
        params = np.zeros(3 * N)
        for i in range(N):
            cx, cy, ang = config_fn(i)
            params[3*i] = cx
            params[3*i + 1] = cy
            params[3*i + 2] = ang
        
        # Use differential evolution for global search
        bounds = [(-5, 5), (-5, 5), (-180, 180)] * N
        
        # Limit iterations based on time
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            continue
        
        max_iter = min(200, int(time_remaining * 5))  # Rough estimate
        if max_iter < 10:
            continue
        
        result = differential_evolution(
            objective,
            bounds,
            max_iter=max_iter,
            popsize=15,
            tol=0.01,
            seed=42 + cfg_idx,
            polish=False,
            init='sobol',
        )
        
        if result.success or result.fun < best_score:
            best_score = result.fun
            best_result = result.x
        
        # Polish with local optimization
        if best_result is not None and time.time() < deadline - 5:
            try:
                local_result = minimize(
                    objective,
                    best_result,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if local_result.fun < best_score:
                    best_score = local_result.fun
                    best_result = local_result.x
            except Exception:
                pass
    
    # If no result found, use initial guess
    if best_result is None:
        best_result = initial_params
        best_score = objective(best_result)
    
    # Extract final hexagon positions
    inner = []
    for i in range(N):
        cx = best_result[3*i]
        cy = best_result[3*i + 1]
        ang = best_result[3*i + 2]
        inner.append([cx, cy, ang])
    
    # Compute the actual required side length (without penalty)
    hexes = [(inner[i][0], inner[i][1], inner[i][2]) for i in range(N)]
    side_length = compute_required_side(hexes)
    
    # The outer hexagon is centered at origin with rotation 0
    return inner, (0.0, 0.0), side_length, 0.0

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
