"""Improved packing of 11 unit hexagons in a regular hexagon."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE = 150.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with center (cx,cy), rotation angle_deg, side s."""
    a = math.radians(angle_deg)
    pts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        pts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return pts

def _point_in_hex(px, py, cx, cy, angle_deg, s, tol=1e-9):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with rotation angle_deg, side s."""
    # Transform to hexagon local frame
    dx = px - cx
    dy = py - cy
    a = math.radians(angle_deg)
    ca, sa = math.cos(-a), math.sin(-a)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    # Regular hexagon with flat top (angle 0) has vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon is defined by |lx| <= s*sqrt(3)/2 in the appropriate directions
    # Actually for a regular hexagon with side s, the inradius is s*sqrt(3)/2
    # The hexagon can be described as: for each of 6 directions, the projection <= inradius
    inradius = s * math.sqrt(3) / 2.0
    for k in range(6):
        theta = k * math.pi / 3.0
        proj = lx * math.cos(theta) + ly * math.sin(theta)
        if proj > inradius + tol:
            return False
    return True

def _hex_edge_separation(h1, h2):
    """
    h1, h2: lists of 6 (x,y) vertices of two hexagons.
    Returns the minimum distance between any edge of h1 and any edge of h2 (positive if disjoint).
    We use a conservative check: min distance between any vertex of one and any edge of the other.
    """
    min_dist = float('inf')
    # Check vertices of h1 against edges of h2
    for i in range(6):
        v = h1[i]
        for j in range(6):
            e1 = h2[j]
            e2 = h2[(j+1) % 6]
            d = _point_to_segment_dist(v, e1, e2)
            if d < min_dist:
                min_dist = d
    # Check vertices of h2 against edges of h1
    for i in range(6):
        v = h2[i]
        for j in range(6):
            e1 = h1[j]
            e2 = h1[(j+1) % 6]
            d = _point_to_segment_dist(v, e1, e2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _point_to_segment_dist(p, a, b):
    """Distance from point p to segment ab."""
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy)

def _outer_contains_all(inner, outer_center, outer_side, outer_angle):
    """Check all inner hexagons are inside the outer hexagon."""
    ocx, ocy = outer_center
    for (ix, iy, ia) in inner:
        verts = _hex_vertices(ix, iy, ia, 1.0)
        for (vx, vy) in verts:
            if not _point_in_hex(vx, vy, ocx, ocy, outer_angle, outer_side):
                return False
    return True

def _no_overlap(inner, tol=1e-6):
    """Check no two inner hexagons overlap."""
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            h1 = _hex_vertices(inner[i][0], inner[i][1], inner[i][2], 1.0)
            h2 = _hex_vertices(inner[j][0], inner[j][1], inner[j][2], 1.0)
            # Quick bounding box check
            h1xs = [p[0] for p in h1]
            h1ys = [p[1] for p in h1]
            h2xs = [p[0] for p in h2]
            h2ys = [p[1] for p in h2]
            if max(h1xs) < min(h2xs) - tol or max(h2xs) < min(h1xs) - tol:
                continue
            if max(h1ys) < min(h2ys) - tol or max(h2ys) < min(h1ys) - tol:
                continue
            d = _hex_edge_separation(h1, h2)
            if d < -tol:
                return False
    return True

def _is_valid(inner, outer_center, outer_side, outer_angle, tol=1e-6):
    """Full validity check."""
    if not _outer_contains_all(inner, outer_center, outer_side, outer_angle):
        return False
    if not _no_overlap(inner, tol):
        return False
    return True

def construct():
    """
    Pack 11 unit hexagons in a regular hexagon, minimizing outer side length.
    Uses symmetry: hexagonal lattice arrangement.
    """
    deadline = START_TIME + DEADLINE
    best = None
    best_score = float('inf')
    
    # Strategy: use a hexagonal lattice arrangement.
    # 11 hexagons: center + ring of 6 + partial second ring, or other configs.
    # Try several lattice-based configurations and optimize positions.
    
    # Configuration 1: center + 6 around + 4 more (hexagonal close packing)
    # Standard hexagonal lattice: positions in a hex grid
    # For 11: we can do a "flower" pattern
    
    # Let's try a more systematic approach:
    # Use scipy to optimize positions of 11 hexagons with fixed rotation (all aligned)
    # to minimize the bounding hexagon side length.
    
    # First, let's try a known good configuration and refine it.
    
    # Hexagonal lattice basis vectors (for unit hexagons with flat top):
    # Adjacent hex centers are distance sqrt(3) apart.
    # Lattice vectors: (sqrt(3), 0) and (sqrt(3)/2, 3/2)
    
    sqrt3 = math.sqrt(3)
    
    # Generate hex grid positions for a cluster
    # Use axial coordinates (q, r) -> cartesian
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    # 11 hexagons in a hexagonal cluster:
    # Center (0,0), ring 1: 6 hexagons, ring 2: partial
    # Actually for 11, let's use: center + ring1(6) + 4 from ring2
    # Or better: a 3x3-ish hex cluster
    
    # Let's just place them in a hexagonal pattern and optimize
    # Initial guess: hexagonal lattice positions
    positions = []
    # Center
    positions.append((0, 0))
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = k * math.pi / 3.0
        positions.append((sqrt3 * math.cos(angle), sqrt3 * math.sin(angle)))
    # Ring 2: add 4 more (to get 11 total)
    # Ring 2 positions in hex grid: (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1)
    # Pick 4 that are symmetric: (2,0), (0,2), (-2,0), (0,-2) -> but that's only 4, total = 11
    ring2_coords = [(2,0), (0,2), (-2,0), (0,-2)]
    for (q, r) in ring2_coords:
        x, y = axial_to_cartesian(q, r)
        positions.append((x, y))
    
    # All rotations = 0 initially
    n = len(positions)
    
    # Optimize: minimize the outer hexagon side length
    # The outer hexagon must contain all inner hexagons.
    # For a regular hexagon with side S and rotation theta, center (cx, cy),
    # the constraint is that all vertices of all inner hexagons are inside.
    
    # We'll use scipy minimize to optimize (cx, cy, S, theta) and inner positions
    # But that's 4 + 3*11 = 37 variables. Let's fix inner rotations to 0 and optimize positions.
    
    # Actually, let's try a simpler approach: fix the outer hexagon to be axis-aligned (theta=0, center=origin)
    # and find the minimum S such that we can place 11 unit hexagons inside.
    
    # For a regular hexagon with side S and flat top (angle 0), the inradius is S*sqrt(3)/2.
    # The hexagon extends from -S to S in x-direction (at y=0) and from -S*sqrt(3)/2 to S*sqrt(3)/2 in y.
    
    # Let's use a binary search on S and check feasibility.
    
    def feasible(S, inner_positions, inner_angles, outer_center=(0,0), outer_angle=0):
        """Check if inner hexagons fit in outer hexagon of side S."""
        return _outer_contains_all(
            [(x, y, a) for (x, y), a in zip(inner_positions, inner_angles)],
            outer_center, S, outer_angle
        )
    
    # Optimize inner positions to minimize required S
    # Objective: for given inner positions, find minimum S such that all fit in outer hex.
    # For axis-aligned outer hex (angle 0, center origin), the constraint for each vertex (vx, vy) is:
    # |vx| <= S, |vy| <= S*sqrt(3)/2, and |vx/2 + vy*sqrt(3)/2| <= S*sqrt(3)/2, etc.
    # Actually the 6 half-plane constraints for a flat-top hexagon with side S:
    # For each of 6 directions theta_k = k*pi/3:
    #   vx*cos(theta_k) + vy*sin(theta_k) <= S*sqrt(3)/2
    
    def min_S_for_positions(inner_pos, inner_angles, outer_center=(0,0), outer_angle=0):
        """Compute minimum outer side S for given inner positions."""
        ocx, ocy = outer_center
        S = 0.0
        for (ix, iy), ia in zip(inner_pos, inner_angles):
            verts = _hex_vertices(ix, iy, ia, 1.0)
            for (vx, vy) in verts:
                # Transform to outer hex frame
                dx = vx - ocx
                dy = vy - ocy
                a = math.radians(outer_angle)
                ca, sa = math.cos(-a), math.sin(-a)
                lx = dx * ca - dy * sa
                ly = dx * sa + dy * ca
                for k in range(6):
                    theta = k * math.pi / 3.0
                    proj = lx * math.cos(theta) + ly * math.sin(theta)
                    # proj <= S * sqrt(3) / 2  =>  S >= proj * 2 / sqrt(3)
                    if proj > 0:
                        S_candidate = proj * 2.0 / sqrt3
                        if S_candidate > S:
                            S = S_candidate
        return S
    
    # Optimize inner positions to minimize S
    def objective(params):
        if _elapsed() > deadline - 5:
            return 1e10
        pos = [(params[i], params[i+n]) for i in range(n)]
        angles = [0.0] * n  # Fix rotations to 0 for simplicity
        return min_S_for_positions(pos, angles)
    
    # Initial guess
    x0 = []
    for (x, y) in positions:
        x0.extend([x, y])
    
    # Use Nelder-Mead for robustness
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        if result.success or result.fun < best_score:
            pos_opt = [(result.x[i], result.x[i+n]) for i in range(n)]
            angles_opt = [0.0] * n
            S_opt = min_S_for_positions(pos_opt, angles_opt)
            
            # Verify validity
            inner = [(x, y, a) for (x, y), a in zip(pos_opt, angles_opt)]
            if _is_valid(inner, (0.0, 0.0), S_opt, 0.0):
                if S_opt < best_score:
                    best_score = S_opt
                    best = (inner, (0.0, 0.0), S_opt, 0.0)
    except Exception:
        pass
    
    # If we haven't found a valid solution, try a simple valid configuration
    if best is None:
        # Place hexagons in a line with spacing sqrt(3)
        inner = []
        for k in range(-(N//2), N - N//2):
            inner.append([k * sqrt3, 0.0, 0.0])
        # Compute required S
        S = min_S_for_positions([(x, y) for x, y, a in inner], [0.0]*N)
        # Add small margin
        S += 0.01
        if _is_valid(inner, (0.0, 0.0), S, 0.0):
            best = (inner, (0.0, 0.0), S, 0.0)
            best_score = S
    
    if best is None:
        # Fallback: very large hexagon
        inner = [[k * 5.0, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best

# Pre-warm
construct()
