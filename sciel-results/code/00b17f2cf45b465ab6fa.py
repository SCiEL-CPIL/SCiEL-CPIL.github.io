"""
Pack 12 unit regular hexagons inside a regular hexagon of minimum side length.

Strategy:
- The 12 unit hexagons are arranged in a 3x4 parallelogram grid (3 rows, 4 columns)
  which is the most compact rectangular-ish arrangement for 12 hexagons.
- The grid has a natural hexagonal symmetry; we use this to place them.
- We then optimize:
  1. The outer hexagon's side length S (minimize)
  2. The outer hexagon's rotation angle
  3. Possibly small translations of the inner grid to better fit

A unit regular hexagon with "flat top" orientation has:
- Width (horizontal) = 2 * side = 2
- Height (vertical) = sqrt(3) * side = sqrt(3)

For a 3x4 grid of unit hexagons in a parallelogram arrangement:
- Horizontal spacing between centers in a row: 1.5 (since hex width is 2, and they tile with 1.5 offset)
  Actually, for flat-top hexagons tiling:
  - Column spacing: 1.5 * side = 1.5
  - Row spacing: sqrt(3) * side = sqrt(3)
  - Odd rows offset by 0.75 horizontally

Let me reconsider. For a standard hexagonal tiling of unit hexagons:
- Each hexagon has vertices at distance 1 from center.
- Flat-top: vertices at angles 0, 60, 120, 180, 240, 300 degrees.
  - Width = 2 (from -1 to +1)
  - Height = sqrt(3) (from -sqrt(3)/2 to +sqrt(3)/2)

Tiling:
- Adjacent hexagons in same row: centers separated by 3/2 = 1.5 horizontally
- Adjacent hexagons in adjacent row: centers separated by sqrt(3) vertically, and 3/4 = 0.75 horizontally (staggered)

For a 3-row, 4-column grid:
Row 0 (even): 4 hexagons at x = 0, 1.5, 3.0, 4.5; y = 0
Row 1 (odd):  4 hexagons at x = 0.75, 2.25, 3.75, 5.25; y = sqrt(3)
Row 2 (even): 4 hexagons at x = 0, 1.5, 3.0, 4.5; y = 2*sqrt(3)

Wait, that's 12 hexagons but the bounding box is:
- x from -1 to 5.25+1 = 6.25 → width 7.25
- y from -sqrt(3)/2 to 2*sqrt(3)+sqrt(3)/2 = 2.5*sqrt(3) ≈ 4.33 → height 4.33

Actually let me re-center. The x-range of centers: 0 to 5.25, so with hex radius 1, x from -1 to 6.25.
The y-range of centers: 0 to 2*sqrt(3) ≈ 3.464, so with hex half-height sqrt(3)/2 ≈ 0.866, y from -0.866 to 4.33.

A regular hexagon containing this needs to be large enough. The outer hexagon side length S has:
- Inradius (apothem) = S * sqrt(3)/2
- Circumradius = S

For the outer hexagon to contain the inner arrangement, we need to find the minimum S such that all 12 unit hexagons fit inside.

Rather than analytically solving, let's use numerical optimization:
- Parameters: S (outer side), theta (outer rotation), and possibly a small translation (dx, dy) of the inner cluster.
- Objective: minimize S subject to all inner hexagons being inside the outer hexagon.

To check if a unit hexagon at (cx, cy) with rotation alpha is inside the outer hexagon at (ox, oy) with side S and rotation beta:
- The outer hexagon can be described as the intersection of 6 half-planes.
- A point is inside the outer hexagon if it satisfies all 6 inequalities.
- But we need the entire inner hexagon inside, not just its center.

For a convex polygon inside another convex polygon, the inner polygon is inside iff all its vertices are inside the outer polygon.

So: for each inner hexagon, compute its 6 vertices, and check that all are inside the outer hexagon.

Let me implement this with scipy.optimize.

Time budget: 150s. We'll do a multi-start optimization with a deadline guard.
"""

import numpy as np
import time
import math

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side length, rotation in degrees."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def point_in_hex(px, py, ox, oy, S, beta_deg):
    """Check if point (px,py) is inside regular hexagon centered at (ox,oy) with side S, rotation beta_deg."""
    # Transform to outer hexagon's local frame
    dx = px - ox
    dy = py - oy
    b = math.radians(beta_deg)
    # Rotate point by -b to align outer hexagon with standard orientation
    cx = dx * math.cos(b) + dy * math.sin(b)
    cy = -dx * math.sin(b) + dy * math.cos(b)
    
    # Standard flat-top hexagon with side S has vertices at angles 0,60,...,300
    # It's the intersection of 6 half-planes.
    # For a hexagon with circumradius S (side length S), the 6 edges are at distance S*sqrt(3)/2 from center.
    # The edge normals point at angles 30, 90, 150, 210, 270, 330 degrees (for flat-top).
    # Actually for flat-top (vertex at 0 deg), edges are at 30, 90, 150, 210, 270, 330.
    # Inradius r = S * sqrt(3)/2
    r = S * math.sqrt(3) / 2.0
    for i in range(6):
        theta = math.radians(30 + 60 * i)
        nx = math.cos(theta)
        ny = math.sin(theta)
        # The edge is at distance r from center in direction theta
        # Point must satisfy: nx*cx + ny*cy <= r
        if nx * cx + ny * cy > r + 1e-9:
            return False
    return True

def all_inside(inner_verts_list, ox, oy, S, beta_deg):
    """Check all vertices of all inner hexagons are inside outer hexagon."""
    for verts in inner_verts_list:
        for vx, vy in verts:
            if not point_in_hex(vx, vy, ox, oy, S, beta_deg):
                return False
    return True

def build_inner_grid(dx, dy):
    """Build the 3x4 hexagonal grid of unit hexagons, translated by (dx,dy)."""
    # 3 rows, 4 columns
    # Row 0: y=0, x offsets: 0, 1.5, 3.0, 4.5
    # Row 1: y=sqrt(3), x offsets: 0.75, 2.25, 3.75, 5.25
    # Row 2: y=2*sqrt(3), x offsets: 0, 1.5, 3.0, 4.5
    sqrt3 = math.sqrt(3)
    centers = []
    for row in range(3):
        for col in range(4):
            if row % 2 == 0:
                x = col * 1.5
            else:
                x = col * 1.5 + 0.75
            y = row * sqrt3
            centers.append((x + dx, y + dy))
    return centers

def compute_inner_verts(centers):
    """Compute vertices for all inner hexagons (unit side, 0 rotation)."""
    verts_list = []
    for cx, cy in centers:
        verts = hex_vertices(cx, cy, 1.0, 0.0)
        verts_list.append(verts)
    return verts_list

def objective(params):
    """Objective: return S if feasible (all inside), else large penalty."""
    S, beta_deg, dx, dy = params
    if S < 2.0:  # minimum possible is at least 2
        return 1e6
    centers = build_inner_grid(dx, dy)
    inner_verts = compute_inner_verts(centers)
    if all_inside(inner_verts, 0.0, 0.0, S, beta_deg):
        return S
    else:
        # Penalty: how much S do we need? Use a simple penalty
        return 1e6 + S

def construct():
    t0 = time.time()
    
    # Initial guess for S: the bounding box diagonal / 2 roughly
    # The grid spans x: -1 to 6.25, y: -sqrt(3)/2 to 2.5*sqrt(3)
    # Rough estimate: need S such that circumradius covers the extent
    # Let's start with S = 7.0
    best_S = 8.0
    best_params = [7.0, 0.0, 0.0, 0.0]
    
    # Multi-start optimization
    from scipy.optimize import minimize
    
    starts = []
    # Various starting points for (S, beta, dx, dy)
    for S0 in [6.5, 7.0, 7.5, 8.0]:
        for beta0 in [0, 30, 15, -15, 45, -30]:
            for dx0 in [-0.5, 0.0, 0.5]:
                for dy0 in [-0.5, 0.0, 0.5]:
                    starts.append([S0, float(beta0), dx0, dy0])
    
    # Shuffle starts for diversity
    np.random.seed(42)
    np.random.shuffle(starts)
    
    for i, x0 in enumerate(starts):
        if time.time() - t0 > DEADLINE - 5:
            break
        
        try:
            res = minimize(
                objective, x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
            )
            if res.fun < best_S and res.fun < 1e5:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Extract best
    S_best = best_params[0]
    beta_best = best_params[1]
    dx_best = best_params[2]
    dy_best = best_params[3]
    
    # Build final inner hexagons
    centers = build_inner_grid(dx_best, dy_best)
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
    
    # Verify feasibility
    inner_verts = compute_inner_verts(centers)
    if not all_inside(inner_verts, 0.0, 0.0, S_best, beta_best):
        # Fallback: use a larger S
        S_best = 8.0
        beta_best = 0.0
        dx_best = 0.0
        dy_best = 0.0
        centers = build_inner_grid(0.0, 0.0)
        inner = [[cx, cy, 0.0] for cx, cy in centers]
    
    return inner, (0.0, 0.0), S_best, beta_best

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
