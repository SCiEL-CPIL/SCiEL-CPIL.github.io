"""Packing 11 unit regular hexagons inside a regular hexagon.

Strategy:
  1. Start with a reasonable initial configuration of 11 unit hexagon centers.
  2. Use a local search (simulated annealing / gradient-free optimization) to
     minimize the side length of the smallest enclosing regular hexagon.
  3. The enclosing hexagon is computed by finding, for a given orientation,
     the minimal regular hexagon that contains all inner hexagons.

Key insight:
  A regular hexagon of side L with center (cx, cy) and rotation angle theta
  contains a point p if and only if, for each of the 6 half-planes defining
  the hexagon, the signed distance from p to the boundary is >= 0.

  For a unit hexagon (side 1) at position (x, y) with rotation phi, we need
  all its 6 vertices to be inside the outer hexagon. The worst-case vertex
  determines the constraint.

  The minimal enclosing regular hexagon for a set of points (with fixed
  orientation) can be found by binary search on the side length, or by
  directly computing the maximum over all "supporting" directions.

  Actually, for a regular hexagon with fixed rotation angle theta and center
  (cx, cy), the constraint that a point (px, py) is inside is:
    For k = 0..5: dot((px-cx, py-cy), n_k) <= apothem
  where n_k are the outward normals of the 6 sides, and apothem = L * sqrt(3)/2.

  But the center (cx, cy) is also a variable. So we have:
    For each inner hexagon i and each vertex v of that hexagon:
      For k = 0..5: dot((v - c), n_k(theta)) <= L * sqrt(3)/2

  This is a system of linear inequalities in (L, cx, cy) for fixed theta.
  We can minimize L subject to these constraints using linear programming,
  or we can use a simpler approach: for fixed theta, the optimal (cx, cy, L)
  is determined by the "tightest" constraints.

  A simpler approach: for fixed theta, we can compute the minimal L and center
  by considering that the hexagon must contain all vertices. The minimal
  enclosing regular hexagon with fixed orientation can be found by:
    - Computing the "width" of the point set in each of the 6 normal directions
    - The required apothem is half the max width in any direction
    - The center is the midpoint of the extremes in each direction

  Actually, for a regular hexagon with fixed orientation, the containment
  constraint for a point set S is:
    For each of the 6 normal directions n_k:
      max_{p in S} dot(p, n_k) - min_{p in S} dot(p, n_k) <= 2 * apothem
    where apothem = L * sqrt(3)/2

  So L >= (max_width / sqrt(3)) where max_width is the maximum over all 6
  directions of (max dot - min dot).

  And the center is the midpoint: c_k = (max_dot + min_dot) / 2 for each
  direction, but these must be consistent. For a regular hexagon, the center
  is uniquely determined if the widths in all 6 directions are compatible.

  In practice, the minimal enclosing regular hexagon with fixed orientation
  has side length L = max_k (width_k / sqrt(3)) and the center is chosen so
  that the hexagon is centered on the "center of mass" of the extremes.

  Let's use a more direct approach: for fixed theta, we can use scipy's
  linprog or just a simple grid search / bisection.

  Given time constraints, I'll use a heuristic:
  - For fixed theta, compute the bounding box of all vertices in the 6
    hexagon-normal directions.
  - The required apothem is half the maximum range in any direction.
  - L = 2 * apothem / sqrt(3).
  - The center is the average of the midpoints in each direction (projected
    back to 2D).

  Then we optimize over the positions of the 11 hexagon centers and the
  orientation theta of the outer hexagon.

  To keep it simple and fast, I'll:
  1. Fix the outer hexagon orientation to 0 degrees (or optimize it).
  2. Use a local search on the 22 position variables + 1 angle variable.
  3. The objective is the computed outer side length.
"""

import numpy as np
import time
import math

N = 11

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon with given center, angle, side."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = angle_rad + k * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def _outer_hex_normals(angle_deg):
    """Return 6 outward unit normals for a regular hexagon with given rotation."""
    angle_rad = math.radians(angle_deg)
    normals = []
    for k in range(6):
        # The outward normal of side k is at angle angle_rad + k*pi/3 + pi/6
        # Actually, for a regular hexagon, the normals are at angles
        # angle_rad + k * pi/3 for k=0..5, offset by pi/6 from the vertices
        theta = angle_rad + k * math.pi / 3
        normals.append((math.cos(theta + math.pi/6), math.sin(theta + math.pi/6)))
    return np.array(normals)

def _compute_enclosing_hexagon(vertices, outer_angle_deg):
    """
    Given a set of 2D points (vertices of all inner hexagons) and the outer
    hexagon rotation angle, compute the minimum side length and center of a
    regular hexagon with that rotation that contains all points.
    
    Returns (center_x, center_y, side_length).
    """
    normals = _outer_hex_normals(outer_angle_deg)
    # normals shape: (6, 2)
    # For each normal direction, compute min and max dot product
    # points shape: (M, 2)
    dots = vertices @ normals.T  # (M, 6)
    mins = dots.min(axis=0)  # (6,)
    maxs = dots.max(axis=0)  # (6,)
    ranges = maxs - mins  # (6,)
    
    # The apothem must be at least half the maximum range
    apothem = np.max(ranges) / 2.0
    side_length = 2.0 * apothem / math.sqrt(3)
    
    # The center in each normal direction is the midpoint
    # center_dot_k = (mins[k] + maxs[k]) / 2
    # We need to find (cx, cy) such that for each k:
    #   dot((cx, cy), normal_k) = center_dot_k
    # This is an overdetermined system; use least squares.
    center_dots = (mins + maxs) / 2.0  # (6,)
    # Solve: normals @ (cx, cy).T = center_dots
    # normals is (6, 2), so we solve the 2x2 normal equations
    A = normals.T  # (2, 6)
    # Least squares: (A^T A) c = A^T b
    ATA = A.T @ A  # (2, 2)
    ATb = A.T @ center_dots  # (2,)
    try:
        center = np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        center = np.array([0.0, 0.0])
    
    return center[0], center[1], side_length

def _objective(positions, angles, outer_angle_deg):
    """
    Compute the outer hexagon side length for given inner hexagon positions
    and angles.
    """
    all_verts = []
    for i in range(N):
        cx, cy = positions[i]
        ang = angles[i]
        verts = _hex_vertices(cx, cy, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # (6*N, 2)
    
    cx, cy, L = _compute_enclosing_hexagon(all_verts, outer_angle_deg)
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration: honeycomb-like arrangement
    # 11 hexagons: 1 center, 6 surrounding, 4 more
    # Let's start with a compact cluster
    np.random.seed(42)
    
    # Initial positions: a rough honeycomb pattern
    # Hex grid spacing: for side 1, center-to-center distance is 2*sqrt(3)/2 = sqrt(3) ~ 1.732
    # Actually, for non-overlapping unit hexagons, the minimum center distance is 2*apothem = sqrt(3)
    # if they share a side. Let's use a grid.
    
    # Start with a simple linear arrangement and then optimize
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # Initial: place them in a compact hexagonal pattern
    # Ring 0: 1 at center
    # Ring 1: 6 around
    # Ring 2: 4 more (partial ring)
    d = math.sqrt(3)  # center-to-center distance for edge-sharing
    
    pos_list = [(0, 0)]  # center
    for k in range(6):
        theta = k * math.pi / 3
        pos_list.append((d * math.cos(theta), d * math.sin(theta)))
    # Add 4 more in the next ring
    d2 = 2 * d
    for k in [0, 2, 4, 5]:
        theta = k * math.pi / 3
        pos_list.append((d2 * math.cos(theta), d2 * math.sin(theta)))
    
    # Shuffle to break symmetry
    np.random.shuffle(pos_list)
    for i in range(N):
        positions[i] = pos_list[i]
    
    outer_angle = 0.0
    current_L = _objective(positions, angles, outer_angle)
    best_L = current_L
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_outer_angle = outer_angle
    
    # Local search with simulated annealing
    T = 1.0  # temperature
    cooling = 0.995
    step_size = 0.5
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        
        # Perturb
        new_positions = positions.copy()
        new_angles = angles.copy()
        new_outer_angle = outer_angle
        
        # Randomly perturb a few positions
        n_perturb = np.random.randint(1, 4)
        idx = np.random.choice(N, n_perturb, replace=False)
        for i in idx:
            new_positions[i] += np.random.normal(0, step_size, 2)
        
        # Occasionally perturb angles
        if np.random.random() < 0.3:
            i = np.random.randint(N)
            new_angles[i] += np.random.uniform(-30, 30)
        
        # Occasionally perturb outer angle
        if np.random.random() < 0.2:
            new_outer_angle += np.random.uniform(-10, 10)
        
        new_L = _objective(new_positions, new_angles, new_outer_angle)
        
        # Acceptance criterion
        if new_L < current_L:
            positions = new_positions
            angles = new_angles
            outer_angle = new_outer_angle
            current_L = new_L
            if current_L < best_L:
                best_L = current_L
                best_positions = positions.copy()
                best_angles = angles.copy()
                best_outer_angle = outer_angle
        else:
            # Metropolis acceptance
            if np.random.random() < math.exp(-(new_L - current_L) / T):
                positions = new_positions
                angles = new_angles
                outer_angle = new_outer_angle
                current_L = new_L
        
        T *= cooling
        step_size *= 0.999
    
    # Build result
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
    
    return inner, (0.0, 0.0), best_L, best_outer_angle

# EVOLVE-BLOCK-START
# The construct function is defined above; this block is for compatibility.
# EVOLVE-BLOCK-END
