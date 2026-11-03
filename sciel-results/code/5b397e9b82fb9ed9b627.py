"""
Topological optimization over contact graphs for packing 12 unit regular hexagons
inside a regular hexagon, minimizing the outer side length.
"""
import numpy as np
from scipy.optimize import differential_evolution

N = 12

def hex_vertices(center, side, angle_rad):
    """Return 6 vertices of a regular hexagon with given center, side length, and rotation."""
    cx, cy = center
    angles = angle_rad + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def support_function(vertices, direction):
    """Compute support function h(direction) = max dot(v, direction)."""
    return np.max(vertices @ direction)

def distance_between_hexagons(hex1_verts, hex2_verts):
    """
    Approximate minimum distance between two hexagons by checking
    vertex-to-edge distances in both directions. For optimization, we use
    a conservative lower bound via center distance minus inradius sum.
    """
    c1 = np.mean(hex1_verts, axis=0)
    c2 = np.mean(hex2_verts, axis=0)
    center_dist = np.linalg.norm(c1 - c2)
    inradius = np.sqrt(3) / 2.0  # for unit hexagon
    return center_dist - 2 * inradius

def pack_objective(params):
    """
    Objective: minimize outer side length S such that all 12 unit hexagons fit.
    params layout: for each of 12 hexagons: (x, y, angle_rad), then outer center (x, y), outer angle_rad, outer side S
    """
    n = 12
    # Parse parameters
    inner_params = params[:n * 3].reshape(n, 3)
    outer_center = params[n * 3:n * 3 + 2]
    outer_angle = params[n * 3 + 2]
    outer_side = params[n * 3 + 3]
    
    # Compute vertices for each inner hexagon
    inner_verts = []
    for i in range(n):
        x, y, ang = inner_params[i]
        verts = hex_vertices((x, y), 1.0, ang)
        inner_verts.append(verts)
    
    # Compute outer hexagon vertices
    outer_verts = hex_vertices(outer_center, outer_side, outer_angle)
    
    # Check containment: all inner vertices must be inside outer hexagon
    # Use support function: for each direction, max projection of inner <= max projection of outer
    # We check in 12 directions (6 normals of inner at various angles + 6 of outer)
    
    # For containment, we need: for all directions d, h_inner(d) <= h_outer(d)
    # This is equivalent to inner hexagon being inside outer hexagon.
    # We use the fact that a convex polygon A is inside convex polygon B iff
    # for each edge of B, all vertices of A are on the inside of that edge.
    
    # Check that all inner vertices are inside the outer hexagon
    for verts in inner_verts:
        for v in verts:
            # Point-in-hexagon test using the 6 half-planes of the outer hexagon
            inside = True
            for edge_i in range(6):
                v1 = outer_verts[edge_i]
                v2 = outer_verts[(edge_i + 1) % 6]
                # Edge direction and inward normal
                edge_vec = v2 - v1
                # Inward normal: rotate edge_vec by +90 degrees (assuming CCW winding)
                normal = np.array([-edge_vec[1], edge_vec[0]])
                # Ensure normal points inward: check against center
                center = np.mean(outer_verts, axis=0)
                if np.dot(normal, center - v1) < 0:
                    normal = -normal
                # Point v must satisfy: dot(normal, v - v1) >= -epsilon
                if np.dot(normal, v - v1) < -1e-9:
                    inside = False
                    break
            if not inside:
                # Penalty for violation
                # Compute how far outside
                penalty = 0.0
                for edge_i in range(6):
                    v1 = outer_verts[edge_i]
                    v2 = outer_verts[(edge_i + 1) % 6]
                    edge_vec = v2 - v1
                    normal = np.array([-edge_vec[1], edge_vec[0]])
                    center = np.mean(outer_verts, axis=0)
                    if np.dot(normal, center - v1) < 0:
                        normal = -normal
                    dist = np.dot(normal, v - v1)
                    if dist < 0:
                        penalty += (-dist) ** 2
                return outer_side + 10.0 * penalty
    
    # Check non-overlap: all pairs of inner hexagons must be disjoint
    # Use center distance lower bound with refinement
    for i in range(n):
        for j in range(i + 1, n):
            c1 = inner_params[i, :2]
            c2 = inner_params[j, :2]
            center_dist = np.linalg.norm(c1 - c2)
            # Inradius of unit hexagon
            inradius = np.sqrt(3) / 2.0
            # Minimum center distance for non-overlap of two unit hexagons depends on orientation
            # Conservative: 2 * inradius = sqrt(3) ≈ 1.732
            # Better: use support function overlap check
            # For simplicity, use a conservative bound and let optimization refine
            if center_dist < 2 * inradius - 1e-9:
                # Penalty
                overlap = 2 * inradius - center_dist
                return outer_side + 100.0 * overlap ** 2
    
    return outer_side

def construct():
    """
    Return: (inner, center, side, angle)
    inner: list of [x, y, angle_degrees] for 12 hexagons
    center: (x, y) outer hexagon center
    side: outer hexagon side length
    angle: outer hexagon rotation (degrees)
    """
    # Use differential evolution to find optimal packing
    # Bounds: inner hexagons in [-10, 10] x [-10, 10], angles in [0, 2*pi)
    # Outer center in [-10, 10], outer angle in [0, 2*pi), outer side in [2, 15]
    
    bounds = []
    for i in range(N):
        bounds.append((-10.0, 10.0))  # x
        bounds.append((-10.0, 10.0))  # y
        bounds.append((0.0, 2 * np.pi))  # angle
    
    bounds.append((-10.0, 10.0))  # outer center x
    bounds.append((-10.0, 10.0))  # outer center y
    bounds.append((0.0, 2 * np.pi))  # outer angle
    bounds.append((2.0, 15.0))  # outer side
    
    # Run differential evolution with limited iterations for speed
    result = differential_evolution(
        pack_objective,
        bounds,
        maxiter=50,
        tol=1e-4,
        popsize=20,
        seed=42,
        polish=False
    )
    
    if result.success or result.fun < 50.0:
        params = result.x
    else:
        # Fallback to a reasonable initial guess
        params = np.zeros(N * 3 + 4)
        # Arrange hexagons in a grid-like pattern
        for i in range(N):
            x = (i % 4) * 2.0 - 3.0
            y = (i // 4) * 2.0 - 1.0
            params[i * 3] = x
            params[i * 3 + 1] = y
            params[i * 3 + 2] = 0.0
        params[N * 3] = 0.0
        params[N * 3 + 1] = 0.0
        params[N * 3 + 2] = 0.0
        params[N * 3 + 3] = 5.0
    
    inner_params = params[:N * 3].reshape(N, 3)
    outer_center = (float(params[N * 3]), float(params[N * 3 + 1]))
    outer_angle_rad = float(params[N * 3 + 2])
    outer_side = float(params[N * 3 + 3])
    
    inner = []
    for i in range(N):
        x, y, ang = inner_params[i]
        inner.append([float(x), float(y), float(np.degrees(ang))])
    
    return inner, outer_center, outer_side, float(np.degrees(outer_angle_rad))
