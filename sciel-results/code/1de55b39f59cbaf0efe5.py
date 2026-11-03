"""Packing 11 unit hexagons into a minimal regular hexagon via geometric optimization."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg=0.0):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg.
    Flat-top orientation: vertices at 0,60,120,180,240,300 degrees."""
    a = np.deg2rad(angle_deg)
    verts = np.zeros((6, 2))
    for i in range(6):
        theta = a + np.deg2rad(60 * i)
        verts[i] = [cx + np.cos(theta), cy + np.sin(theta)]
    return verts

def _outer_side_length(points, angle_deg):
    """Given points and outer hexagon rotation angle_deg (pointy-top support directions),
    compute the minimal side length of a regular hexagon (centered at origin) with that
    rotation that encloses all points. Returns (side_length, center_offset)."""
    a = np.deg2rad(angle_deg)
    # Support directions for pointy-top hexagon: normals at 90,150,210,270,330,30
    # Equivalently the 6 outward normals of a regular hexagon.
    # A regular hexagon with 'pointy top' has vertices at 90,150,210,270,330,30 and
    # outward normals at 0,60,120,180,240,300? Let's use consistent definition:
    # Outer hexagon: set of points p such that for all 6 directions d_k, d_k . p <= R,
    # where R is the apothem (inradius). Side length s = 2*R / sqrt(3).
    # Directions d_k are the outward normals of the 6 faces.
    # For a regular hexagon rotated by angle_deg, the 6 face normals are at
    # angle_deg + 0, 60, 120, 180, 240, 300 (degrees).
    dirs = np.zeros((6, 2))
    for k in range(6):
        theta = a + np.deg2rad(60 * k)
        dirs[k] = [np.cos(theta), np.sin(theta)]
    # For each direction, max projection
    R = 0.0
    for d in dirs:
        proj = points @ d
        R = max(R, proj.max())
    # Also need to consider negative? No, since we center at origin and hexagon is
    # symmetric, we need |d.p| <= R for all 6 dirs. But if we use 6 dirs at 60 deg spacing
    # starting from angle a, they cover all 360 degrees with 6 directions, and the hexagon
    # is the intersection of half-planes d.p <= R AND -d.p <= R? No, that would be a dodecagon.
    # Actually a regular hexagon has 6 faces, so 6 half-planes. The 6 outward normals
    # are at angles a, a+60, ..., a+300. The hexagon is {p : d_k . p <= R for k=0..5}.
    # But this only constrains in the direction of d_k, not -d_k. Since d_{k+3} = -d_k,
    # the constraint d_{k+3} . p <= R is -d_k . p <= R, i.e., d_k . p >= -R.
    # So we need |d_k . p| <= R for k=0,1,2 (or all 6).
    R = 0.0
    for k in range(3):
        d = dirs[k]
        proj = points @ d
        R = max(R, proj.max(), -proj.min())
    # Side length of regular hexagon with inradius R: s = 2*R / sqrt(3)
    side = 2.0 * R / np.sqrt(3.0)
    return side

def construct():
    # Cluster: center + 6 neighbors + 4 additional in a compact arrangement
    # Hex lattice spacing for side-1 hexagons (pointy-top) is sqrt(3) horizontally
    # Let's use pointy-top orientation for inner hexagons.
    # For pointy-top unit hexagons, centers of adjacent hexagons are distance sqrt(3) apart.
    # Arrangement: center at (0,0), 6 neighbors at distance sqrt(3) in 6 directions,
    # plus 4 more to make 11.
    
    # Let me use a known compact arrangement for 11 hexagons:
    # Row-based or flower pattern.
    # Center + ring of 6 + 4 more placed in the "gaps" or extending the ring.
    
    # Actually, let me just place them in a compact hexagonal cluster:
    # A hexagonal number pattern: 1 + 6 + 12 = 19 for full, but we only need 11.
    # 1 (center) + 6 (first ring) = 7, need 4 more.
    
    # Place 4 additional hexagons adjacent to the first ring, filling in compactly.
    # First ring centers (pointy-top, spacing sqrt(3)):
    s3 = np.sqrt(3.0)
    centers = [(0.0, 0.0)]
    for i in range(6):
        theta = np.deg2rad(60 * i + 30)  # pointy-top neighbor directions
        centers.append((s3 * np.cos(theta), s3 * np.sin(theta)))
    # 4 more: place them at positions that extend the cluster compactly
    # Add hexagons at distance 2*s3 in 4 of the 6 directions, or in the "second ring" gaps.
    # Let me add them adjacent to ring-1 hexagons, in the outward directions.
    # For compactness, add at positions that are neighbors of ring-1 hexagons.
    # Ring-1 hexagon at angle 30 deg: (s3*cos30, s3*sin30) = (1.5, s3/2)
    # Its outward neighbor would be at (s3*cos30 + s3*cos30, s3*sin30 + s3*sin30) = (3.0, s3)
    # But that might not be compact. Let me try a different arrangement.
    
    # Alternative: use a 3-4-4 or similar row arrangement.
    # Let me just use a simple compact blob and let the optimizer handle it.
    
    # For now, use center + 6 ring + 4 placed at second-ring positions (compact)
    # Second ring positions that are adjacent to ring-1:
    # The 6 ring-1 hexagons are at angles 30,90,150,210,270,330 (pointy-top neighbors)
    # Add 4 more at: (0, 2*s3*sin(30)) ... let me just compute valid positions.
    
    # Let me use a known good arrangement: place 11 hexagons in a "hexagonal" shape
    # with rows of 1,3,5,2 (pointy-top) or similar.
    
    # Actually, let me just place them in a compact cluster and compute the enclosing hexagon.
    # For 11 hexagons, a good arrangement is a central hexagon surrounded by 6,
    # plus 4 more attached to the outside.
    
    # Let me define the 11 centers explicitly for a compact arrangement:
    # Using pointy-top hexagons with center-to-center distance sqrt(3) for adjacency.
    dx = s3 * np.cos(np.deg2rad(30))  # = s3 * sqrt(3)/2 = 1.5
    dy = s3 * np.sin(np.deg2rad(30))  # = s3 * 0.5 = sqrt(3)/2
    
    centers = [
        (0.0, 0.0),                    # center
        (dx, dy), (0.0, 2*dy), (-dx, dy),  # top 3
        (-dx, -dy), (0.0, -2*dy), (dx, -dy),  # bottom 3
        # 4 more, placed to extend compactly
        (2*dx, 0.0),                   # right
        (0.0, 3*dy),                   # top
        (0.0, -3*dy),                  # bottom
        (-2*dx, 0.0),                  # left
    ]
    
    # All inner hexagons are pointy-top (angle 0)
    angles = [0.0] * 11
    
    # Collect all vertices
    all_points = []
    for (cx, cy), ang in zip(centers, angles):
        verts = _hex_vertices(cx, cy, ang)
        all_points.append(verts)
    all_points = np.vstack(all_points)
    
    # Optimize the outer hexagon rotation angle to minimize side length
    # The outer hexagon center is at the centroid of the inner hexagons for symmetry
    centroid = np.mean(all_points, axis=0)
    # Translate points so centroid is at origin
    points_centered = all_points - centroid
    
    def objective(angle_deg):
        return _outer_side_length(points_centered, angle_deg)
    
    # Search over angle from 0 to 60 degrees (periodicity)
    result = minimize_scalar(objective, bounds=(0, 60), method='bounded',
                              options={'xatol': 1e-6})
    best_angle = result.x
    best_side = result.fun
    
    # The outer hexagon center is at the centroid
    outer_center = (float(centroid[0]), float(centroid[1]))
    outer_angle = float(best_angle)
    outer_side = float(best_side)
    
    # Build inner list
    inner = [[float(cx), float(cy), float(ang)] for (cx, cy), ang in zip(centers, angles)]
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
