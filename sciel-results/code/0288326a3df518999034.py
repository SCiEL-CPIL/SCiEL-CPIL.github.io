"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Unit regular hexagon vertices at angle 0 (flat-top)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    hex_angles = np.arange(0, 360, 60) * np.pi / 180
    hex_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    # Define a compact cluster of 11 hexagons
    # Layout: 3 rows of staggered hexagons
    # Row 0 (y=0): 4 hexagons at x = -1.5, -0.5, 0.5, 1.5 (spacing sqrt(3))
    # Row 1 (y=sqrt(3)): 3 hexagons at x = -1.0, 0, 1.0 (offset by sqrt(3)/2)
    # Row 2 (y=2*sqrt(3)): 4 hexagons at x = -1.5, -0.5, 0.5, 1.5
    # Wait, let's use the standard honeycomb tiling coordinates.
    # In a honeycomb, adjacent hexagons in the same row are spaced by sqrt(3).
    # Adjacent rows are spaced by 1.5 in y.
    # Let's use a "brick wall" pattern which is denser in terms of bounding box.
    
    # Let's try a 3x4 arrangement but trimmed.
    # Or better, let's use the exact positions for a compact cluster.
    # Consider the cluster formed by a central hexagon and its 6 neighbors (7 total), 
    # plus 4 more to make 11.
    # Let's place them on a grid.
    # Position (i, j) in hex grid:
    # x = (i + j/2) * sqrt(3)
    # y = j * 1.5
    
    # Let's select 11 positions from a 5x5 grid centered at 0.
    # Indices: -2, -1, 0, 1, 2
    # We want to minimize the diameter.
    # A 3x3 block is 9 hexes. Add 2 more.
    # Or a 2x6 block.
    
    # Let's try a 2x6 block rotated.
    # Row 0: 6 hexes
    # Row 1: 5 hexes (staggered)
    # Total 11.
    # Coordinates for Row 0 (y=0): x = -2.5*sqrt(3), -1.5*sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3), 1.5*sqrt(3), 2.5*sqrt(3)
    # Coordinates for Row 1 (y=1.5): x = -2.0*sqrt(3), -1.0*sqrt(3), 0, 1.0*sqrt(3), 2.0*sqrt(3)
    
    sqrt3 = np.sqrt(3)
    centers = []
    # Row 0
    for i in range(6):
        x = (i - 2.5) * sqrt3
        y = 0.0
        centers.append((x, y))
    # Row 1
    for i in range(5):
        x = (i - 2.0) * sqrt3
        y = 1.5
        centers.append((x, y))
        
    centers = np.array(centers)
    
    # Compute the convex hull of the entire cluster (union of all hexagons)
    # Since the hexagons are disjoint and touching, the boundary of the union is a polygon.
    # We can collect all vertices of all hexagons and compute the convex hull of those points.
    # This works because the union of convex polygons is convex if and only if... wait.
    # The union of disjoint convex polygons is NOT necessarily convex.
    # However, for packing inside a convex container, we only care about the convex hull of the union.
    # The minimal enclosing hexagon of the union is the same as the minimal enclosing hexagon of the convex hull of the union.
    # The convex hull of the union of polygons is the convex hull of all vertices of the polygons.
    
    all_points = []
    for cx, cy in centers:
        # Rotate hex_verts (which are at unit distance) - no, they are unit side, so circumradius is 1.
        # Vertices are at distance 1 from center.
        pts = hex_verts + np.array([cx, cy])
        all_points.append(pts)
        
    all_points = np.vstack(all_points)
    
    # Use scipy.spatial.ConvexHull
    from scipy.spatial import ConvexHull
    hull = ConvexHull(all_points)
    hull_points = all_points[hull.vertices]
    
    def required_side_length(global_angle_deg):
        """
        Compute the minimal side length of a regular hexagon (centered at origin, rotated by global_angle_deg)
        that can enclose the hull_points.
        The hexagon is defined by 6 half-planes.
        For a hexagon with side length S and center (0,0) rotated by theta:
        The support function h(theta) gives the maximum projection of the points onto the direction theta.
        For a regular hexagon centered at origin with side length S, the apothem (distance to side) is a = S * sqrt(3) / 2.
        The condition for a point p to be inside the hexagon rotated by phi is:
        max_{k=0..5} (p . n_k) <= a
        where n_k are the unit normals to the sides of the hexagon.
        The normals are at angles phi, phi+60, ..., phi+300.
        
        So for a given global rotation angle phi of the container,
        the required apothem a must be at least max_{p in hull_points, k=0..5} (p . n_{phi + k*60}).
        Then S = 2 * a / sqrt(3).
        """
        phi = np.deg2rad(global_angle_deg)
        # Normals of the hexagon sides
        normal_angles = phi + np.arange(0, 360, 60) * np.pi / 180
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Project hull points onto all 6 normal directions
        # hull_points shape (M, 2), normals shape (6, 2)
        # Projections: hull_points @ normals.T -> (M, 6)
        projections = hull_points @ normals.T
        max_projection = np.max(projections)
        
        apothem = max_projection
        side_length = 2 * apothem / np.sqrt(3)
        
        return side_length

    # Minimize required_side_length over global_angle_deg in [0, 60) due to symmetry
    result = minimize_scalar(required_side_length, bounds=(0, 60), method='bounded', 
                             options={'xatol': 1e-10})
    best_angle = result.x
    best_side = result.fun
    
    # Construct the inner hexagon list
    # The angle in the output is the rotation of the individual hexagons.
    # In the seed, they were all 0.0. We can keep them at 0.0 relative to the cluster,
    # and the "global rotation" is handled by the outer hexagon's rotation.
    # The output format is: inner rows (x, y, angle_degrees)
    # The outer hexagon rotation is the 4th return value.
    # The outer hexagon center is the 3rd return value (actually 2nd tuple is center, 3rd is side, 4th is angle? 
    # Wait: "return inner, (0.0, 0.0), 40.0, 0.0" -> inner, center, side, angle.
    # So:
    # inner: list of [x, y, angle]
    # center: (x, y)
    # side: float
    # angle: float (outer hexagon rotation)
    
    # The cluster centers were defined in a local coordinate system.
    # The outer hexagon is centered at (0,0) in the global frame?
    # The problem says "outer hexagon center (x, y)". We can set it to (0,0) for simplicity,
    # and the inner hexagons are placed relative to that.
    # However, the cluster's centroid might not be (0,0).
    # It is better to center the cluster at (0,0) to exploit the symmetry of the enclosing hexagon.
    
    centroid = np.mean(centers, axis=0)
    centered_centers = centers - centroid
    
    inner_list = []
    for cx, cy in centered_centers:
        inner_list.append([cx, cy, 0.0])
        
    # The outer hexagon angle is best_angle.
    # Note: The calculation of required_side_length assumed the hull points are at their original coordinates.
    # If we shift the hull points by -centroid, the projections change?
    # No, the enclosing hexagon is centered at the origin.
    # If we shift the inner cluster by -centroid, the hull points are shifted.
    # The minimal enclosing hexagon centered at (0,0) for the shifted points is what we want.
    # Let's recompute the hull for the centered points to be precise, or just shift the hull points.
    # Shifting hull points by -centroid is equivalent to shifting the cluster.
    # The function required_side_length used the original hull_points.
    # We need to ensure the minimization was done for the centered configuration.
    
    # Let's redo the calculation with centered points to be safe.
    all_points_centered = all_points - centroid
    hull_centered = ConvexHull(all_points_centered)
    hull_points_centered = all_points_centered[hull_centered.vertices]
    
    def required_side_length_centered(global_angle_deg):
        phi = np.deg2rad(global_angle_deg)
        normal_angles = phi + np.arange(0, 360, 60) * np.pi / 180
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        projections = hull_points_centered @ normals.T
        max_projection = np.max(projections)
        apothem = max_projection
        side_length = 2 * apothem / np.sqrt(3)
        return side_length

    result_c = minimize_scalar(required_side_length_centered, bounds=(0, 60), method='bounded', 
                               options={'xatol': 1e-10})
    best_angle_c = result_c.x
    best_side_c = result_c.fun

    return inner_list, (0.0, 0.0), best_side_c, best_angle_c
# EVOLVE-BLOCK-END
