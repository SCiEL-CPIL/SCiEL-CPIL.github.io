import numpy as np
import math

def construct():
    # Structural Change: Use a rigid honeycomb lattice structure.
    # The previous attempt failed because individual rotations caused overlaps
    # and incorrect radial distances for the second ring.
    # We place 1 center, 6 first-ring, and 4 second-ring hexagons on a standard
    # hexagonal lattice with identical orientations. This guarantees no overlap.
    
    centers = []
    
    # 1. Center Hexagon
    centers.append((0.0, 0.0))
    
    # 2. First Ring: 6 hexagons at distance sqrt(3)
    r1 = np.sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = r1 * np.cos(angle)
        y = r1 * np.sin(angle)
        centers.append((x, y))
        
    # 3. Second Ring: 4 hexagons at distance 2*sqrt(3)
    # We select angles 60, 120, 240, 300 (dropping 0 and 180) to maintain symmetry
    # and minimize the bounding box.
    r2 = 2 * np.sqrt(3)
    for angle_deg in [60, 120, 240, 300]:
        angle = np.deg2rad(angle_deg)
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        centers.append((x, y))
        
    n_hex = len(centers)
    
    # 4. Orientations
    # In a standard honeycomb packing, all hexagons share the same orientation.
    # We set the angle to 0.0 (vertex at 0 degrees, or flat side at 30, depending on convention).
    # Using a constant angle ensures the lattice gaps are closed without overlap.
    inner = []
    for i in range(n_hex):
        cx, cy = centers[i]
        # All inner hexagons have the same orientation
        angle_deg = 0.0 
        inner.append([cx, cy, angle_deg])
        
    # 5. Calculate Outer Hexagon Side Length
    # The cluster is symmetric. The maximum radial extent is determined by the
    # second-ring hexagons.
    # Distance of center = 2*sqrt(3).
    # Max projection of a unit hexagon (aligned) in the radial direction:
    # If the radial direction aligns with a flat side normal, projection is apothem = sqrt(3)/2.
    # If it aligns with a vertex, projection is 1.
    # In our lattice, the second-ring centers are at 60, 120, etc.
    # The hexagons are oriented with angle 0.
    # Let's check the projection of a hexagon at angle 0 (vertices at 0, 60...) 
    # in the direction of 60 degrees.
    # The vertex at 60 degrees is at distance 1.
    # So the max extent is R_center + 1.
    # Wait, if the hexagon has a vertex at 60, the extent in direction 60 is 1.
    # If it had a flat side normal at 60, the extent would be sqrt(3)/2.
    # Our hexagon has vertices at 0, 60, 120...
    # So in the direction of the second-ring centers (60, 120...), there is a vertex.
    # Thus, max radial distance = 2*sqrt(3) + 1.
    
    R_max = 2 * np.sqrt(3) + 1
    
    # The outer hexagon must contain this radius.
    # The minimal enclosing regular hexagon for a set of points depends on the rotation.
    # If we align the outer hexagon such that its flat sides are perpendicular to the
    # directions of the maximum extent (i.e., flat side normal at 60, 120...),
    # then the apothem H must be at least R_max.
    # Side length S = 2 * H / sqrt(3).
    
    H = R_max
    side = 2 * H / np.sqrt(3)
    
    # Add a tiny margin for floating point safety
    side += 1e-9
    
    # Outer hexagon center (0,0) and rotation.
    # To have flat side normals at 60, 120..., the hexagon vertices should be at 
    # 30, 90, 150... (offset by 30 from the normals).
    # Standard "angle 0" often means vertex at 0.
    # If we want flat side normal at 60, the side is at 60. The vertices adjacent are at 30 and 90.
    # So the hexagon should be rotated by 30 degrees relative to the "vertex at 0" standard.
    # Let's set the outer angle to 30.0 to align flat sides with the 60/120/... axes?
    # Actually, if inner hexagons are at angle 0 (vertex at 0), their vertices are at 0, 60...
    # The second ring is at 60, 120...
    # The vertex of the inner hexagon at 60 points towards the second ring center.
    # So the "bulge" is at 60.
    # The outer hexagon should have a flat side facing 60 to minimize width?
    # If a flat side faces 60, the apothem is in direction 60.
    # So we want the outer hexagon's apothem direction to be 60, 120, etc.
    # For a hexagon with vertices at $\alpha + k \cdot 60$, the apothems are at $\alpha + 30 + k \cdot 60$.
    # We want apothems at 60, 120, ...
    # So $\alpha + 30 = 60 \implies \alpha = 30$.
    # So outer angle = 30.0.
    
    return inner, (0.0, 0.0), side, 30.0

if __name__ == "__main__":
    construct()
