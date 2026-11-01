import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define axial coordinates for a compact cluster of 11 hexagons
    # Core 7:
    # (0,0)
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Additional 4:
    # (2,0), (2,-1), (1,-2), (0,-2)
    
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2)
    ]
    
    # Convert to Cartesian coordinates
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in axial_coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # Calculate minimal enclosing regular hexagon side length L
    # A regular hexagon with side L has apothem a = L * sqrt(3) / 2.
    # The hexagon is the intersection of 3 strips of width 2a.
    # The strips are aligned with angles 0, 60, 120 degrees.
    # For a hexagon centered at origin, the constraints for a point (x,y) to be inside
    # (considering the point is the center of an inner hexagon which has its own size)
    # are more complex.
    # Actually, the inner hexagons are regular hexagons of side 1.
    # We need the container to cover the union of these hexagons.
    # This is equivalent to finding the minimal L such that the Minkowski sum of the container
    # and the "negative" of the inner hexagon covers the centers?
    # Simpler: The distance from the origin to the boundary of the container in direction theta
    # must be >= distance from origin to the farthest point of the inner hexagon in direction theta.
    
    # The farthest point of a unit hexagon from its center in direction theta is the support function h(theta).
    # For a regular hexagon of side 1, the support function is:
    # h(theta) = cos(theta - k*pi/3) / cos(pi/6) for theta in [k*pi/3 - pi/6, k*pi/3 + pi/6]
    # Actually, simpler: The max projection of a unit hexagon onto a unit vector u is:
    # If u is aligned with a vertex normal, it's 1.
    # If u is aligned with a side normal, it's sqrt(3)/2.
    # Generally, max_proj(u) = max( |dot(u, v_i)| for vertices v_i )
    # Vertices of unit hexagon: (1,0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1,0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2)
    
    vertices = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    
    # The container hexagon has side L.
    # Its support function in direction u is L * h_container(u).
    # For a regular hexagon of side L, h_container(u) is proportional to L.
    # Specifically, if we scale the unit hexagon by L, the support function scales by L.
    # So we need: L * h_unit(u) >= |dot(c_i, u)| + h_unit(u) for all u and all centers c_i?
    # No.
    # The condition is: The container (centered at 0) must contain the inner hexagon (centered at c_i).
    # This is equivalent to: The container centered at 0 must contain the Minkowski sum of the inner hexagon (centered at c_i) and the negative of the inner hexagon (centered at 0)?
    # No.
    # Let C be the container set (hexagon side L).
    # Let H_i be the inner hexagon set (center c_i, side 1).
    # We need H_i subset C.
    # This is equivalent to: c_i + H_0 subset C, where H_0 is unit hexagon at 0.
    # This is equivalent to: c_i subset C - H_0 (Minkowski difference).
    # C - H_0 is a hexagon (since C and H_0 are homothetic).
    # C is a hexagon of side L. H_0 is a hexagon of side 1.
    # C - H_0 is a hexagon of side L-1.
    # So the condition is: All centers c_i must lie inside a regular hexagon of side L-1 centered at 0.
    
    # So we just need to find the minimal side length L' of a regular hexagon (aligned with the lattice)
    # that contains all centers c_i.
    # Then L = L' + 1.
    
    # To find L' for the centers:
    # The hexagon is defined by 3 pairs of parallel lines.
    # The width of the hexagon in direction u is 2 * a' where a' is the apothem.
    # a' = L' * sqrt(3) / 2.
    # We need max_i |dot(c_i, u)| <= a' for the 3 axis directions u1, u2, u3.
    # u1 = (1, 0)
    # u2 = (cos(60), sin(60)) = (0.5, sqrt3/2)
    # u3 = (cos(120), sin(120)) = (-0.5, sqrt3/2)
    
    u1 = np.array([1.0, 0.0])
    u2 = np.array([0.5, sqrt3/2])
    u3 = np.array([-0.5, sqrt3/2])
    
    max_proj_1 = 0
    max_proj_2 = 0
    max_proj_3 = 0
    
    for c in centers:
        c_arr = np.array(c)
        p1 = abs(np.dot(c_arr, u1))
        p2 = abs(np.dot(c_arr, u2))
        p3 = abs(np.dot(c_arr, u3))
        if p1 > max_proj_1: max_proj_1 = p1
        if p2 > max_proj_2: max_proj_2 = p2
        if p3 > max_proj_3: max_proj_3 = p3
    
    a_prime = max(max_proj_1, max_proj_2, max_proj_3)
    L_prime = 2 * a_prime / sqrt3
    L = L_prime + 1.0
    
    # The inner hexagons are rotated 0 degrees relative to the container.
    inner = [[x, y, 0.0] for x, y in centers]
    
    return inner, (0.0, 0.0), L, 0.0
# EVOLVE-BLOCK-END
