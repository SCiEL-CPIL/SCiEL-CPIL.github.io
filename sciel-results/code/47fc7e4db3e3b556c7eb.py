"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import cKDTree
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # For N=11, we need to pack 11 unit regular hexagons into a regular hexagon.
    # The side length of a unit hexagon is 1.
    # A unit hexagon has width (distance between parallel sides) = sqrt(3) and height (distance between opposite vertices) = 2.
    # If we arrange them in a honeycomb pattern, we can fit them densely.
    
    # Let's try to find a good configuration.
    # One known packing for small N:
    # For N=1, side length = 1 (just one hexagon, but the outer hexagon must contain it.
    # Wait, the inner hexagons are unit side length. The outer hexagon contains them.
    # If we place 1 hexagon, the outer hexagon side length must be at least 1 (if aligned) or less?
    # Actually, if the inner hexagon is side 1, its "radius" (center to vertex) is 1.
    # The outer hexagon must contain it. If aligned, outer side length >= 1.
    
    # For N=11, let's try a central hexagon surrounded by 6, and then 4 more?
    # A honeycomb cluster of 19 hexagons has side length 3 in hex-grid units.
    # But we only have 11.
    
    # Let's use a grid-based local search or simulated annealing to minimize the outer side length.
    # However, implementing a full packing solver within 120s for N=11 is complex.
    # Let's try a heuristic: place hexagons in a hexagonal lattice pattern.
    
    # Hexagonal lattice coordinates:
    # Each hexagon center is at (3/2 * i, sqrt(3)/2 * (2j + i)) for integer i, j?
    # Actually, for flat-top hexagons (pointy top), the distance between centers in a row is 1.5 * side = 1.5.
    # Vertical distance is sqrt(3) * side = sqrt(3).
    
    # Let's try to place 11 hexagons in a compact cluster.
    # Center at (0,0).
    # Neighbors at distance sqrt(3) (if touching).
    # Wait, two unit hexagons touch if their centers are distance sqrt(3) apart (for flat-top, side-by-side).
    # Or 2*sqrt(3)/2? No.
    # Distance between centers of two touching unit hexagons (flat-top) is 1.5 + 1.5 = 3? No.
    # The width of a hexagon is sqrt(3). So if they are side by side, centers are sqrt(3) apart?
    # Let's check: flat-top hexagon, side 1.
    # Vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2).
    # Width (x-direction) = 2. Height (y-direction) = sqrt(3).
    # If placed side-by-side horizontally, they touch at x=0.5 and x=-0.5 of the next?
    # No, the rightmost point is (1,0). The leftmost is (-1,0).
    # So if two hexagons are centered at (0,0) and (3,0), they don't touch.
    # If centered at (0,0) and (1.5, sqrt(3)/2), do they touch?
    # Distance = sqrt(1.5^2 + (sqrt(3)/2)^2) = sqrt(2.25 + 0.75) = sqrt(3).
    # Yes, distance sqrt(3) is the center-to-center distance for touching hexagons in a honeycomb.
    
    # So, let's generate a honeycomb cluster of 11 hexagons.
    # Center: (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3) from center, angles 0, 60, 120, 180, 240, 300 degrees.
    # That's 7 hexagons.
    # Ring 2: We need 4 more. Place them at some positions in Ring 2.
    # Ring 2 positions are at distance 2*sqrt(3) or sqrt(3)*sqrt(3)=3?
    # Actually, the next layer in a hex grid is at distance 2*sqrt(3) from center? No.
    # The centers form a triangular lattice with spacing sqrt(3).
    # So Ring 2 centers are at distance 2*sqrt(3) from the center? No, the distance from center to Ring 2 is 2 * (distance to Ring 1) = 2*sqrt(3)?
    # Wait, the lattice spacing is sqrt(3). So Ring 1 is at distance sqrt(3). Ring 2 is at distance 2*sqrt(3).
    # There are 12 positions in Ring 2. We pick 4.
    
    # Let's define the positions.
    centers = [(0.0, 0.0)]
    # Ring 1
    for k in range(6):
        angle = k * np.pi / 3
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        centers.append((x, y))
    # Ring 2 (pick 4)
    # Angles for Ring 2: 30, 90, 150, 210, 270, 330, etc.
    # Actually, the positions are at angles (k+0.5)*60 degrees?
    # Let's just use the lattice vectors.
    # v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 3/2)
    # Ring 2 positions: 2*v1, 2*v2, 2*(v1-v2), etc.
    # Let's pick 4 symmetric positions.
    ring2_angles = [np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6]  # 30, 90, 150, 210
    for angle in ring2_angles:
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        centers.append((x, y))
    
    # Now we have 11 centers.
    # We need to find the minimum outer hexagon side length that contains all these hexagons.
    # The outer hexagon is also regular, side length S, center (cx, cy), rotation theta.
    # A point (x,y) is inside the outer hexagon if...
    # For a regular hexagon with side S, center (0,0), rotation 0 (flat-top):
    # The vertices are at (S,0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S,0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2).
    # The inequalities for a point (x,y) to be inside are:
    # |x| <= S
    # |y| <= S*sqrt(3)/2
    # |sqrt(3)*x + y| <= S*sqrt(3)  ? No.
    # Let's use the support function. The hexagon is the intersection of 6 half-planes.
    # For a flat-top hexagon with side S:
    # 1. x <= S
    # 2. x >= -S
    # 3. sqrt(3)*x + y <= S*sqrt(3)  (wait, at (S/2, S*sqrt(3)/2), sqrt(3)*S/2 + S*sqrt(3)/2 = S*sqrt(3). Correct.)
    # 4. sqrt(3)*x - y >= -S*sqrt(3)  (at (S/2, -S*sqrt(3)/2), sqrt(3)*S/2 - (-S*sqrt(3)/2) = S*sqrt(3). So sqrt(3)*x - y <= S*sqrt(3)? No, y is negative.
    # Let's check the normal vectors.
    # The 6 normals are at angles 0, 60, 120, 180, 240, 300.
    # n_k = (cos(k*pi/3), sin(k*pi/3))
    # The distance from center to the side is S*sqrt(3)/2.
    # So the constraint is: n_k . (p - c) <= S*sqrt(3)/2 for all k.
    # Wait, for a hexagon with side S, the apothem (distance from center to side) is S*sqrt(3)/2.
    # Yes.
    
    # So, for a given set of inner hexagons (each with center p_i and rotation 0),
    # we need to find the minimum S such that all points of all inner hexagons are inside the outer hexagon.
    # The extreme points of the inner hexagons are their vertices.
    # For a unit hexagon at (x_i, y_i) with rotation 0, the vertices are:
    # (x_i + cos(a), y_i + sin(a)) for a in [0, 60, 120, 180, 240, 300] degrees.
    
    # We can compute the required S for a given outer center (cx, cy) and rotation theta.
    # For each inner hexagon vertex v, and for each of the 6 normals of the outer hexagon (rotated by theta),
    # we compute the projection. The maximum projection over all vertices and all normals gives the required apothem.
    # S = apothem / (sqrt(3)/2) = 2 * apothem / sqrt(3).
    
    # Let's try to optimize (cx, cy, theta) to minimize S.
    # We can use a simple grid search or local optimization.
    
    # First, let's compute the vertices of all inner hexagons.
    vertices = []
    for (cx_i, cy_i) in centers:
        for k in range(6):
            angle = k * np.pi / 3
            vx = cx_i + np.cos(angle)
            vy = cy_i + np.sin(angle)
            vertices.append((vx, vy))
    vertices = np.array(vertices)
    
    # Function to compute the required side length S for a given outer center and rotation.
    def compute_S(cx, cy, theta):
        # Normals of the outer hexagon
        angles = [theta + k * np.pi / 3 for k in range(6)]
        normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
        # Project all vertices onto the normals
        # relative vertices
        rel_vertices = vertices - np.array([cx, cy])
        # Projections: (642, 6)
        projections = rel_vertices @ normals.T
        # Max projection for each normal
        max_projections = np.max(projections, axis=0)
        # The apothem must be at least the max of these
        apothem = np.max(max_projections)
        # S = apothem / (sqrt(3)/2)
        return 2 * apothem / np.sqrt(3)
    
    # Let's do a coarse grid search over theta in [0, pi/3) and cx, cy in a small range.
    best_S = float('inf')
    best_params = (0.0, 0.0, 0.0)
    
    # Coarse search
    thetas = np.linspace(0, np.pi/3, 12, endpoint=False)
    # The cluster is roughly centered at (0,0). Let's search cx, cy in [-1, 1].
    cxs = np.linspace(-1, 1, 11)
    cys = np.linspace(-1, 1, 11)
    
    for theta in thetas:
        for cx in cxs:
            for cy in cys:
                if time.time() > deadline - 10:
                    break
                S = compute_S(cx, cy, theta)
                if S < best_S:
                    best_S = S
                    best_params = (cx, cy, theta)
        if time.time() > deadline - 10:
            break
    if time.time() > deadline - 10:
        break
    
    # Local refinement using Nelder-Mead or similar
    # We'll use a simple gradient-free local search.
    def objective(params):
        cx, cy, theta = params
        return compute_S(cx, cy, theta)
    
    # Use scipy.optimize.minimize
    from scipy.optimize import minimize
    
    try:
        res = minimize(objective, best_params, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 100})
        if res.fun < best_S:
            best_S = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Construct the inner list
    inner = []
    for (cx_i, cy_i) in centers:
        inner.append([cx_i, cy_i, 0.0])
    
    cx, cy, theta = best_params
    return inner, (cx, cy), best_S, np.degrees(theta)
# EVOLVE-BLOCK-END
