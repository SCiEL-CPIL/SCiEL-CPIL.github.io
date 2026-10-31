import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Precompute unit hex vertices (pointy-top: vertices at 0, 60, 120, 180, 240, 300)
    # Using standard math orientation where 0 degrees is along x-axis
    base_verts = np.array([(math.cos(math.radians(60*i)), math.sin(math.radians(60*i))) for i in range(6)])
    
    # Precompute rotation matrices for the 6 discrete orientations (0, 60, 120, 180, 240, 300)
    # Since hexagons have 6-fold symmetry, only 6 unique orientations exist.
    # We can fix the angle to one of these to reduce search space and ensure validity.
    # Actually, for packing, rotating a hexagon by 60 degrees is equivalent to the same shape.
    # So we can fix all angles to 0 without loss of generality for the shape, 
    # BUT the problem allows rotation. However, for a regular hexagon, 
    # any rotation is just a relabeling of vertices if we consider the set of points.
    # Wait, the orientation matters for how it fits in the container if the container is rotated.
    # But since we can rotate the container too, relative orientation is what matters.
    # For simplicity and speed, we will fix inner hexagon angles to 0. 
    # The container can be rotated.
    # Is there a benefit to rotating inner hexagons? 
    # If all inner hexagons are parallel, it's a standard packing.
    # If we mix orientations, we might pack tighter? 
    # For regular hexagons, mixing orientations often doesn't help for small N 
    # because the "flat" sides align best. 
    # Let's stick to 0 degrees for inner hexagons to simplify the physics.
    
    # Adjacency matrix for the "flower" + 4 pattern or 3-4-3 pattern.
    # Let's use a generic dense graph initially or a specific one.
    # A good initial guess is the 3-4-3 pattern (10 hexes) + 1 center? No, 3-4-3 is 10.
    # 11 hexes: 1 center, 6 around, 4 next.
    # Let's define indices for the flower pattern (1 center, 6 neighbors).
    # Indices: 0 is center. 1-6 are neighbors.
    # We need 4 more. Let's place them at positions 7, 8, 9, 10.
    
    # We will use a force-directed approach.
    # Nodes: 11 hexagons.
    # Variables: x, y for each hexagon. (22 variables)
    # Objective: Minimize the size of the enclosing hexagon + Penalty for overlaps.
    
    def get_verts(x, y, angle_deg=0.0):
        # Returns 6x2 array of vertices
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        R = np.array([[ca, -sa], [sa, ca]])
        return base_verts @ R.T + np.array([x, y])

    def compute_enclosing_hex_size(positions, container_angle_deg=0.0):
        # positions: list of (x, y)
        # Returns the minimum side length S of a hexagon rotated by container_angle_deg
        # that contains all positions.
        # The hexagon is centered at (0,0).
        # Vertices of a unit hexagon (side 1) are at distance 1 from center? No, side length 1.
        # Distance from center to vertex (circumradius) is 1.
        # Distance from center to edge (apothem) is sqrt(3)/2.
        # If we scale by S, circumradius is S, apothem is S * sqrt(3)/2.
        
        # To find if a point (px, py) is inside a hexagon of side S centered at origin, rotated by theta:
        # The hexagon is the intersection of 6 half-planes.
        # The normals to the sides are at angles theta + 30, 90, 150, 210, 270, 330.
        # For a side with normal n, the constraint is dot(p, n) <= S * sqrt(3)/2.
        
        a = math.radians(container_angle_deg)
        max_S = 0.0
        apothem_factor = math.sqrt(3) / 2.0
        
        # Normals to the 6 sides
        for i in range(6):
            theta = a + math.radians(30 + 60*i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for px, py in positions:
                proj = px*nx + py*ny
                if proj > max_proj:
                    max_proj = proj
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def check_overlap(i, j, pos_i, pos_j):
        # Check if hexagon i and j overlap.
        # Since we fix angles to 0, we can use Minkowski sum.
        # Two unit hexagons overlap if the distance between centers is less than the 
        # radius of the Minkowski difference (which is a hexagon of side 2 * sqrt(3)/2 * 2? No).
        # Actually, for two identical regular polygons with parallel sides, 
        # they are disjoint if the distance between centers is >= 2 * apothem = sqrt(3).
        # This is only true if they are aligned. 
        # If one is rotated, the minimum distance to touch is smaller.
        # But we fixed angles to 0. So they are aligned.
        # Distance between centers >= sqrt(3) ensures no overlap.
        dx = pos_i[0] - pos_j[0]
        dy = pos_i[1] - pos_j[1]
        return (dx*dx + dy*dy) < 3.0 - 1e-9 # sqrt(3)^2 = 3

    def objective(params):
        # params: [x0, y0, x1, y1, ..., x10, y10]
        positions = [(params[2*i], params[2*i+1]) for i in range(N)]
        
        # Calculate enclosing hex size
        S = compute_enclosing_hex_size(positions, 0.0)
        
        # Calculate overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if check_overlap(i, j, positions[i], positions[j]):
                    # Add penalty proportional to how deep they overlap
                    # Approximate overlap depth
                    dist = math.sqrt((positions[i][0]-positions[j][0])**2 + (positions[i][1]-positions[j][1])**2)
                    overlap = math.sqrt(3) - dist
                    penalty += overlap * 10.0 # Weight penalty heavily
        
        return S + penalty

    # Initial guess: 3-4-3 pattern centered at 0,0
    # dist = sqrt(3)
    d = math.sqrt(3)
    # Row 0 (y = -d*sqrt(3)/2): 3 hexes. x = -d, 0, d
    # Row 1 (y = 0): 4 hexes. x = -1.5d, -0.5d, 0.5d, 1.5d
    # Row 2 (y = d*sqrt(3)/2): 3 hexes. x = -d, 0, d
    # Total 10. Add one at (0,0)? No, (0,0) is in row 1? 
    # Let's check coordinates for 3-4-3:
    # y = -h/2: x = -d, 0, d
    # y = 0: x = -1.5d, -0.5d, 0.5d, 1.5d
    # y = h/2: x = -d, 0, d
    # This is 10 hexes.
    # Where to put the 11th? 
    # Maybe a 2-3-3-3 pattern? Or 1-6-4 flower?
    # Let's use the flower pattern for initialization.
    # Center (0,0)
    # 6 neighbors at distance d, angles 0, 60, ...
    # 4 more at distance 2d? Or next ring.
    
    init_pos = []
    # Center
    init_pos.append((0.0, 0.0))
    # 6 neighbors
    for k in range(6):
        angle = math.radians(60*k)
        init_pos.append((d * math.cos(angle), d * math.sin(angle)))
    # 4 more: place them at the "corners" of the hexagon formed by the 6 neighbors.
    # The 6 neighbors form a hexagon of side d. 
    # The vertices of this "center-hexagon" are at distance d from center.
    # The "corners" of the packing are further out.
    # Let's place them at distance 2*d, at angles 30, 90, 150, 210?
    # Or just perturb the 6 neighbors and add 4 more randomly close.
    # Better: Use a known compact 11-hex cluster.
    # The "star" or "flower" with 4 extra on the outside.
    # Let's just use the 3-4-3 + 1 center idea, but shift it.
    # Actually, the 3-4-3 is 10. If we add one, where?
    # Maybe 2-4-4-1?
    # Let's just randomize the 4 extra ones around the perimeter.
    
    for k in range(4):
        angle = math.radians(60*k + 30) # Between the neighbors
        # Place them at distance 2*d
        init_pos.append((2*d * math.cos(angle), 2*d * math.sin(angle)))
    
    # Center the initial configuration
    cx = sum(p[0] for p in init_pos) / N
    cy = sum(p[1] for p in init_pos) / N
    init_pos = [(p[0]-cx, p[1]-cy) for p in init_pos]
    
    x0 = []
    for p in init_pos:
        x0.append(p[0])
        x0.append(p[1])
        
    # Run BFGS optimization
    try:
        res = minimize(objective, x0, method='BFGS', options={'maxiter': 200, 'gtol': 1e-6})
        
        if res.fun < best_S:
            final_pos = [(res.x[2*i], res.x[2*i+1]) for i in range(N)]
            # Re-center
            cx = sum(p[0] for p in final_pos) / N
            cy = sum(p[1] for p in final_pos) / N
            final_pos = [(p[0]-cx, p[1]-cy) for p in final_pos]
            
            # Check validity
            S = compute_enclosing_hex_size(final_pos, 0.0)
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if check_overlap(i, j, final_pos[i], final_pos[j]):
                        valid = False
                        break
                if not valid: break
            
            if valid:
                best_S = S
                best_inner = [[p[0], p[1], 0.0] for p in final_pos]
                best_center = (0.0, 0.0)
                best_angle = 0.0
                best_valid = True
                
    except Exception:
        pass

    # If BFGS failed or didn't find a good solution, try a few random restarts with BFGS
    if not best_valid or best_S > 5.0:
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            
            # Random initial positions in a small area
            rng = random.Random(restart)
            rand_pos = []
            for _ in range(N):
                x = rng.uniform(-3, 3)
                y = rng.uniform(-3, 3)
                rand_pos.append((x, y))
            
            cx = sum(p[0] for p in rand_pos) / N
            cy = sum(p[1] for p in rand_pos) / N
            rand_pos = [(p[0]-cx, p[1]-cy) for p in rand_pos]
            
            x0 = []
            for p in rand_pos:
                x0.append(p[0])
                x0.append(p[1])
                
            try:
                res = minimize(objective, x0, method='BFGS', options={'maxiter': 100, 'gtol': 1e-5})
                if res.fun < best_S:
                    final_pos = [(res.x[2*i], res.x[2*i+1]) for i in range(N)]
                    cx = sum(p[0] for p in final_pos) / N
                    cy = sum(p[1] for p in final_pos) / N
                    final_pos = [(p[0]-cx, p[1]-cy) for p in final_pos]
                    
                    S = compute_enclosing_hex_size(final_pos, 0.0)
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            if check_overlap(i, j, final_pos[i], final_pos[j]):
                                valid = False
                                break
                        if not valid: break
                    
                    if valid:
                        best_S = S
                        best_inner = [[p[0], p[1], 0.0] for p in final_pos]
                        best_center = (0.0, 0.0)
                        best_angle = 0.0
                        best_valid = True
                        
            except Exception:
                pass

    if best_valid and best_inner is not None:
        return best_inner, best_center, best_S, best_angle

    # Fallback: Simple lattice 3-4-3-1 (approx)
    d = math.sqrt(3)
    inner = []
    # 3-4-3
    ys = [-d*math.sqrt(3)/2, 0, d*math.sqrt(3)/2]
    counts = [3, 4, 3]
    for i, count in enumerate(counts):
        y = ys[i]
        start_x = -d * (count - 1) / 2.0
        for k in range(count):
            x = start_x + k * d
            inner.append([x, y, 0.0])
    # Add 11th at center (0,0) if not already there? 
    # In 3-4-3, (0,0) is in the middle row.
    # So we have 10. Add one at (0, d*sqrt(3))?
    inner.append([0, d*math.sqrt(3), 0.0])
    
    # Center it
    cx = sum(p[0] for p in inner) / N
    cy = sum(p[1] for p in inner) / N
    inner = [[p[0]-cx, p[1]-cy, p[2]] for p in inner]
    
    S = compute_enclosing_hex_size([(p[0], p[1]) for p in inner], 0.0)
    return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-END
