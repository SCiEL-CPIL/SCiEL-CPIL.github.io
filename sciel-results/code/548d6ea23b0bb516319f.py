"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Initial Seed: "Flower" pattern (1 center + 6 ring + 4 extra)
    # This is a compact starting point that is valid (if scaled down) 
    # but we want to find the minimal *unscaled* container.
    # Actually, we will place them in a valid configuration relative to each other
    # and then find the minimal enclosing hexagon.
    
    # Generate initial positions
    # Center at (0,0)
    # Ring 1 at distance 2 (touching)
    # Ring 2 partial
    
    positions = []
    angles = [0.0] * N
    
    # Center
    positions.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons
    for i in range(6):
        angle = math.pi / 3 * i
        x = 2.0 * math.cos(angle)
        y = 2.0 * math.sin(angle)
        positions.append((x, y))
        
    # Ring 2: Need 4 more to reach 11.
    # Place them in the gaps or next layer.
    # Gaps in Ring 1 are at angles pi/6 + i*pi/3, distance 2*sqrt(3) ~ 3.46
    # But those are far. Let's just place them in a rough hexagonal lattice expansion.
    # Hex lattice vectors: u=(2,0), v=(1, sqrt(3))
    # Let's just scatter the remaining 4 in the second shell roughly.
    # To be safe, let's start with a valid, slightly loose packing.
    # We will optimize positions to minimize the bounding box.
    
    # A better seed: use a known dense packing structure.
    # For N=11, a "hexagonal number" 1+6+6 is 13. 
    # So 11 is 1+6+4.
    # Let's place the 4 in the second ring.
    # Second ring coordinates (axial):
    # (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1)
    # Cartesian: x = 1.5*q, y = sqrt(3)/2*q + sqrt(3)*r
    # Let's map a few.
    extra_coords = [
        (2, 0),   # q=2, r=0 -> x=3, y=0
        (1, 1),   # q=1, r=1 -> x=1.5, y=sqrt(3)
        (0, 2),   # q=0, r=2 -> x=0, y=2*sqrt(3)
        (-1, 1)   # q=-1, r=1 -> x=-1.5, y=sqrt(3)
    ]
    for q, r in extra_coords:
        x = 1.5 * q
        y = (math.sqrt(3) / 2) * q + math.sqrt(3) * r
        positions.append((x, y))
        
    # Ensure we have exactly N
    while len(positions) < N:
        # Fallback: random scatter if seed logic fails
        positions.append((np.random.uniform(-5, 5), np.random.uniform(-5, 5)))
    positions = positions[:N]
    
    # Convert to numpy array for vectorization
    pos = np.array(positions, dtype=float)
    
    # Optimization Loop
    # We want to minimize the size of the minimal enclosing regular hexagon.
    # This is equivalent to minimizing the maximum distance from the center 
    # to any vertex of any inner hexagon, projected onto the 6 normal directions 
    # of the outer hexagon, assuming the outer hexagon is centered at the 
    # centroid of the inner set (or we can optimize the center too).
    
    # Simpler approach: 
    # 1. Minimize the spread (variance) of positions to keep them compact.
    # 2. Ensure non-overlap.
    # 3. Calculate the minimal enclosing hexagon size.
    
    # We will use a simple gradient-free or gradient-based local search.
    # Given time constraints, a simple iterative repulsion + centering is robust.
    
    best_side = float('inf')
    best_pos = pos.copy()
    best_center = (0.0, 0.0)
    
    # Helper: Calculate minimal enclosing hexagon side length
    def calc_enclosing_hex_side(points, center):
        # points: (N, 2)
        # center: (2,)
        # Outer hexagon is regular, centered at 'center'.
        # We can rotate the outer hexagon? The task says "outer ... rotation".
        # So we should find the minimal side length over all rotations too.
        # However, usually the optimal orientation aligns with the packing symmetry.
        # Let's try a few discrete rotations or optimize it.
        
        # The condition for a point p to be inside a regular hexagon of side s, 
        # center c, rotation theta is:
        # For each of 6 edges, distance <= s * sqrt(3)/2 (inradius)
        # The inradius R_in = s * sqrt(3) / 2
        # So s = 2 * R_in / sqrt(3)
        
        # We need to find min s such that all points are inside.
        # This is equivalent to finding the minimal R_in such that all points 
        # are within the hexagon.
        
        # For a fixed rotation theta, the max required R_in is:
        # max over all points p and all 6 normals n_i(theta) of:
        # dot(p - c, n_i)
        # (Assuming n_i are unit outward normals)
        
        # Let's optimize theta as well.
        # Discrete search for theta might be enough if the packing is symmetric,
        # but let's do a small continuous search.
        
        min_side = float('inf')
        
        # Try a few rotations
        # The symmetry of the hexagon is 60 deg.
        # Let's sample 12 rotations (30 deg steps) or optimize.
        
        # Actually, we can just compute the support function.
        # For a convex polygon P, the minimal enclosing hexagon size is related
        # to the maximum width in the 6 directions.
        
        # Let's just iterate theta from 0 to 60 deg in small steps.
        # Or better: The optimal theta aligns one of the hexagon edges/vertices 
        # with the "widest" part of the cluster.
        
        # For speed, let's assume the optimal rotation is one of the 6 symmetric 
        # ones relative to the principal axes of the point cloud, or just 0.
        # Given the "flower" seed, 0 is likely good.
        
        # Let's check theta = 0.
        # Normals for theta=0 (flat top/bottom or pointy top?):
        # Standard hexagon vertices at angles 0, 60, ...
        # Normals at 30, 90, ...
        
        # Let's define normals for a hexagon with side s.
        # Inradius r = s * sqrt(3) / 2.
        # Distance from center to edge is r.
        # So we need max(dot(p-c, n_i)) <= r for all i.
        
        # n_i for regular hexagon (unit normals):
        # If vertices are at 0, 60, 120..., normals are at 30, 90, 150...
        base_normals = []
        for i in range(6):
            angle = math.pi / 6 + i * math.pi / 3
            base_normals.append((math.cos(angle), math.sin(angle)))
        base_normals = np.array(base_normals)
        
        # Relative positions
        rel = points - center
        
        # For a given rotation theta, normals are rotated by theta.
        # We want to minimize max_i max_p dot(rel_p, n_i(theta))
        
        # Let's try optimizing theta.
        # The function f(theta) = max_p max_i dot(rel_p, R(theta) n_i) is convex-ish.
        
        # Simple grid search for theta
        best_r_in = float('inf')
        for theta in np.linspace(0, math.pi/3, 20):
            c, s = math.cos(theta), math.sin(theta)
            rot = np.array([[c, -s], [s, c]])
            normals = base_normals @ rot.T
            
            # dot products: (N, 6)
            dots = rel @ normals.T
            max_proj = np.max(dots)
            if max_proj < best_r_in:
                best_r_in = max_proj
                
        if best_r_in < 0:
            return 0.0 # Should not happen if center is inside
            
        side = 2 * best_r_in / math.sqrt(3)
        return side
        
    # Optimization Loop
    # We will use a simple "relaxation" method.
    # 1. Compute repulsive forces between all pairs of hexagons.
    # 2. Compute attractive force towards the center (to keep them compact).
    # 3. Update positions.
    
    # Since we need to minimize the *enclosing* size, not just pairwise distances,
    # the "centering" force should be towards the centroid of the current set,
    # and we should also allow the "center" of the enclosing hexagon to shift?
    # The task allows the outer center to be anywhere.
    # So effectively, we are minimizing the diameter/spread of the set.
    
    # Let's just minimize the variance of the positions (keep them tight)
    # while enforcing non-overlap.
    
    # Non-overlap constraint:
    # Distance between centers >= 2 * inradius? No.
    # For two regular hexagons of side 1, the minimum distance between centers
    # to be disjoint is 2 * inradius = sqrt(3) if they are aligned?
    # No, if they are rotated, it can be different.
    # Actually, the minimum distance between centers for *any* rotation is 
    # the sum of inradii? No, that's for parallel edges.
    # The minimum distance is 2 * inradius = sqrt(3) ~ 1.732.
    # Wait, if they touch at a vertex, the distance can be smaller?
    # No, the inradius is the distance from center to edge.
    # If two convex sets touch, the distance between centers is at least the sum of their "radii" in that direction.
    # The minimum "radius" of a hexagon is the inradius.
    # So center distance >= sqrt(3) is a sufficient condition for non-overlap 
    # regardless of rotation?
    # Let's verify.
    # If two hexagons are rotated 30 deg relative to each other, the "pointy" 
    # vertex of one can touch the "flat" edge of the other.
    # Distance from center to vertex is 1. Distance from center to edge is sqrt(3)/2.
    # If vertex touches edge, distance = 1 + sqrt(3)/2 ~ 1.866.
    # If edge touches edge (parallel), distance = 2 * sqrt(3)/2 = sqrt(3) ~ 1.732.
    # So the minimum distance is sqrt(3).
    # So we can enforce dist >= sqrt(3) as a hard constraint.
    
    min_dist = math.sqrt(3)
    
    # Simple iterative solver
    step_size = 0.1
    damping = 0.9
    pos = best_pos.copy()
    
    # Calculate initial side to have a baseline
    center = np.mean(pos, axis=0)
    current_side = calc_enclosing_hex_side(pos, center)
    best_side = current_side
    best_pos = pos.copy()
    best_center = center
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # 1. Repulsion
        forces = np.zeros_like(pos)
        for i in range(N):
            for j in range(i+1, N):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < min_dist:
                    # Push apart
                    if dist < 1e-6:
                        d = np.random.normal(0, 0.1, 2)
                        dist = np.linalg.norm(d)
                    f = (min_dist - dist) * 0.5
                    forces[i] += (d / dist) * f
                    forces[j] -= (d / dist) * f
        
        # 2. Centering / Compactness
        # Pull towards the mean
        centroid = np.mean(pos, axis=0)
        centering_force = (centroid - pos) * 0.05
        
        # 3. Update
        pos += step_size * (forces + centering_force)
        
        # Check validity (non-overlap)
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(pos[i] - pos[j]) < min_dist - 1e-9:
                    valid = False
                    break
            if not valid: break
            
        if valid:
            new_center = np.mean(pos, axis=0)
            new_side = calc_enclosing_hex_side(pos, new_center)
            if new_side < best_side:
                best_side = new_side
                best_pos = pos.copy()
                best_center = new_center
        
        # Decay step size
        step_size *= 0.999
        if step_size < 0.001:
            break
            
    # Final return
    # Ensure best_pos is a list of lists as required
    inner = [[x, y, 0.0] for x, y in best_pos]
    return inner, (best_center[0], best_center[1]), best_side, 0.0

# EVOLVE-BLOCK-END
