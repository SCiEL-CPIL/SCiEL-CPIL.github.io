import math
import time
import random
import numpy as np

N = 11

def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Precompute base hex vertices (side 1, angle 0)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    base_verts = []
    for i in range(6):
        theta = math.radians(60 * i)
        base_verts.append((math.cos(theta), math.sin(theta)))
    
    base_verts = np.array(base_verts)

    def transform_verts(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        # Rotation matrix
        R = np.array([[ca, -sa], [sa, ca]])
        verts = base_verts @ R.T * S + np.array([cx, cy])
        return verts

    def compute_outer_hex(inner, ocx, ocy, oangle):
        """
        Computes the minimum side length S of an outer hexagon centered at (ocx, ocy)
        with rotation oangle that contains all inner hexagons.
        """
        inner_verts = []
        for (ix, iy, iangle) in inner:
            inner_verts.append(transform_verts(ix, iy, 1.0, iangle))
        
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        
        # The outer hexagon is defined by 6 half-planes.
        # Normal vectors for the half-planes are at angles 30, 90, 150, 210, 270, 330 relative to the hexagon's orientation.
        # If the hexagon is rotated by oangle, the normals are at oangle + 30 + 60*i.
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            
            max_proj = 0.0
            for iv in inner_verts:
                # Project all vertices onto the normal
                projs = (iv[:, 0] - ocx) * nx + (iv[:, 1] - ocy) * ny
                proj = np.max(projs)
                if proj > max_proj:
                    max_proj = proj
            
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def all_hexagons_fit(inner, ocx, ocy, S, oangle):
        """
        Checks if all inner hexagons are inside the outer hexagon and disjoint from each other.
        """
        outer_verts = transform_verts(ocx, ocy, S, oangle)
        inner_verts = []
        for (ix, iy, iangle) in inner:
            inner_verts.append(transform_verts(ix, iy, 1.0, iangle))
        
        # Check containment
        # For a regular polygon, checking if all vertices of inner are inside outer is sufficient
        # provided inner is convex and outer is convex.
        # We use a point-in-polygon check for vertices.
        # Since we are dealing with convex polygons, we can also check half-planes.
        # Half-plane check is faster and more robust for convex shapes.
        
        a = math.radians(oangle)
        apothem = S * math.sqrt(3) / 2
        
        for iv in inner_verts:
            for i in range(6):
                theta = a + math.radians(30 + 60 * i)
                nx, ny = math.cos(theta), math.sin(theta)
                # Distance from center to the line
                # The line is defined by n . (x - c) = apothem
                # Actually, the support function.
                # Max projection of inner vertices on normal must be <= apothem + center_offset?
                # Let's stick to the definition:
                # Outer hexagon vertices are at distance S from center.
                # Apothem is S * sqrt(3)/2.
                # The boundary is at distance apothem from center along the normal.
                
                # Project inner vertices relative to outer center
                projs = (iv[:, 0] - ocx) * nx + (iv[:, 1] - ocy) * ny
                if np.max(projs) > apothem + 1e-9:
                    return False

        # Check disjointness
        # Two convex polygons are disjoint if their vertices are not inside each other and edges don't intersect.
        # For efficiency in the simulation, we might approximate or use a faster check.
        # But for the final validation, we need to be sure.
        # Given N=11, O(N^2) checks are fine.
        
        def point_in_convex_polygon(p, poly):
            # poly is 6x2 array
            # Cross product check
            for i in range(6):
                p1 = poly[i]
                p2 = poly[(i+1)%6]
                v1 = p1 - p
                v2 = p2 - p1
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                if cross < -1e-9: # Assuming CCW ordering
                    return False
            return True

        # Ensure base_verts are CCW
        # (1,0), (0.5, 0.866), (-0.5, 0.866)...
        # Cross of (1,0)->(0.5, 0.866) and (0.5, 0.866)->(-0.5, 0.866)
        # v1 = (-0.5, 0.866), v2 = (-1, 0). Cross = 0.866 > 0. CCW is correct.

        for i in range(len(inner_verts)):
            vi = inner_verts[i]
            for j in range(i+1, len(inner_verts)):
                vj = inner_verts[j]
                # Check if any vertex of i is inside j
                for v in vi:
                    if point_in_convex_polygon(v, vj):
                        return False
                # Check if any vertex of j is inside i
                for v in vj:
                    if point_in_convex_polygon(v, vi):
                        return False
                # Check edge intersections
                # For convex polygons, if no vertices are inside, edges might still intersect?
                # No, for convex polygons, if no vertex is inside the other, they are disjoint.
                # Wait, is that true?
                # If A and B are convex, and no vertex of A is in B and no vertex of B is in A,
                # can they intersect?
                # Yes, they can cross like an X? No, convex sets.
                # If they intersect, the intersection is a convex set.
                # If the intersection contains no vertices of A or B, it must be formed by edge crossings.
                # Can two convex polygons intersect only at edge crossings without vertices inside?
                # Yes. Example: two squares rotated 45 deg relative to each other, centered at same point.
                # Vertices of one are outside the other, but edges cross.
                # So we MUST check edge intersections.
                
                # Edge intersection check
                def segments_intersect(p1, p2, p3, p4):
                    def cross(o, a, b):
                        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
                    d1 = cross(p3, p4, p1)
                    d2 = cross(p3, p4, p2)
                    d3 = cross(p1, p2, p3)
                    d4 = cross(p1, p2, p4)
                    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                        return True
                    # Check collinearity / touching?
                    # For "disjoint", touching is usually allowed (interiors disjoint).
                    # So strict intersection is what we want to avoid.
                    # If they just touch, d's are 0.
                    # The above condition handles proper intersection.
                    # If they touch at a point, it's not "overlapping" in the sense of area.
                    # So we return False for touching.
                    return False

                for k in range(6):
                    e1_p1 = vi[k]
                    e1_p2 = vi[(k+1)%6]
                    for m in range(6):
                        e2_p1 = vj[m]
                        e2_p2 = vj[(m+1)%6]
                        if segments_intersect(e1_p1, e1_p2, e2_p1, e2_p2):
                            return False
        return True

    def simulate_packing(initial_positions, outer_S, outer_angle, seed, max_iters=5000):
        """
        Force-directed simulation to pack N hexagons inside an outer hexagon.
        Returns (inner_positions, success).
        """
        rng = random.Random(seed)
        
        # State: positions (N, 2), angles (N,)
        # We fix angles to 0 for simplicity in the simulation?
        # The task allows rotation. But for a first pass, let's see if translation alone works.
        # Actually, the inspiration says "inner hexagons as particles".
        # Let's allow rotation in the simulation too?
        # That increases dimensionality. Let's start with fixed angles (0) to see if it's enough.
        # If not, we can add rotation.
        # Given the symmetry of hexagons, aligning them is often optimal for dense packing.
        
        pos = np.array([p[:2] for p in initial_positions], dtype=float)
        angles = np.zeros(N)
        
        # Forces
        # 1. Repulsion between inner hexagons
        # 2. Attraction to center / Confinement by outer hexagon
        
        # Potential:
        # U_overlap = sum over pairs of max(0, overlap_depth)^2
        # U_boundary = sum over hexes of max(0, distance_to_boundary)^2
        
        # We need a way to compute "overlap_depth" and "distance_to_boundary" efficiently.
        # This is hard for arbitrary convex polygons.
        # Approximation:
        # Treat hexagons as circles?
        # Hexagon circumradius = 1. Apothem = sqrt(3)/2.
        # If we treat them as circles of radius R_eff, we might get a good packing.
        # R_eff = 1.0 (circumradius) is too big, they will overlap in corners.
        # R_eff = 0.866 (apothem) is too small, they can overlap in sides.
        # Let's use a circle of radius 0.95?
        # Or better: use the actual geometry for the force calculation.
        
        # Alternative: Use a soft constraint solver.
        # Minimize E = sum_{i<j} max(0, dist(i,j) - 2*apothem)^2 + sum_i max(0, dist_to_boundary(i))^2
        # This is still hard to compute dist(i,j) for hexagons.
        
        # Let's try a simpler heuristic:
        # Use a "repulsive force" based on center distance.
        # If center distance < D_min, push apart.
        # D_min = 2 * apothem = sqrt(3) approx 1.732.
        # This ensures they don't overlap if they are aligned.
        # If they are rotated, D_min might need to be larger.
        # Let's assume alignment (angle 0) for the simulation, then optimize angles later if needed.
        
        D_min = math.sqrt(3)
        
        # Boundary constraint:
        # Keep centers inside a slightly smaller hexagon.
        # The hexagon must be inside the outer hexagon.
        # So the center must be at least 'apothem' away from the boundary.
        # So the center must be inside a hexagon of side S - 2*apothem?
        # No.
        # If the hexagon has side 1, its apothem is a = sqrt(3)/2.
        # The outer hexagon has side S, apothem A = S*sqrt(3)/2.
        # The center of the inner hexagon must be at least 'a' away from the boundary of the outer hexagon.
        # So the center must be inside a hexagon with apothem A - a.
        # Side S_inner = 2 * (A - a) / sqrt(3) = 2 * (S*sqrt(3)/2 - sqrt(3)/2) / sqrt(3) = S - 1.
        
        # So, constraint: Centers must be inside a regular hexagon of side S - 1.
        # This is a standard convex constraint.
        
        S_effective = outer_S - 1.0
        if S_effective <= 0:
            return None, False
            
        # Simulation Loop
        T = 1.0 # Temperature
        cooling = 0.999
        step_size = 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            forces = np.zeros_like(pos)
            
            # Pairwise repulsion
            for i in range(N):
                for j in range(i+1, N):
                    diff = pos[i] - pos[j]
                    dist = np.linalg.norm(diff)
                    if dist < D_min:
                        # Force proportional to overlap
                        overlap = D_min - dist
                        if dist > 1e-6:
                            dir_vec = diff / dist
                        else:
                            dir_vec = np.array([rng.uniform(-1,1), rng.uniform(-1,1)])
                            dir_vec /= np.linalg.norm(dir_vec)
                        force = overlap * 10.0 # Spring constant
                        forces[i] += force * dir_vec
                        forces[j] -= force * dir_vec
                        
            # Boundary confinement
            # Center must be inside hexagon of side S_effective.
            # Hexagon vertices: (S_eff, 0), (S_eff/2, S_eff*sqrt(3)/2), ...
            # Half-planes:
            # n_k . x <= A_eff, where A_eff = S_eff * sqrt(3)/2
            # n_k are unit normals at angles 30, 90, ...
            
            A_eff = S_effective * math.sqrt(3) / 2
            for i in range(N):
                for k in range(6):
                    theta = math.radians(30 + 60 * k)
                    nx, ny = math.cos(theta), math.sin(theta)
                    # Distance from boundary
                    # n . x = A_eff is the boundary.
                    # If n . x > A_eff, it's outside.
                    val = pos[i][0]*nx + pos[i][1]*ny
                    if val > A_eff:
                        # Push back
                        force = (val - A_eff) * 10.0
                        forces[i][0] -= force * nx
                        forces[i][1] -= force * ny
                        
            # Update positions
            # Simple gradient descent with random perturbation
            # pos += forces * step_size + noise
            noise = rng.gauss(0, T, size=pos.shape)
            pos += forces * step_size + noise
            
            T *= cooling
            
        # After simulation, check validity
        inner = []
        for i in range(N):
            inner.append([pos[i][0], pos[i][1], 0.0])
            
        # Recompute S to be safe
        S_new = compute_outer_hex(inner, 0.0, 0.0, outer_angle)
        if all_hexagons_fit(inner, 0.0, 0.0, S_new, outer_angle):
            return inner, True
        return None, False

    # Strategy:
    # 1. Start with a rough estimate of S.
    # 2. Run simulation to find a valid configuration for that S.
    # 3. If valid, try to reduce S slightly and re-simulate.
    # 4. Or, run simulation with a fixed S, and see what S is actually needed.
    
    # Let's try a binary search on S?
    # Or just start with a large S and shrink.
    
    # Initial guess for S:
    # Area of 11 hexagons = 11 * (3*sqrt(3)/2) * 1^2 = 16.5 * sqrt(3) approx 28.58.
    # Area of outer hexagon = (3*sqrt(3)/2) * S^2.
    # S^2 = 28.58 / (2.598) approx 11. S approx 3.3.
    # Let's start with S = 4.0.
    
    current_S = 4.0
    found_valid = False
    
    # We will iterate, reducing S.
    # We need good initial positions for the simulation.
    # Random positions inside the allowed region.
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Try to find a valid config for current_S
        # Run simulation multiple times with different seeds
        success = False
        best_inner_for_S = None
        
        for seed in range(10):
            if time.time() > deadline:
                break
            
            # Generate random initial positions inside the hexagon of side current_S - 1
            S_eff = current_S - 1.0
            if S_eff <= 0:
                continue
                
            A_eff = S_eff * math.sqrt(3) / 2
            init_pos = []
            for _ in range(N):
                # Sample uniformly inside hexagon
                # Rejection sampling
                while True:
                    x = rng.uniform(-S_eff, S_eff)
                    y = rng.uniform(-S_eff, S_eff)
                    # Check if inside hexagon
                    inside = True
                    for k in range(6):
                        theta = math.radians(30 + 60 * k)
                        nx, ny = math.cos(theta), math.sin(theta)
                        if x*nx + y*ny > A_eff:
                            inside = False
                            break
                    if inside:
                        init_pos.append([x, y, 0.0])
                        break
            
            inner, valid = simulate_packing(init_pos, current_S, 0.0, seed + attempt*100)
            if valid:
                success = True
                # Check if this is better (smaller S is the goal, but we are fixing S)
                # Actually, simulate_packing returns a config that fits in current_S.
                # We can compute the *actual* minimal S for this config.
                S_actual = compute_outer_hex(inner, 0.0, 0.0, 0.0)
                if best_inner_for_S is None or S_actual < best_inner_for_S[1]:
                    best_inner_for_S = (inner, S_actual)
        
        if success:
            inner, S_actual = best_inner_for_S
            if S_actual < best_S:
                best_S = S_actual
                best_inner = inner
                best_center = (0.0, 0.0)
                best_angle = 0.0
                best_valid = True
                
            # Shrink S for next iteration
            # How much to shrink?
            # If we found a valid config, we can try S_actual - 0.05
            # Or just reduce current_S by a factor.
            current_S = S_actual - 0.05
            if current_S < 3.0: # Lower bound
                break
        else:
            # Failed to find valid config for current_S
            # Increase S?
            # If we are stuck, maybe start over with a larger S.
            current_S += 0.1
            if current_S > 5.0:
                break

    # Fallback if no valid solution found
    if not best_valid:
        # Linear arrangement
        sp = math.sqrt(3)
        fallback_inner = []
        for k in range(N):
            fallback_inner.append([(k - N // 2) * sp, 0.0, 0.0])
        S = compute_outer_hex(fallback_inner, 0.0, 0.0, 0.0)
        # Ensure it fits
        # This linear arrangement definitely fits in a long thin hexagon, but we want minimal S.
        # For N=11, S will be large.
        # Just return it.
        return fallback_inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
