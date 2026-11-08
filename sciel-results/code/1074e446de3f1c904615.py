import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Helper: Generate vertices of a regular hexagon
    def hex_verts(cx, cy, s, rot_deg):
        rot_rad = np.deg2rad(rot_deg)
        angles = rot_rad + np.deg2rad(np.arange(6) * 60)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def min_dist_polygons(v1, v2):
        """
        Exact minimum distance between two convex polygons.
        Returns 0 if intersecting (including touching), else min distance.
        """
        min_d = float('inf')
        # Check vertex of v1 to edges of v2
        for i in range(6):
            p1 = v1[i]
            for j in range(6):
                p2 = v2[j]
                p3 = v2[(j+1)%6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        # Check vertex of v2 to edges of v1
        for i in range(6):
            p1 = v2[i]
            for j in range(6):
                p2 = v1[j]
                p3 = v1[(j+1)%6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        
        # Check for intersection. 
        # If min_d is very small, they might be touching or intersecting.
        # For robustness in physics sim, we treat min_d <= 0 as overlap? 
        # Actually min_dist returns >= 0.
        # If they intersect, min distance between boundaries is 0.
        # We need to know if they are strictly inside each other.
        # A simple check: if any vertex of v1 is inside v2 or vice versa.
        # But for force calculation, we can just use the min_d.
        # If min_d < eps, they are close/overlapping.
        # To get a "penetration depth" for force, we can use a heuristic:
        # If they intersect, we need a stronger repulsion.
        # Let's assume if min_d < 1e-9, they are intersecting/touching.
        # We will return -min_d if we want to indicate overlap? No, standard is distance.
        # For the physics engine, we will calculate force based on min_d.
        # If min_d == 0, force is max.
        return min_d

    def point_in_hex(p, cx, cy, s, rot_deg):
        """Check if point p is inside hexagon centered at (cx,cy) with side s and rot."""
        # Transform p to local coords of hexagon
        p_local = np.array([p[0] - cx, p[1] - cy])
        rot_rad = np.deg2rad(rot_deg)
        rot_mat = np.array([
            [math.cos(-rot_rad), -math.sin(-rot_rad)],
            [math.sin(-rot_rad), math.cos(-rot_rad)]
        ])
        p_local = rot_mat @ p_local
        
        # Hexagon is intersection of 6 half-planes.
        # Normals at 30, 90, 150, 210, 270, 330 degrees.
        # Distance from center to side is r = s * sqrt(3) / 2
        r = s * math.sqrt(3) / 2
        normals = np.deg2rad(30 + np.arange(6)*60)
        n_vecs = np.column_stack([np.cos(normals), np.sin(normals)])
        
        dists = np.dot(p_local, n_vecs.T)
        # If any dist > r, it's outside
        return np.all(dists <= r + 1e-9)

    def compute_forces(params, outer_s):
        """
        params: [x1, y1, ..., x12, y12]
        Returns forces for each hexagon center.
        """
        xs = params[0:12]
        ys = params[12:24]
        forces = np.zeros((12, 2))
        
        inner_verts = []
        for i in range(12):
            v = hex_verts(xs[i], ys[i], 1.0, 0.0)
            inner_verts.append(v)
            
        # 1. Containment Force
        # Outer hexagon: center (0,0), side outer_s, angle 0.
        r = outer_s * math.sqrt(3) / 2.0
        normals = np.deg2rad(30 + np.arange(6)*60)
        n_vecs = np.column_stack([np.cos(normals), np.sin(normals)])
        
        for i in range(12):
            # Check vertices of inner hexagon i
            for v in inner_verts[i]:
                dists = np.dot(v, n_vecs.T)
                # If dist > r, it's outside.
                # Force should push it back in.
                # The normal direction is n_vecs[k].
                # We add force in direction -n_vecs[k] proportional to penetration.
                for k in range(6):
                    if dists[k] > r:
                        penetration = dists[k] - r
                        # Force is opposite to normal
                        force_dir = -n_vecs[k]
                        forces[i] += force_dir * penetration * 10.0 # Stiffness

        # 2. Repulsion Force (Overlap)
        # We want to push hexagons apart if they are too close.
        # We use min_dist_polygons.
        # If d < threshold, apply force.
        # To get direction, we need the closest points. 
        # For simplicity and speed, we can use the vector between centers 
        # if they are close, but that fails for side-by-side hexagons.
        # Better: use the gradient of the distance function.
        # Approximation: If min_dist is small, the direction is roughly 
        # from the closest point on j to closest point on i.
        # Let's implement a slightly more expensive but accurate direction finder.
        
        for i in range(12):
            for j in range(i+1, 12):
                d = min_dist_polygons(inner_verts[i], inner_verts[j])
                # If they are intersecting, d is 0. 
                # We need to know if they are actually overlapping in area.
                # Check if center of i is inside j or vice versa? 
                # Or just check if d < eps.
                # If d < 0.01, consider them "in contact" or overlapping.
                # To be safe, let's check intersection properly.
                # If any vertex of i is inside j, they overlap.
                overlap = False
                for v in inner_verts[i]:
                    if point_in_hex(v, xs[j], ys[j], 1.0, 0.0):
                        overlap = True
                        break
                if not overlap:
                    for v in inner_verts[j]:
                        if point_in_hex(v, xs[i], ys[i], 1.0, 0.0):
                            overlap = True
                            break
                
                if overlap or d < 0.05:
                    # Calculate repulsion force
                    # Direction: from center of j to center of i (push i away from j)
                    # This is a heuristic but works well for convex shapes if they are close.
                    # For more accuracy, we'd need the normal at the contact point.
                    # Let's stick to center-to-center for now, but scaled by d.
                    # If d is small, force is large.
                    # If they are side-by-side, center-to-center is perpendicular to contact normal?
                    # No, for hexagons, center-to-center is often close to normal.
                    # Let's try to find the normal direction using the closest vertices.
                    
                    # Find closest vertex from i to j and j to i
                    # This is expensive. 
                    # Alternative: Use a "soft" repulsion based on center distance 
                    # combined with the min_dist check.
                    # If min_dist < 0.05, they are close.
                    # Force direction: normalize(ci - cj)
                    vec = np.array([xs[i] - xs[j], ys[i] - ys[j]])
                    dist_centers = np.linalg.norm(vec)
                    if dist_centers < 1e-6:
                        # Random direction
                        vec = np.array([np.random.randn(), np.random.randn()])
                        dist_centers = np.linalg.norm(vec)
                    normal = vec / dist_centers
                    
                    # Force magnitude: 
                    # If overlapping, strong push.
                    # If just close (d < 0.05), weaker push to prevent jitter.
                    if overlap:
                        # Strong repulsion
                        # Magnitude proportional to "penetration"
                        # We don't have exact penetration, so use a fixed strong force
                        # or scale by (1 - d) if d was negative? 
                        # Let's use a fixed strong force if overlapping.
                        f_mag = 5.0
                    else:
                        # d is between 0 and 0.05
                        f_mag = 2.0 * (0.05 - d) / 0.05
                    
                    forces[i] += normal * f_mag
                    forces[j] -= normal * f_mag
        
        return forces

    def simulate(outer_s, num_steps=2000, dt=0.01, damping=0.9):
        """
        Run Langevin dynamics to find a stable configuration.
        """
        # Initialize positions
        # Use a grid-like initial guess to help convergence
        xs = np.zeros(12)
        ys = np.zeros(12)
        # 4x3 grid
        for i in range(12):
            x = (i % 4) * 2.0 - 3.0
            y = (i // 4) * 2.0 - 2.0
            xs[i] = x + np.random.normal(0, 0.1)
            ys[i] = y + np.random.normal(0, 0.1)
            
        params = np.concatenate([xs, ys])
        velocities = np.zeros(24)
        
        best_params = params.copy()
        best_violation = float('inf')
        
        for step in range(num_steps):
            if time.time() > deadline:
                break
                
            forces = compute_forces(params, outer_s)
            
            # Update velocities (Langevin)
            # F = m*a. m=1.
            # v = (v + F*dt) * damping + noise
            velocities = (velocities + forces * dt) * damping
            # Add thermal noise to escape local minima
            noise = np.random.normal(0, 0.05, 24)
            velocities += noise
            
            # Update positions
            params += velocities * dt
            
            # Check validity
            # Calculate violation for this state
            # If violation is 0, we have a valid packing.
            # We can stop early if we find a valid one? 
            # But we want to minimize S, so we just need ANY valid config for this S.
            # Let's check if it's valid.
            if step % 10 == 0:
                # Quick check: min distance between all pairs > 0 and containment
                # This is expensive. Let's just run the full sim.
                pass
                
        return params

    # Strategy: Binary search on Outer Side Length S
    LB = 2.5
    UB = 10.0
    
    best_solution = None
    best_s = float('inf')
    
    # Binary search loop
    for step in range(20):
        if time.time() > deadline:
            break
            
        mid_s = (LB + UB) / 2.0
        
        # Try to find a valid configuration for this S
        # Use multiple random restarts
        found_valid = False
        
        for restart in range(5):
            if time.time() > deadline:
                break
                
            params = simulate(mid_s, num_steps=1000)
            
            # Verify if the final configuration is valid
            xs = params[0:12]
            ys = params[12:24]
            valid = True
            
            # Containment check
            r = mid_s * math.sqrt(3) / 2.0
            normals = np.deg2rad(30 + np.arange(6)*60)
            n_vecs = np.column_stack([np.cos(normals), np.sin(normals)])
            
            for i in range(12):
                v = hex_verts(xs[i], ys[i], 1.0, 0.0)
                for vert in v:
                    dists = np.dot(vert, n_vecs.T)
                    if np.any(dists > r + 1e-6):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Overlap check
                inner_verts = [hex_verts(xs[i], ys[i], 1.0, 0.0) for i in range(12)]
                for i in range(12):
                    for j in range(i+1, 12):
                        d = min_dist_polygons(inner_verts[i], inner_verts[j])
                        if d < 1e-6:
                            # Check for actual overlap
                            # If d is 0, they might be touching or overlapping.
                            # We need to be sure they are disjoint.
                            # The task says "disjoint". Touching is usually allowed?
                            # "disjoint" in math sets means no common points.
                            # But in packing problems, "non-overlapping" often allows touching.
                            # Let's assume touching is OK, but strict overlap is not.
                            # My min_dist returns 0 for touching.
                            # If they overlap, min_dist is also 0.
                            # We need to distinguish.
                            # Use point_in_hex check.
                            overlap = False
                            for v in inner_verts[i]:
                                if point_in_hex(v, xs[j], ys[j], 1.0, 0.0):
                                    overlap = True
                                    break
                            if not overlap:
                                for v in inner_verts[j]:
                                    if point_in_hex(v, xs[i], ys[i], 1.0, 0.0):
                                        overlap = True
                                        break
                            if overlap:
                                valid = False
                                break
                    if not valid:
                        break
            
            if valid:
                found_valid = True
                best_solution = params.copy()
                break
        
        if found_valid:
            UB = mid_s
            best_s = mid_s
        else:
            LB = mid_s
            
    if best_solution is None:
        # Fallback
        inner = []
        for i in range(12):
            x = (i % 4) * 2.5 - 3.75
            y = (i // 4) * 2.2 - 2.2
            inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

    xs = best_solution[0:12]
    ys = best_solution[12:24]
    
    inner = []
    for i in range(12):
        inner.append([xs[i], ys[i], 0.0])
        
    return inner, (0.0, 0.0), best_s, 0.0

