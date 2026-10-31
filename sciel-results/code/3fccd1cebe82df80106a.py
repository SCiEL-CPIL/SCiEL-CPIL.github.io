import math
import time
import random
import numpy as np

N = 11

def construct():
    deadline = time.time() + 140.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Base hexagon vertices (side 1, centered at origin, pointy-right)
    base_verts = []
    for i in range(6):
        theta = math.radians(60 * i)
        base_verts.append((math.cos(theta), math.sin(theta)))
    base_verts = np.array(base_verts)

    def get_transformed_verts(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        rot = np.array([[ca, -sa], [sa, ca]])
        verts = base_verts @ rot.T
        return (verts * S) + np.array([cx, cy])

    def compute_outer_hex(inner, ocx, ocy, oangle):
        """
        Compute the minimum side length S of a regular hexagon centered at (ocx, ocy)
        with rotation oangle that contains all inner hexagons.
        """
        all_pts = []
        for (ix, iy, iangle) in inner:
            verts = get_transformed_verts(ix, iy, 1.0, iangle)
            all_pts.append(verts)
        all_pts = np.vstack(all_pts)
        
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        # The 6 outward normals of the outer hexagon
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            n_vec = np.array([nx, ny])
            
            # Project all points onto this normal
            projs = (all_pts - np.array([ocx, ocy])) @ n_vec
            max_proj = np.max(projs)
            
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def check_validity(inner, S, oangle):
        """
        Check if all inner hexagons are strictly inside the outer hexagon
        and disjoint from each other.
        """
        # 1. Check containment
        outer_verts = get_transformed_verts(0.0, 0.0, S, oangle)
        # For each inner hex, check if all vertices are inside outer
        # Using winding number or ray casting is slow, but for hexagons, 
        # checking vertices against the 6 half-planes of the outer hex is faster.
        
        a = math.radians(oangle)
        apothem = S * math.sqrt(3) / 2
        
        for (ix, iy, iangle) in inner:
            verts = get_transformed_verts(ix, iy, 1.0, iangle)
            for v in verts:
                # Check against 6 half-planes
                for k in range(6):
                    theta = a + math.radians(30 + 60 * k)
                    nx, ny = math.cos(theta), math.sin(theta)
                    # Distance from center to plane is apothem
                    # Normal points outward. 
                    # Vector from center to vertex:
                    vx, vy = v[0], v[1]
                    proj = vx * nx + vy * ny
                    if proj > apothem + 1e-9:
                        return False

        # 2. Check disjointness (pairwise)
        # For hexagons, disjointness is equivalent to:
        # a) No vertex of A is inside B
        # b) No vertex of B is inside A
        # c) No edge of A intersects edge of B
        # This is O(N^2 * 36) which is fast for N=11.
        
        inner_verts = [get_transformed_verts(ix, iy, 1.0, iangle) for (ix, iy, iangle) in inner]
        
        def point_in_hex(p, hex_verts):
            # Check if p is inside hex_verts
            # Use cross products for consistent winding
            sign = 0
            for i in range(6):
                p1 = hex_verts[i]
                p2 = hex_verts[(i+1)%6]
                cross = (p2[0]-p1[0])*(p[1]-p1[1]) - (p2[1]-p1[1])*(p[0]-p1[0])
                if abs(cross) < 1e-9:
                    continue
                cur_sign = 1 if cross > 0 else -1
                if sign == 0:
                    sign = cur_sign
                elif sign != cur_sign:
                    return False
            return True

        def segments_intersect(p1, p2, p3, p4):
            def cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            d1 = cross(p3, p4, p1)
            d2 = cross(p3, p4, p2)
            d3 = cross(p1, p2, p3)
            d4 = cross(p1, p2, p4)
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return True
            return False

        for i in range(N):
            for j in range(i+1, N):
                vi = inner_verts[i]
                vj = inner_verts[j]
                # Check vertices
                for v in vi:
                    if point_in_hex(v, vj):
                        return False
                for v in vj:
                    if point_in_hex(v, vi):
                        return False
                # Check edges
                for k in range(6):
                    a1, a2 = vi[k], vi[(k+1)%6]
                    for l in range(6):
                        b1, b2 = vj[l], vj[(l+1)%6]
                        if segments_intersect(a1, a2, b1, b2):
                            return False
        return True

    def run_simulation(initial_positions, seed):
        """
        Run a force-directed simulation to pack the hexagons.
        """
        rng = random.Random(seed)
        # State: positions (x, y) and angles
        pos = np.array([p[:2] for p in initial_positions], dtype=float)
        ang = np.array([p[2] for p in initial_positions], dtype=float)
        
        # Velocity
        vel = np.zeros((N, 2))
        
        # Parameters
        repulsion_k = 100.0
        boundary_k = 50.0
        damping = 0.9
        dt = 0.01
        
        # We simulate in a "virtual" container that shrinks.
        # Or we just minimize the objective: Max_Projection + Penalty(Overlap)
        # Actually, simpler: 
        # 1. Repel overlapping hexes.
        # 2. Attract hexes towards the center (or towards each other).
        # 3. The "size" S is determined by the extent.
        
        # Let's use a potential energy approach.
        # E = Sum(Overlap_Penalty) + Attraction_to_Center
        
        # Overlap detection for force calculation is expensive if done exactly.
        # Approximation: Treat hexes as circles of radius R = apothem = sqrt(3)/2?
        # No, that's too loose.
        # Let's use the exact vertex check for validity, but for forces, 
        # we can use a soft distance metric.
        
        # Better approach for forces:
        # If two hexes overlap, compute a repulsive force.
        # Simple proxy: Distance between centers.
        # Min distance for disjoint hexes (aligned) is 2 * apothem = sqrt(3).
        # If dist < sqrt(3), they likely overlap (unless rotated favorably).
        # We will use a soft potential: V = k * (d_min - d)^2 if d < d_min.
        # d_min depends on angle difference. For simplicity, use d_min = sqrt(3).
        
        d_min = math.sqrt(3)
        
        for step in range(2000):
            if time.time() > deadline:
                break
                
            forces = np.zeros((N, 2))
            
            # 1. Repulsion between pairs
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = pos[i] - pos[j]
                    d = np.linalg.norm(d_vec)
                    if d < 1e-6:
                        d_vec = np.array([rng.gauss(0, 0.1), rng.gauss(0, 0.1)])
                        d = np.linalg.norm(d_vec)
                    
                    # If they are too close, repel
                    if d < d_min:
                        # Force magnitude proportional to penetration
                        f_mag = repulsion_k * (d_min - d)
                        f_dir = d_vec / d
                        forces[i] += f_dir * f_mag
                        forces[j] -= f_dir * f_mag
            
            # 2. Attraction to center (to keep them compact)
            # We want them to cluster, not fly apart.
            # But we also want to minimize the bounding box.
            # A simple spring to the centroid works well.
            centroid = np.mean(pos, axis=0)
            for i in range(N):
                forces[i] -= 0.5 * (pos[i] - centroid) # Weak spring to centroid
            
            # 3. Update velocity and position
            vel = (vel + forces * dt) * damping
            pos += vel * dt
            
            # 4. Calculate current S
            inner_curr = []
            for i in range(N):
                inner_curr.append((pos[i][0], pos[i][1], ang[i]))
            
            # Try a few outer angles to find the best fit for this configuration
            best_local_S = float('inf')
            best_local_angle = 0.0
            for oangle in [0, 30, 60]:
                S = compute_outer_hex(inner_curr, 0.0, 0.0, oangle)
                # We don't check validity here every step (too slow), 
                # but we can check if S is improving.
                # Actually, we should check validity periodically.
                if S < best_local_S:
                    best_local_S = S
                    best_local_angle = oangle
            
            # Periodic validity check
            if step % 50 == 0 or step == 1999:
                if check_validity(inner_curr, best_local_S, best_local_angle):
                    if best_local_S < best_S:
                        best_S = best_local_S
                        best_inner = [list(x) for x in inner_curr]
                        best_center = (0.0, 0.0)
                        best_angle = best_local_angle
                        best_valid = True
                        # If we found a valid one, we can try to tighten it further
                        # by running a local refinement step? 
                        # For now, the simulation continues to compress.

    # Initial configurations
    # 1. Lattice start (good for initialization)
    # 2. Random start
    
    # Lattice start
    sp = math.sqrt(3)
    lattice_pts = []
    # Hexagonal cluster of 11
    # Center + 6 + 4? 
    # 1 in center, 6 around it, then 4 more? 
    # Or just take 11 closest points from a large grid.
    grid = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * sp + j * sp / 2
            y = j * sp * math.sqrt(3) / 2
            grid.append((x, y))
    grid.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_lattice = [(p[0], p[1], 0.0) for p in grid[:N]]
    
    # Random start
    init_random = []
    for _ in range(N):
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
        init_random.append((x, y, random.uniform(0, 60)))

    # Run simulations with different seeds and initializations
    # We have ~140s. 
    # Simulation with 2000 steps * N^2 (121) checks is fast.
    # 10 runs should be fine.
    
    inits = [init_lattice, init_random]
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    idx = 0
    for seed in seeds:
        if time.time() > deadline:
            break
        init = inits[idx % len(inits)]
        # Jitter the initial positions slightly to avoid symmetry traps
        jittered_init = []
        for p in init:
            jittered_init.append((p[0] + random.gauss(0, 0.1), p[1] + random.gauss(0, 0.1), p[2]))
        
        run_simulation(jittered_init, seed)
        idx += 1

    # Fallback if no valid solution found
    if not best_valid or best_inner is None:
        # Construct a valid linear chain as fallback
        fallback_inner = []
        for k in range(N):
            fallback_inner.append([(k - N // 2) * (math.sqrt(3) + 0.1), 0.0, 0.0])
        S = compute_outer_hex(fallback_inner, 0.0, 0.0, 0.0)
        # Ensure it fits
        while not check_validity(fallback_inner, S, 0.0):
            for k in range(N):
                fallback_inner[k][0] *= 1.05
            S = compute_outer_hex(fallback_inner, 0.0, 0.0, 0.0)
        return fallback_inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
