import numpy as np
import time
import math
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    # --- Geometry Helpers ---
    
    def hex_vertices(cx, cy, side, angle_deg):
        """
        Returns vertices of a regular hexagon centered at (cx, cy) with given side and angle.
        Angle 0 corresponds to a 'pointy-top' hexagon where one vertex is at (cx+side, cy).
        """
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def min_containing_side(vertices, outer_angle_deg=0.0):
        """
        Calculates the minimum side length S of a regular hexagon centered at (0,0)
        with rotation outer_angle_deg that contains all points in 'vertices'.
        
        A regular hexagon with side S and rotation alpha is the intersection of 3 strips.
        The distance from center to each side (apothem) is a = S * sqrt(3) / 2.
        The normals to the sides are at angles alpha + 30, alpha + 90, alpha + 150 degrees.
        Actually, for a hexagon with vertices at alpha + 0, 60, ..., the sides are perpendicular 
        to directions alpha + 30, alpha + 90, alpha + 150.
        
        Let n_k be the unit normal to the k-th side.
        The constraint is n_k . p <= a for all points p.
        So a >= max( n_k . p ) for all k, p.
        Since S = 2 * a / sqrt(3), we have S = 2/sqrt(3) * max_k max_p (n_k . p).
        Also we must consider the negative side: -n_k . p <= a => a >= max_p (-n_k . p).
        So a >= max( max_p(n_k . p), max_p(-n_k . p) ) = max_p |n_k . p|.
        
        Thus S = (2/sqrt(3)) * max_{k in 0..2} max_{p in vertices} | n_k . p |
        """
        alpha = np.radians(outer_angle_deg)
        # Normals to the 3 pairs of parallel sides
        # Vertices are at alpha, alpha+60, ...
        # Side between vertex 0 and 1 has normal at alpha + 30.
        # Side between vertex 1 and 2 has normal at alpha + 90.
        # Side between vertex 2 and 3 has normal at alpha + 150.
        normal_angles = [alpha + np.pi/6, alpha + np.pi/6 + np.pi/3, alpha + np.pi/6 + 2*np.pi/3]
        
        max_proj = 0.0
        for n_ang in normal_angles:
            n = np.array([np.cos(n_ang), np.sin(n_ang)])
            projs = np.abs(vertices.dot(n))
            if np.max(projs) > max_proj:
                max_proj = np.max(projs)
        
        return max_proj * 2.0 / np.sqrt(3.0)

    def check_overlap(pos1, pos2, ang1, ang2, eps=1e-5):
        """
        Checks if two unit hexagons at pos1, ang1 and pos2, ang2 overlap.
        Uses SAT (Separating Axis Theorem).
        """
        v1 = hex_vertices(pos1[0], pos1[1], 1.0, ang1)
        v2 = hex_vertices(pos2[0], pos2[1], 1.0, ang2)
        
        # Axes to check: normals to edges of v1 and v2
        axes = []
        for i in range(6):
            edge1 = v1[(i+1)%6] - v1[i]
            axes.append(np.array([-edge1[1], edge1[0]]))
            edge2 = v2[(i+1)%6] - v2[i]
            axes.append(np.array([-edge2[1], edge2[0]]))
            
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-10: continue
            axis = axis / norm
            
            proj1 = v1.dot(axis)
            proj2 = v2.dot(axis)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - eps or max2 < min1 - eps:
                return False # Disjoint
        
        return True # Overlapping

    def is_valid_config(positions, angles):
        """
        Checks if the entire configuration is valid (no overlaps).
        positions: list of (x,y)
        angles: list of degrees
        """
        for i in range(N):
            for j in range(i+1, N):
                if check_overlap(positions[i], positions[j], angles[i], angles[j]):
                    return False
        return True

    def get_score(positions, angles, outer_angle=0.0):
        """
        Returns the side length of the minimal containing hexagon.
        If invalid, returns infinity.
        """
        if not is_valid_config(positions, angles):
            return float('inf')
        
        # Collect all vertices of all inner hexagons
        all_verts = []
        for i in range(N):
            v = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        return min_containing_side(all_verts, outer_angle)

    # --- Optimization Strategy: Simulated Annealing on Hexagonal Lattice ---
    
    # We restrict positions to a hexagonal lattice to maintain near-tight packing 
    # and reduce the search space significantly.
    # Axial coordinates (q, r) -> Cartesian (x, y)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)

    def cartesian_to_axial(x, y):
        # Approximate inverse, then round to nearest valid axial coordinate
        q = (math.sqrt(3) / 3 * x + 1 / 3 * y)
        r = (2 / 3 * y)
        
        # Round to nearest integers
        q_int = round(q)
        r_int = round(r)
        
        # The third coordinate s = -q - r. We should check if (q,r,s) is close to integer.
        # For now, just returning the rounded q,r is a decent approximation for lattice search.
        return (q_int, r_int)

    # Initial Configuration: A compact cluster
    # We generate a set of candidate axial coordinates in a small radius
    # and select 12 that are mutually non-overlapping (distance >= 2*apothem = sqrt(3))
    # Actually, on a hex lattice, neighbors are at distance sqrt(3).
    # So any set of distinct lattice points is non-overlapping (touching is allowed).
    
    # Generate a grid of axial coordinates
    max_radius = 4
    candidates = []
    for q in range(-max_radius, max_radius + 1):
        for r in range(-max_radius, max_radius + 1):
            if abs(q + r) <= max_radius:
                candidates.append((q, r))
                
    # Randomly shuffle candidates to break symmetry
    random.seed(42)
    random.shuffle(candidates)
    
    # Select first 12 distinct candidates as initial positions
    # Since they are on a lattice, they are guaranteed to be non-overlapping (touching at most)
    initial_axial = candidates[:N]
    
    positions = [axial_to_cartesian(q, r) for q, r in initial_axial]
    angles = [0.0] * N  # Start with all aligned
    
    current_score = get_score(positions, angles, 0.0)
    best_score = current_score
    best_positions = [p.copy() for p in positions]
    best_angles = angles.copy()
    best_outer_angle = 0.0
    
    if not math.isinf(current_score):
        # Update best if valid
        pass

    # Simulated Annealing Parameters
    T = 1.0  # Temperature
    T_min = 0.01
    alpha_cool = 0.999
    max_iter = 50000
    iter_count = 0
    
    # Directions to move in axial space
    axial_dirs = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random hexagon to move
        i = random.randint(0, N - 1)
        
        # Pick a random direction in axial space
        dq, dr = random.choice(axial_dirs)
        
        # Get current axial coordinates (approximate)
        # Note: We store cartesian positions. To move on lattice, we need to track axial coords.
        # Let's refactor to store axial coords internally for the SA loop.
        
        # Refactoring state to axial for cleaner lattice moves
        # This requires re-initializing or converting. Let's do a quick conversion for the current best/curr.
        # Since we started on lattice and only move by lattice steps, we can track axial coords.
        
        # Let's restart the SA loop with axial state tracking for efficiency and correctness.
        # This is a significant structural change inside the loop, so let's just implement the axial version directly.
        
        break # Exit this simple loop, implement axial SA below

    # --- Axial Simulated Annealing Implementation ---
    
    # State: list of (q, r)
    # We need to ensure we don't pick duplicate coordinates.
    
    def get_axial_score(axial_coords, angles, outer_angle=0.0):
        positions = [axial_to_cartesian(q, r) for q, r in axial_coords]
        return get_score(positions, angles, outer_angle)

    # Initialize axial state
    current_axial = [tuple(c) for c in initial_axial]
    current_angles = [0.0] * N
    current_score = get_axial_score(current_axial, current_angles, 0.0)
    
    if math.isinf(current_score):
        # Should not happen for distinct lattice points, but just in case
        pass

    best_axial = [tuple(c) for c in current_axial]
    best_angles = current_angles.copy()
    best_score = current_score
    best_outer_angle = 0.0

    T = 10.0
    T_min = 0.01
    alpha_cool = 0.9995
    
    # Precompute valid moves? No, dynamic is fine.
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # 1. Move a random hexagon to a neighboring lattice site
        i = random.randint(0, N - 1)
        dq, dr = random.choice(axial_dirs)
        
        new_q = current_axial[i][0] + dq
        new_r = current_axial[i][1] + dr
        new_coord = (new_q, new_r)
        
        # Check if new_coord is occupied by another hexagon
        occupied = False
        for j in range(N):
            if j == i: continue
            if current_axial[j] == new_coord:
                occupied = True
                break
        
        if occupied:
            T *= alpha_cool
            continue
            
        # Perform move
        old_coord = current_axial[i]
        current_axial[i] = new_coord
        
        # Calculate new score
        new_score = get_axial_score(current_axial, current_angles, 0.0)
        
        # Acceptance criterion
        if new_score < current_score:
            current_score = new_score
            # Update best
            if new_score < best_score:
                best_score = new_score
                best_axial = [tuple(c) for c in current_axial]
                best_angles = current_angles.copy()
        else:
            # Probability of accepting worse solution
            delta = new_score - current_score
            if delta < 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / T)
                
            if random.random() < prob:
                current_score = new_score
            else:
                # Reject move
                current_axial[i] = old_coord
                
        T *= alpha_cool
        
        # 2. Occasionally rotate a hexagon (30 degrees)
        # Rotations change the geometry relative to the lattice, potentially allowing tighter packing
        # if the outer hexagon is also rotated, or just to fit in gaps better.
        # However, on a lattice, rotating inner hexagons might cause overlaps with neighbors 
        # that were previously just touching.
        # Let's try rotating by 30 degrees (pointy to flat).
        if random.random() < 0.1:
            i = random.randint(0, N - 1)
            old_angle = current_angles[i]
            current_angles[i] = (old_angle + 30.0) % 360.0
            
            new_score = get_axial_score(current_axial, current_angles, 0.0)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_axial = [tuple(c) for c in current_axial]
                    best_angles = current_angles.copy()
            else:
                delta = new_score - current_score
                prob = math.exp(-delta / T) if delta > 0 else 1.0
                if random.random() < prob:
                    current_score = new_score
                else:
                    current_angles[i] = old_angle

    # --- Final Refinement: Optimize Outer Hexagon Angle ---
    # The inner positions are fixed on the lattice. We can rotate the outer hexagon to minimize its size.
    # We scan outer angle from 0 to 60 degrees.
    
    final_positions = [axial_to_cartesian(q, r) for q, r in best_axial]
    
    # Collect all vertices
    all_verts = []
    for i in range(N):
        v = hex_vertices(final_positions[i][0], final_positions[i][1], 1.0, best_angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    best_outer_angle = 0.0
    best_side = best_score
    
    # Scan angles
    for ang in range(0, 61, 1):
        s = min_containing_side(all_verts, float(ang))
        if s < best_side:
            best_side = s
            best_outer_angle = float(ang)
            
    # Fine-tune outer angle
    # Use a small local search around best_outer_angle
    for _ in range(50):
        step = 0.1
        for da in [-step, step]:
            new_ang = best_outer_angle + da
            if new_ang < 0: new_ang += 60
            if new_ang >= 60: new_ang -= 60
            s = min_containing_side(all_verts, new_ang)
            if s < best_side:
                best_side = s
                best_outer_angle = new_ang
                step *= 0.5

    # Construct final output
    inner = []
    for i in range(N):
        x, y = final_positions[i]
        inner.append([float(x), float(y), float(best_angles[i])])
        
    # Outer hexagon center is (0,0) because we centered the vertices in min_containing_side logic?
    # Wait, min_containing_side assumed center (0,0).
    # The positions are relative to the centroid of the cluster?
    # No, the axial coordinates are absolute.
    # If the cluster is not centered at (0,0), the outer hexagon centered at (0,0) might not be optimal 
    # or might not contain them if they are far from origin.
    # However, the problem allows the outer hexagon to have a center (x,y).
    # My `min_containing_side` calculates the size assuming the outer hex is at (0,0).
    # To be general, I should translate the inner hexagons so their "center of mass" or bounding box center
    # is at (0,0), or simply return the outer center as the centroid of the inner hexagons' bounding box.
    
    # Let's calculate the centroid of the inner hexagons to shift the coordinate system
    # so that the outer hexagon is centered at (0,0) in the returned local frame, 
    # or just return the absolute positions and the outer center.
    
    # The interface allows outer center (x,y).
    # If I use `min_containing_side` on raw coordinates, it finds the smallest hex at (0,0) containing them.
    # This is equivalent to a hex at (Cx, Cy) containing them if we shift coordinates by -C.
    # So, if I return inner positions as calculated, and outer center as (0,0), 
    # the evaluator will check if inner hexes are inside outer hex at (0,0).
    # This works IF the cluster is roughly centered at (0,0).
    # My initial cluster was centered near (0,0) because axial coords were small and balanced.
    # However, SA might drift.
    
    # To be safe, let's center the cluster at (0,0) before returning.
    # Calculate centroid of all inner hexagon centers.
    cx_avg = sum(p[0] for p in final_positions) / N
    cy_avg = sum(p[1] for p in final_positions) / N
    
    # Shift all inner positions by -centroid
    shifted_positions = [(p[0] - cx_avg, p[1] - cy_avg) for p in final_positions]
    
    # Recalculate best_side with shifted positions to be precise (should be same as before due to translation invariance of size)
    # Actually, the size is invariant to translation. The shape is the same.
    # So best_side is correct.
    
    # But we must ensure the outer hexagon centered at (0,0) contains the shifted inner hexagons.
    # Yes, because we calculated best_side based on the geometry relative to the cluster's own center?
    # No, `min_containing_side` assumed the OUTER hex is at (0,0).
    # If the inner cluster was at (100, 0), `min_containing_side` would return a huge S.
    # By shifting inner positions to be centered at (0,0), we ensure the outer hex at (0,0) is tight.
    
    # Let's re-verify:
    # 1. We have positions P.
    # 2. We computed S = min_containing_side(P, angle). This S is the size of hex at (0,0) containing P.
    # 3. We shift P to P' = P - Centroid(P).
    # 4. We return P' and Outer Center (0,0).
    # 5. The evaluator checks if P' is inside Hex(0,0, S, angle).
    # 6. Since P' is just P translated, and Hex(0,0) is the reference, 
    #    does Hex(0,0) contain P'?
    #    Only if Hex(0,0) contained P - Centroid.
    #    My calculation `min_containing_side(P)` finds S such that Hex(0,0, S) contains P.
    #    It does NOT guarantee Hex(0,0, S) contains P - Centroid.
    #    In fact, if P is far from 0, S is large. P - Centroid is near 0.
    #    The minimal S for P - Centroid would be smaller!
    
    # CORRECTION:
    # I should calculate the score based on the SHIFTED positions.
    # Let's shift the positions first, then calculate the optimal outer angle and size.
    
    shifted_positions = [(p[0] - cx_avg, p[1] - cy_avg) for p in final_positions]
    
    # Recalculate all vertices with shifted positions
    all_verts_shifted = []
    for i in range(N):
        x, y = shifted_positions[i]
        v = hex_vertices(x, y, 1.0, best_angles[i])
        all_verts_shifted.append(v)
    all_verts_shifted = np.vstack(all_verts_shifted)
    
    best_outer_angle = 0.0
    best_side = float('inf')
    
    # Scan angles again for the shifted configuration
    for ang in range(0, 61, 1):
        s = min_containing_side(all_verts_shifted, float(ang))
        if s < best_side:
            best_side = s
            best_outer_angle = float(ang)
            
    # Fine tune
    step = 0.1
    for _ in range(50):
        improved = False
        for da in [-step, step]:
            new_ang = best_outer_angle + da
            if new_ang < 0: new_ang += 60
            if new_ang >= 60: new_ang -= 60
            s = min_containing_side(all_verts_shifted, new_ang)
            if s < best_side:
                best_side = s
                best_outer_angle = new_ang
                step *= 0.5
                improved = True
        if not improved:
            step *= 0.5
            if step < 0.001:
                break

    # Final Inner List
    inner = []
    for i in range(N):
        x, y = shifted_positions[i]
        inner.append([float(x), float(y), float(best_angles[i])])
        
    return inner, (0.0, 0.0), float(best_side), float(best_outer_angle)

# EVOLVE-BLOCK-END
