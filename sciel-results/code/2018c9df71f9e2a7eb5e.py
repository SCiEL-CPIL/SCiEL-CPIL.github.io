import numpy as np
import time
import random

N = 12
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Helper: Get vertices of a regular hexagon
    # Side length 1, centered at (cx, cy), angle in degrees
    # Angle 0 means a vertex points to the right (standard math orientation for cos/sin start)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    def get_hex_verts(cx, cy, angle_deg):
        rad = np.radians(angle_deg)
        angles = rad + np.radians(np.arange(0, 360, 60))
        xs = cx + np.cos(angles)
        ys = cy + np.sin(angles)
        return np.column_stack([xs, ys])

    # Helper: Check if two convex polygons are disjoint using SAT
    # Returns (is_disjoint, min_translation_vector)
    # If disjoint, MTV is None. If overlapping, MTV is the vector to push p1 away from p2
    # (or p2 away from p1, we will define it as vector to add to p1's center to separate)
    def sat_check(p1, p2):
        # p1, p2 are (6, 2) arrays
        # We check axes from both polygons
        min_overlap = float('inf')
        best_axis = None
        
        for i in range(6):
            # Edge normal from p1
            v1 = p1[i]
            v2 = p1[(i + 1) % 6]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            
            # Project p1 and p2 onto normal
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 < min2 - 1e-8 or max2 < min1 - 1e-8:
                return True, None # Disjoint
            
            # Calculate overlap
            overlap1 = min1 - max2
            overlap2 = min2 - max1
            
            # We want the axis with the smallest overlap
            # If overlap1 is positive, p1 is "left" of p2 on this axis? 
            # Let's determine direction.
            # Vector from center of p1 to center of p2
            c1 = np.mean(p1, axis=0)
            c2 = np.mean(p2, axis=0)
            center_vec = c2 - c1
            
            # If center_vec dot normal > 0, p2 is in the direction of the normal.
            # So p1 needs to move in -normal direction to separate?
            # Actually, we just need the magnitude and the axis.
            
            if overlap1 > overlap2:
                # p1's min is past p2's max. 
                # This means p1 is "beyond" p2 in the normal direction.
                # To separate, move p1 in -normal direction? 
                # Wait. If min1 > max2, p1 is to the "right" of p2.
                # So moving p1 left (-normal) separates them.
                # Vector to add to p1: -normal * (min1 - max2)
                # But we want the MTV that minimizes movement.
                # Let's just store the overlap magnitude and the normal.
                # Direction will be determined by center vectors later.
                overlap = min1 - max2
                # If overlap > 0, they overlap.
                # The amount to push is 'overlap'.
                # Direction: p1 should move away from p2.
                # If center_vec dot normal > 0, p2 is in +normal direction.
                # So p1 should move in -normal direction.
                # MTV = -normal * overlap
                if np.dot(center_vec, normal) > 0:
                    mtv = -normal * overlap
                else:
                    mtv = normal * overlap
            else:
                # p2's min is past p1's max.
                # p2 is to the "left" of p1 (or p1 is to the right).
                # p1 should move in +normal direction?
                # If center_vec dot normal < 0, p2 is in -normal direction.
                # So p1 should move in +normal direction.
                overlap = min2 - max1
                if np.dot(center_vec, normal) > 0:
                    mtv = normal * overlap
                else:
                    mtv = -normal * overlap
            
            if overlap < min_overlap:
                min_overlap = overlap
                best_axis = mtv
        
        # Check p2 axes too
        for i in range(6):
            v1 = p2[i]
            v2 = p2[(i + 1) % 6]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - 1e-8 or max2 < min1 - 1e-8:
                return True, None
            
            c1 = np.mean(p1, axis=0)
            c2 = np.mean(p2, axis=0)
            center_vec = c2 - c1
            
            overlap1 = min1 - max2
            overlap2 = min2 - max1
            
            if overlap1 > overlap2:
                overlap = min1 - max2
                if np.dot(center_vec, normal) > 0:
                    mtv = -normal * overlap
                else:
                    mtv = normal * overlap
            else:
                overlap = min2 - max1
                if np.dot(center_vec, normal) > 0:
                    mtv = normal * overlap
                else:
                    mtv = -normal * overlap
            
            if overlap < min_overlap:
                min_overlap = overlap
                best_axis = mtv

        if min_overlap == float('inf'):
            return True, None
        else:
            return False, best_axis

    def calculate_bounding_side(positions, angles):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation 0 (vertex at (S,0)) that contains all vertices of the inner hexagons.
        Note: We assume outer hexagon is fixed at angle 0 for this calculation.
        If we allow outer rotation, we would need to optimize that too.
        Given the symmetry, aligning inner and outer is often optimal.
        """
        max_s = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, ang)
            for vx, vy in verts:
                # Constraint for regular hexagon (side S, angle 0):
                # |x| <= S
                # |y| <= S * sqrt(3)/2  -> S >= |y| * 2/sqrt(3)
                # |x| + |y|/sqrt(3) <= S -> S >= |x| + |y|/sqrt(3)
                
                s1 = abs(vx)
                s2 = abs(vy) * 2.0 / SQRT3
                s3 = abs(vx) + abs(vy) / SQRT3
                
                s_req = max(s1, s2, s3)
                if s_req > max_s:
                    max_s = s_req
        return max_s

    def check_all_disjoint(positions, angles):
        """
        Check if all hexagons are disjoint.
        Returns list of (i, j, mtv) for overlapping pairs.
        """
        overlaps = []
        verts = [get_hex_verts(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                is_dis, mtv = sat_check(verts[i], verts[j])
                if not is_dis:
                    overlaps.append((i, j, mtv))
        return overlaps

    def objective(positions, angles):
        """
        Score = Bounding Side + Penalty * Overlap Count
        We want to minimize this.
        """
        side = calculate_bounding_side(positions, angles)
        overlaps = check_all_disjoint(positions, angles)
        penalty = 10.0 * len(overlaps)
        return side + penalty

    # 1. Initialization: Compact Honeycomb
    # Axial coords (q, r)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    positions = []
    angles = []
    for q, r in coords:
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        positions.append([x, y])
        angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    
    # Verify initial validity
    # The honeycomb with these coords has touching sides (distance sqrt(3) between centers).
    # Side length 1, apothem sqrt(3)/2. Distance between centers = 2 * apothem = sqrt(3).
    # So they are touching. SAT should report disjoint (with eps).
    
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_score = objective(best_positions, best_angles)
    
    # 2. Simulated Annealing / Hill Climbing
    # We will perturb positions.
    # Temperature T starts high, decays.
    
    T = 0.5
    cooling = 0.999
    step_size = 0.5
    
    current_positions = positions.copy()
    current_angles = angles.copy()
    current_score = best_score
    
    iter_count = 0
    
    while time.time() < deadline:
        iter_count += 1
        
        # Pick a random hexagon to move
        i = random.randint(0, N - 1)
        
        # Generate a random move
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        d_angle = random.gauss(0, 5.0) # Small rotation
        
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        new_positions[i][0] += dx
        new_positions[i][1] += dy
        new_angles[i] += d_angle
        
        new_score = objective(new_positions, new_angles)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_positions = new_positions
            current_angles = new_angles
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_positions = current_positions.copy()
                best_angles = current_angles.copy()
                
                # If we found a valid configuration (score is just side length, no penalty),
                # we can try to reduce step size slightly
                if current_score < 10.0: # Heuristic: penalty is 10 per overlap, so <10 means likely valid or very small overlap
                    step_size *= 0.999
        else:
            # Revert
            pass
            
        # Cool down
        T *= cooling
        
        # If T is very low, restart with higher T if stuck?
        if T < 1e-4:
            T = 0.1
            step_size = 0.2
            # Optional: restart from best
            # current_positions = best_positions.copy()
            # current_angles = best_angles.copy()
            # current_score = best_score

    # Final Polish: Try to remove any remaining overlaps by pushing apart
    # If best_score > 10, it might have overlaps. Let's try to fix them.
    # But usually SA finds valid ones.
    
    # Ensure we return a valid configuration if possible.
    # If best_score is high, it's invalid.
    # Let's do a few iterations of "push apart" on the best solution to ensure validity.
    
    final_positions = best_positions.copy()
    final_angles = best_angles.copy()
    
    for _ in range(50):
        overlaps = check_all_disjoint(final_positions, final_angles)
        if not overlaps:
            break
        # Push the first overlapping pair apart
        i, j, mtv = overlaps[0]
        # Split the MTV
        # mtv is vector to push i away from j?
        # In my sat_check, mtv was calculated as vector to add to p1 (i) to separate.
        # So add mtv to i, and -mtv to j?
        # Wait, if I push i by mtv, does j move?
        # To maintain center of mass or just separate them:
        # Let's move i by 0.5 * mtv and j by -0.5 * mtv
        # But mtv magnitude is the overlap. Moving by full mtv ensures separation.
        # Moving by 0.5 * mtv might not be enough if there are other constraints, but let's try.
        # Actually, just moving i by mtv is safe for that pair, but might push i into others.
        # Let's move i by mtv * 1.01 and j by -mtv * 0.01 (bias towards moving the one further from center?)
        # Simpler: Just move i by mtv.
        final_positions[i] += mtv
        # Check if this made it worse? We just want validity.
        
    # Recalculate final score
    final_side = calculate_bounding_side(final_positions, final_angles)
    final_overlaps = check_all_disjoint(final_positions, final_angles)
    
    # If still invalid, fall back to a known valid configuration (expanded honeycomb)
    if final_overlaps:
        # Expand the initial honeycomb until valid
        scale = 1.0
        while scale < 10.0:
            test_pos = positions * scale
            test_ang = angles
            if not check_all_disjoint(test_pos, test_ang):
                scale += 0.1
            else:
                final_positions = test_pos
                final_angles = test_ang
                final_side = calculate_bounding_side(final_positions, final_angles)
                break
        if scale >= 10.0:
            # Emergency fallback
            final_positions = np.array([[i * 4.0, 0.0] for i in range(N)])
            final_angles = np.zeros(N)
            final_side = 40.0

    # Construct output
    inner = []
    for i in range(N):
        inner.append([float(final_positions[i][0]), float(final_positions[i][1]), float(final_angles[i])])
    
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
