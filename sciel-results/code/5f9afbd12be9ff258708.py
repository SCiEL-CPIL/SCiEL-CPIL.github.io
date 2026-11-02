import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin for 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def polygons_disjoint_sat(v1, v2):
        """
        Check if two convex polygons are disjoint using Separating Axis Theorem.
        Returns True if disjoint, False if overlapping.
        """
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    def min_outer_side(inner_positions, outer_angle_deg):
        """
        Calculate the minimum side length of a regular hexagon with given angle
        and center (0,0) that contains all inner hexagons.
        """
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        
        if not all_verts:
            return 0.0
        
        all_verts = np.vstack(all_verts)
        
        base_normals = np.radians(np.arange(30, 360, 60))
        outer_rad = np.radians(outer_angle_deg)
        normals = base_normals + outer_rad
        
        max_proj = 0.0
        for na in normals:
            nx = np.cos(na)
            ny = np.sin(na)
            projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        
        return max_proj / (np.sqrt(3) / 2)

    def get_min_outer_config(inner_positions):
        """
        Find the best outer hexagon (center, side, angle) for given inner positions.
        """
        best_s = float('inf')
        best_theta = 0.0
        
        # Scan theta
        for theta in np.arange(0, 60, 1.0):
            s = min_outer_side(inner_positions, theta)
            if s < best_s:
                best_s = s
                best_theta = theta
        
        # Refine theta around best
        for theta in np.arange(best_theta - 1.0, best_theta + 1.0, 0.1):
            s = min_outer_side(inner_positions, theta)
            if s < best_s:
                best_s = s
                best_theta = theta
                
        return best_s, best_theta

    def objective(params):
        """
        params: 3*N values for inner hexagons (x, y, angle).
        Outer hexagon is assumed centered at (0,0).
        Returns the side length of the minimal enclosing outer hexagon + penalty for overlap.
        """
        inner_positions = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_positions.append((x, y, a))
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                if not polygons_disjoint_sat(v1, v2):
                    penalty += 10.0
        
        s, theta = get_min_outer_config(inner_positions)
        
        return s + penalty

    def local_search(params, step_size, angle_step, max_iters=500):
        """
        Perform local search by trying to move each hexagon to a better neighbor.
        """
        current_params = params.copy()
        current_score = objective(current_params)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                # Try 8 spatial neighbors + 3 angle neighbors + original
                # Spatial: N, S, E, W, NE, NW, SE, SW
                # Angle: +angle_step, -angle_step
                
                best_neighbor_params = current_params
                best_neighbor_score = current_score
                
                # Spatial moves
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        trial = current_params.copy()
                        trial[3*i] += dx
                        trial[3*i+1] += dy
                        score = objective(trial)
                        if score < best_neighbor_score:
                            best_neighbor_score = score
                            best_neighbor_params = trial
                            
                # Angle moves
                for da in [-angle_step, angle_step]:
                    trial = current_params.copy()
                    trial[3*i+2] += da
                    score = objective(trial)
                    if score < best_neighbor_score:
                        best_neighbor_score = score
                        best_neighbor_params = trial
                        
                if best_neighbor_score < current_score:
                    current_params = best_neighbor_params
                    current_score = best_neighbor_score
                    improved = True
                    
            if not improved:
                break
                
        return current_params, current_score

    def generate_flower_config(seed):
        """
        Generate a 'flower' configuration: 1 center, 6 surrounding, 5 outer.
        This is a topologically distinct starting point from a random grid.
        """
        np.random.seed(seed)
        init_params = np.zeros(3 * N)
        
        # Center
        init_params[0:3] = [0.0, 0.0, 0.0]
        
        # Ring 1: 6 hexagons touching the center
        # Distance between centers of touching unit hexagons is sqrt(3) * 2 / 2 * 2? 
        # No, apothem is sqrt(3)/2. Distance between centers of touching hexagons is 2 * apothem = sqrt(3).
        # Wait, if they touch edge-to-edge, the distance is 2 * apothem = sqrt(3).
        # Let's verify: Hexagon side 1. Apothem sqrt(3)/2. 
        # If two hexagons touch at an edge, the distance between centers is 2 * apothem = sqrt(3).
        ring1_dist = math.sqrt(3)
        
        for k in range(6):
            angle = math.radians(60 * k)
            idx = (k + 1) * 3
            init_params[idx:idx+3] = [ring1_dist * math.cos(angle), ring1_dist * math.sin(angle), 0.0]
            
        # Ring 2: 5 hexagons placed in the "valleys" between the ring 1 hexagons
        # The valleys are at angles 30, 90, 150...
        # The distance to the valley center is roughly ring1_dist * 2 / sqrt(3)?
        # Let's just place them slightly further out and let the optimizer adjust.
        # A good guess for the second ring in a hexagonal lattice is distance 2 * sqrt(3) from center?
        # Actually, in a hexagonal lattice of touching hexagons:
        # Center (0,0)
        # Ring 1: 6 at distance sqrt(3)
        # Ring 2: 12 at distance 2*sqrt(3) (6 aligned with Ring 1, 6 in between)
        # We only have 5 left. Let's pick 5 of the 12 positions in Ring 2.
        ring2_dist = 2 * math.sqrt(3)
        
        # The 12 positions in Ring 2 are at angles 0, 30, 60, ... 330
        # We pick 5. To maintain symmetry-ish, maybe skip every other one?
        # Indices 1, 3, 5, 7, 9 correspond to angles 30, 90, 150, 210, 270?
        # Let's just pick 5 distinct angles.
        angles_to_use = [30, 90, 150, 210, 270]
        
        for k, ang_deg in enumerate(angles_to_use):
            angle = math.radians(ang_deg)
            idx = (k + 7) * 3
            init_params[idx:idx+3] = [ring2_dist * math.cos(angle), ring2_dist * math.sin(angle), 0.0]
            
        # Add slight random perturbation to break symmetry and help escape local minima
        for i in range(1, N):
            init_params[3*i] += np.random.uniform(-0.1, 0.1)
            init_params[3*i+1] += np.random.uniform(-0.1, 0.1)
            init_params[3*i+2] += np.random.uniform(-5, 5)
            
        return init_params

    best_params = None
    best_score = float('inf')
    
    # Multi-start with Flower Configuration
    # We will run several local searches with different starting points and step sizes.
    
    starts = [
        (100, 0.2, 5.0),
        (101, 0.2, 5.0),
        (102, 0.2, 5.0),
        (103, 0.2, 5.0),
        (104, 0.2, 5.0),
        (105, 0.2, 5.0),
        (106, 0.2, 5.0),
        (107, 0.2, 5.0),
    ]
    
    for seed, step, angle_step in starts:
        if time.time() > deadline:
            break
            
        init_params = generate_flower_config(seed)
        
        # Coarse search
        p1, s1 = local_search(init_params, step, angle_step, max_iters=50)
        
        # Medium search
        p2, s2 = local_search(p1, step * 0.5, angle_step * 0.5, max_iters=50)
        
        # Fine search
        p3, s3 = local_search(p2, step * 0.25, angle_step * 0.25, max_iters=50)
        
        if s3 < best_score:
            best_score = s3
            best_params = p3
            
    if best_params is None:
        # Fallback to initial
        best_params = generate_flower_config(100)
        best_score = objective(best_params)
        
    # Final refinement with very small steps
    if time.time() < deadline - 10:
        best_params, best_score = local_search(best_params, 0.05, 1.0, max_iters=50)
        
    # Extract best inner positions
    inner = []
    for i in range(N):
        inner.append((best_params[3*i], best_params[3*i+1], best_params[3*i+2]))
        
    # Compute final outer side and angle
    s, theta = get_min_outer_config(inner)
    
    # Center is (0,0)
    center = (0.0, 0.0)
    side = s
    angle = theta
    
    # Validation check
    valid = True
    for cx, cy, ang in inner:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, center[0], center[1], side, angle):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback: expand side slightly
        side *= 1.01
        
    return inner, center, side, angle

# EVOLVE-BLOCK-END
