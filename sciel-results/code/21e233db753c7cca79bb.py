"""
Strategy: Simulated Annealing with direct outer-hexagon minimization.
Instead of optimizing the outer hexagon parameters alongside the inner ones, 
we treat the outer hexagon as the minimal container for the current inner 
configuration. This reduces the dimensionality and allows the SA to focus 
on packing the inner hexagons tightly. The outer side length is computed 
by scanning the rotation angle of the outer hexagon to find the minimum 
size that contains all inner hexagons.
"""
import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

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
        # Normals to the sides of a regular hexagon centered at origin
        # Sides are at 30, 90, 150, 210, 270, 330 degrees
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
        We assume the outer hexagon is centered at (0,0) for this calculation,
        and we will shift the inner positions relative to the outer center later.
        Actually, it's better to compute the bounding box of the inner hexagons
        in the coordinate system of the outer hexagon.
        
        For a regular hexagon with side S and angle theta, the half-width in 
        direction phi is S * cos(phi - theta - 30deg) ... wait.
        
        Let's use the support function.
        The outer hexagon H_out is defined by 6 half-planes.
        For a fixed outer angle theta_out, the hexagon is:
        { (x,y) | n_i . (x,y) <= S * apothem_factor }
        where n_i are unit normals to the sides.
        For a regular hexagon of side S, the apothem is S * sqrt(3)/2.
        So the constraint is n_i . (x,y) <= S * sqrt(3)/2.
        To find minimum S containing a set of points P:
        S = max_{p in P, i in 1..6} (n_i . p) / (sqrt(3)/2)
        
        We need to consider all vertices of all inner hexagons as points P.
        """
        # Collect all vertices of all inner hexagons
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        
        if not all_verts:
            return 0.0
        
        all_verts = np.vstack(all_verts)
        
        # Normals for the outer hexagon at angle outer_angle_deg
        # The sides of a regular hexagon are perpendicular to vectors at 30, 90, ...
        # If the hexagon is rotated by outer_angle_deg, the normals are rotated too.
        base_normals = np.radians(np.arange(30, 360, 60))
        outer_rad = np.radians(outer_angle_deg)
        normals = base_normals + outer_rad
        
        # Calculate max projection
        max_proj = 0.0
        for na in normals:
            nx = np.cos(na)
            ny = np.sin(na)
            projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        
        # S * sqrt(3)/2 >= max_proj  => S >= max_proj / (sqrt(3)/2)
        return max_proj / (np.sqrt(3) / 2)

    def get_min_outer_config(inner_positions):
        """
        Find the best outer hexagon (center, side, angle) for given inner positions.
        We assume the outer hexagon is centered at the centroid of the inner hexagons' bounding box?
        Or simply, we can fix the outer center to (0,0) and shift the inner positions?
        No, the interface returns inner positions and outer center.
        
        To simplify, let's fix the outer center to (0,0) and optimize inner positions
        such that they are packed around (0,0). Then the outer side is determined.
        However, the outer hexagon can be shifted. But shifting the outer hexagon is
        equivalent to shifting all inner hexagons in the opposite direction.
        So we can fix the outer center at (0,0) and just return the inner positions
        relative to (0,0). The score is the side length.
        
        Wait, the interface is:
        inner: rows (x, y, angle)
        center: (x, y) of outer
        side: outer side
        angle: outer angle
        
        If I fix outer center at (0,0), I just return (0,0) as center.
        The inner positions are absolute.
        
        So, for a given set of inner positions, I can compute the min side length
        for a given outer angle. To find the global min side, I should scan the outer angle.
        """
        # Scan outer angle from 0 to 60 degrees (symmetry)
        # Also, we might want to center the inner cluster.
        # Let's compute the centroid of the inner hexagons and shift them to center at (0,0)?
        # No, if we shift inner positions, the relative geometry is preserved.
        # But the "min enclosing hexagon" depends on the position relative to the center.
        # Actually, for a convex container, the optimal position of the container relative
        # to the cluster is such that the cluster is "centered" in some sense.
        # However, if we allow the outer hexagon to move, we are effectively allowing
        # the inner positions to move relative to the outer center.
        # So fixing outer center at (0,0) and optimizing inner positions is equivalent
        # to optimizing both.
        
        # So, for a given inner configuration (relative to 0,0), we find min S over theta.
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
        Returns the side length of the minimal enclosing outer hexagon.
        """
        inner_positions = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_positions.append((x, y, a))
        
        # Check disjointness
        disjoint_ok = True
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
                    disjoint_ok = False
                    # Penalty: overlap depth? Just a large constant for now.
                    penalty += 10.0
        
        s, theta = get_min_outer_config(inner_positions)
        
        if not disjoint_ok:
            return s + penalty
        
        return s

    # Initial configuration: Hexagonal lattice
    # 1 center, 6 ring, 5 second ring
    init_params = np.zeros(3 * N)
    
    # Center
    init_params[0:3] = [0.0, 0.0, 0.0]
    
    # Ring 1
    for k in range(6):
        angle = np.radians(60 * k)
        idx = (k + 1) * 3
        init_params[idx:idx+3] = [2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0]
        
    # Ring 2 (5 hexagons)
    # Place them at 120, 180, 240, 300, 0? No, just pick 5 spots in the second ring.
    # Second ring radius is approx 3.46 (sqrt(3)*2) for touching?
    # Distance between centers in hex lattice:
    # (0,0) to (2,0) is 2.
    # (0,0) to (1, sqrt(3)) is 2.
    # So ring 1 is at distance 2.
    # Ring 2 hexagons are at distance 4? No.
    # In a hex grid, distance to 2nd neighbor is 2*sqrt(3) ~ 3.46?
    # Let's just place them at distance 3.5 and angles 30, 90, 150, 210, 270.
    for k in range(5):
        angle = np.radians(60 * k + 30)
        idx = (k + 7) * 3
        init_params[idx:idx+3] = [3.5 * np.cos(angle), 3.5 * np.sin(angle), 0.0]
        
    # One more to make 12? 
    # 1 + 6 + 5 = 12. Yes.
    
    # Simulated Annealing
    np.random.seed(42)
    
    current_params = init_params.copy()
    current_score = objective(current_params)
    
    best_params = current_params.copy()
    best_score = current_score
    
    T = 1.0  # Initial temperature
    cooling = 0.995
    
    max_iters = 5000
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Perturb
        new_params = current_params.copy()
        # Move 1 or 2 hexagons
        n_move = np.random.randint(1, 3)
        indices = np.random.choice(N, n_move, replace=False)
        
        for idx in indices:
            # Random walk in x, y, angle
            new_params[3*idx] += np.random.uniform(-0.5, 0.5)
            new_params[3*idx+1] += np.random.uniform(-0.5, 0.5)
            new_params[3*idx+2] += np.random.uniform(-30, 30)
            
        new_score = objective(new_params)
        
        # Acceptance criterion
        if new_score < current_score or np.random.rand() < np.exp(-(new_score - current_score) / T):
            current_params = new_params
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
                
        T *= cooling
        if T < 0.001:
            T = 0.001
            
    # Final refinement: try to center the configuration?
    # The objective function assumes outer center is (0,0).
    # If the cluster is not centered at (0,0), the enclosing hexagon might be larger than necessary
    # if we allowed the outer hexagon to shift.
    # However, shifting the outer hexagon is equivalent to shifting the inner hexagons.
    # Since we are optimizing inner hexagons in absolute coordinates, and the outer is fixed at (0,0),
    # the SA should naturally center the cluster around (0,0) to minimize the max projection.
    
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
    # Ensure all inner are inside
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
