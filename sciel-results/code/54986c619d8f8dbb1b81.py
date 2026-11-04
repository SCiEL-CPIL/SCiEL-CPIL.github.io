import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0 (flat-top)
    # Vertices at 0, 60, ..., 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals for the outer hexagon sides
        # For a regular hexagon, the distance from center to side (apothem) is S * sqrt(3) / 2.
        # The normals to the sides are at angles theta + 30 + k*60.
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / SQRT3

    def find_best_outer_angle(positions, angles, coarse=False):
        """
        Find the optimal rotation of the outer hexagon that minimizes its side length.
        """
        best_side = float('inf')
        best_angle = 0.0
        
        if coarse:
            test_angles = np.linspace(0, 60, 61)
        else:
            test_angles = np.linspace(0, 60, 361)

        for test_angle in test_angles:
            side = calc_min_outer_side(positions, angles, test_angle)
            if side < best_side:
                best_side = side
                best_angle = test_angle
                
        return best_side, best_angle

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value (approximate penetration depth).
        If they are separated, returns the positive distance.
        """
        min_gap = float('inf')
        
        # Axes from poly1 edges
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap on this axis
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        # Axes from poly2 edges
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        return min_gap

    def is_valid(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    # --- PHASE 1: DISCRETE GRID SEARCH FOR TOPOLOGY ---
    # Instead of optimizing continuous coordinates from a random start, we search
    # a lattice of potential centers and a set of discrete angles to find a valid
    # configuration. This avoids getting stuck in local minima of the penalty method.
    
    sqrt3 = math.sqrt(3)
    
    # Generate candidate centers in a hexagonal lattice pattern
    # The ideal packing for hexagons is on a triangular lattice.
    # Lattice vectors: a1 = (1.5, sqrt(3)/2), a2 = (0, sqrt(3))
    # We search a range of offsets and scales.
    
    best_valid_config = None
    best_valid_score = float('inf')
    
    # Search space for lattice scaling and offset
    # A unit hexagon has width 2. To pack them, centers should be at least ~2 apart in some directions.
    # Let's try lattice spacings that allow tight packing.
    
    # Pre-computed discrete angles (multiples of 30 degrees are often optimal for hexagons)
    discrete_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    # We will search for configurations on a lattice defined by:
    # P_i = origin + i * v1 + j * v2
    # where v1, v2 are lattice vectors.
    
    # Common tight packings use specific lattice vectors.
    # For hexagons, the "standard" dense packing has centers at distance 2*sqrt(3)/2 * 2? 
    # Distance between centers in dense hex packing is 2 * (inradius) + 2 * (inradius) = 2 * sqrt(3) * 1 = 2*sqrt(3) ~ 3.46?
    # Wait, inradius of unit hex is sqrt(3)/2. Distance between parallel sides is sqrt(3).
    # If aligned, centers need to be sqrt(3) apart in y and 1.5 in x?
    # Lattice vectors for aligned hexagons:
    # v1 = (1.5, sqrt(3)/2), v2 = (0, sqrt(3))
    # This creates a grid where hexagons touch.
    
    base_v1 = np.array([1.5, sqrt3/2])
    base_v2 = np.array([0, sqrt3])
    
    # We will vary the scale of this lattice and the rotation of the whole cluster.
    # Also allow small perturbations to break symmetry if needed? 
    # For now, stick to exact lattice points to ensure validity, then refine.
    
    candidates = []
    
    # Try different lattice scales (stretching/compressing)
    for scale_x in [0.92, 0.96, 1.0, 1.04, 1.08]:
        for scale_y in [0.92, 0.96, 1.0, 1.04, 1.08]:
            v1 = base_v1 * np.array([scale_x, 1.0])
            v2 = base_v2 * np.array([1.0, scale_y])
            
            # Generate a large enough grid to pick 12 points
            # We need to pick 12 points from this lattice that form a compact cluster.
            # The "best" 12 points are usually the ones closest to the centroid.
            
            points = []
            for i in range(-4, 5):
                for j in range(-4, 5):
                    p = i * v1 + j * v2
                    points.append(p)
            
            # Sort by distance from origin
            points.sort(key=lambda p: np.dot(p, p))
            
            # Take the first 12
            selected = np.array(points[:12])
            
            # Center the cluster
            centroid = np.mean(selected, axis=0)
            selected -= centroid
            
            # Try different global rotations for this cluster
            for rot_deg in [0, 15, 30, 45, 60, 75, 90]:
                rot_rad = math.radians(rot_deg)
                rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)], 
                                    [math.sin(rot_rad), math.cos(rot_rad)]])
                
                rotated_selected = selected @ rot_mat.T
                rotated_angles = np.zeros(N) # Inner hexagons keep orientation relative to global? 
                # Wait, if we rotate the lattice, do we rotate the hexagons too?
                # Yes, rigid body assembly.
                
                # Check validity
                if is_valid(rotated_selected, rotated_angles):
                    side, _ = find_best_outer_angle(rotated_selected, rotated_angles, coarse=True)
                    if side < best_valid_score:
                        best_valid_score = side
                        best_valid_config = (rotated_selected, rotated_angles)
                        
    if best_valid_config is None:
        # Fallback to standard honeycomb if no valid scaled lattice found
        v1 = base_v1
        v2 = base_v2
        points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                p = i * v1 + j * v2
                points.append(p)
        points.sort(key=lambda p: np.dot(p, p))
        selected = np.array(points[:12])
        centroid = np.mean(selected, axis=0)
        selected -= centroid
        best_valid_config = (selected, np.zeros(N))
        best_valid_score = find_best_outer_angle(selected, np.zeros(N), coarse=True)[0]

    # --- PHASE 2: CONTINUOUS REFINEMENT ---
    # Use the best discrete configuration as a starting point for local optimization.
    # We optimize positions and angles simultaneously to squeeze out more space.
    
    def objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Calculate side length with optimal outer angle (coarse for speed)
        side, _ = find_best_outer_angle(positions, angles, coarse=True)
        
        # Check validity
        if not is_valid(positions, angles, tol=1e-4):
            # Penalty for overlap
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    gap = sat_gap(polys[i], polys[j])
                    if gap < 0:
                        penalty += gap**2
            return side + 10.0 * penalty
            
        return side

    def local_objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Finer outer angle search for local refinement
        side, _ = find_best_outer_angle(positions, angles, coarse=False)
        
        if not is_valid(positions, angles, tol=1e-6):
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    gap = sat_gap(polys[i], polys[j])
                    if gap < 0:
                        penalty += gap**2
            return side + 100.0 * penalty
            
        return side

    init_pos, init_ang = best_valid_config
    init_params = np.concatenate([init_pos.flatten(), init_ang])
    
    # Run Nelder-Mead
    try:
        res = minimize(
            objective,
            init_params,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6}
        )
        if res.fun < best_valid_score:
            best_valid_score = res.fun
            init_params = res.x
    except Exception:
        pass
    
    # Polish with tighter tolerance
    if time.time() < deadline - 10:
        try:
            res2 = minimize(
                local_objective,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res2.fun < best_valid_score:
                # Verify validity before accepting
                pos = res2.x[:2*N].reshape(N, 2)
                ang = res2.x[2*N:]
                if is_valid(pos, ang, tol=1e-6):
                    best_valid_score = res2.fun
                    init_params = res2.x
        except Exception:
            pass
    
    final_pos = init_params[:2*N].reshape(N, 2)
    final_ang = init_params[2*N:]
    
    # Final precise calculation of outer side and angle
    final_side, final_outer_angle = find_best_outer_angle(final_pos, final_ang, coarse=False)
    
    inner = []
    for i in range(N):
        inner.append([final_pos[i][0], final_pos[i][1], final_ang[i]])
    
    return inner, (0.0, 0.0), final_side, final_outer_angle

if __name__ == "__main__":
    construct()
