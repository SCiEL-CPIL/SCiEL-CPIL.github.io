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

    def get_all_verts(positions, angles):
        """
        Vectorized computation of all hexagon vertices.
        positions: (N, 2)
        angles: (N,)
        Returns: (N, 6, 2)
        """
        n = positions.shape[0]
        angles_rad = np.radians(angles)
        cos_a = np.cos(angles_rad)
        sin_a = np.sin(angles_rad)
        
        # Rotation matrices: (N, 2, 2)
        rot = np.zeros((n, 2, 2))
        rot[:, 0, 0] = cos_a
        rot[:, 0, 1] = -sin_a
        rot[:, 1, 0] = sin_a
        rot[:, 1, 1] = cos_a
        
        # Rotate base vertices: (6, 2) @ (N, 2, 2).T -> (6, N, 2) -> transpose to (N, 6, 2)
        # Actually: unit_verts_0 (6,2). We want to apply rot (N,2,2) to each.
        # verts[i] = unit_verts_0 @ rot[i].T
        # Using einsum: 'ij,klj->kli' ? No.
        # Let's do: (N, 6, 2) = (N, 2, 2) @ (2, 6).T ?
        # rot @ unit_verts_0.T gives (N, 2, 6). Transpose to (N, 6, 2).
        rotated = np.einsum('nij,jk->nik', rot, unit_verts_0.T) # (N, 2, 6)
        rotated = np.transpose(rotated, (0, 2, 1)) # (N, 6, 2)
        
        # Add centers
        rotated += positions[:, np.newaxis, :]
        
        return rotated

    def calc_min_outer_side_vectorized(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals for the outer hexagon sides
        # Distance from center to side (apothem) is S * sqrt(3) / 2.
        # Normals are at angles theta + 30 + k*60.
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        # Get all vertices (N, 6, 2)
        verts = get_all_verts(positions, angles)
        
        # Project all vertices onto all normals
        # verts: (N, 6, 2)
        # normals: (6, 2)
        # We want max projection for each normal.
        # proj[i, k, j] = dot(verts[i, j], normals[k])
        # Using einsum: 'ijl,kl->ijk' where l is dim 2
        projs = np.einsum('ijl,kl->ijk', verts, normals) # (N, 6, 6)
        
        # Max projection per normal (across all hexagons and vertices)
        # We need max over i (hexagon) and j (vertex) for each k (normal)
        max_projs = np.max(projs, axis=(0, 1)) # (6,)
        
        # The side length S is such that apothem = S * sqrt(3) / 2
        # The projection distance from center is the apothem.
        # So S = max_proj * 2 / sqrt(3)
        return np.max(max_projs) * 2.0 / SQRT3

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
            side = calc_min_outer_side_vectorized(positions, angles, test_angle)
            if side < best_side:
                best_side = side
                best_angle = test_angle
                
        return best_side, best_angle

    def sat_gap_fast(poly1, poly2):
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
                if sat_gap_fast(polys[i], polys[j]) < -tol:
                    return False
        return True

    def calc_overlap_penalty(positions, angles):
        """
        Calculate a smooth penalty for overlaps.
        """
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap_fast(polys[i], polys[j])
                if gap < 0:
                    penalty += gap**2
        return penalty

    # --- PHASE 1: DISCRETE GRID SEARCH FOR TOPOLOGY ---
    
    sqrt3 = math.sqrt(3)
    
    best_valid_config = None
    best_valid_score = float('inf')
    
    # Lattice vectors for aligned hexagons
    base_v1 = np.array([1.5, sqrt3/2])
    base_v2 = np.array([0, sqrt3])
    
    # Try different lattice scales
    for scale_x in [0.92, 0.96, 1.0, 1.04, 1.08]:
        for scale_y in [0.92, 0.96, 1.0, 1.04, 1.08]:
            v1 = base_v1 * np.array([scale_x, 1.0])
            v2 = base_v2 * np.array([1.0, scale_y])
            
            points = []
            for i in range(-4, 5):
                for j in range(-4, 5):
                    p = i * v1 + j * v2
                    points.append(p)
            
            points.sort(key=lambda p: np.dot(p, p))
            selected = np.array(points[:12])
            
            centroid = np.mean(selected, axis=0)
            selected -= centroid
            
            for rot_deg in [0, 15, 30, 45, 60, 75, 90]:
                rot_rad = math.radians(rot_deg)
                rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)], 
                                    [math.sin(rot_rad), math.cos(rot_rad)]])
                
                rotated_selected = selected @ rot_mat.T
                rotated_angles = np.zeros(N)
                
                if is_valid(rotated_selected, rotated_angles):
                    side, _ = find_best_outer_angle(rotated_selected, rotated_angles, coarse=True)
                    if side < best_valid_score:
                        best_valid_score = side
                        best_valid_config = (rotated_selected, rotated_angles)
                        
    if best_valid_config is None:
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
    
    def objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        side, _ = find_best_outer_angle(positions, angles, coarse=True)
        
        penalty = calc_overlap_penalty(positions, angles)
        return side + 10.0 * penalty

    def local_objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        side, _ = find_best_outer_angle(positions, angles, coarse=False)
        
        penalty = calc_overlap_penalty(positions, angles)
        return side + 100.0 * penalty

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
                pos = res2.x[:2*N].reshape(N, 2)
                ang = res2.x[2*N:]
                if is_valid(pos, ang, tol=1e-6):
                    best_valid_score = res2.fun
                    init_params = res2.x
        except Exception:
            pass
    
    final_pos = init_params[:2*N].reshape(N, 2)
    final_ang = init_params[2*N:]
    
    final_side, final_outer_angle = find_best_outer_angle(final_pos, final_ang, coarse=False)
    
    inner = []
    for i in range(N):
        inner.append([final_pos[i][0], final_pos[i][1], final_ang[i]])
    
    return inner, (0.0, 0.0), final_side, final_outer_angle

if __name__ == "__main__":
    construct()
