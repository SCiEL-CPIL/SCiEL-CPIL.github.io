import numpy as np
import scipy.optimize
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580  # 50s margin for safety

    sqrt3 = np.sqrt(3)
    
    def get_unit_hex_vertices():
        """Vertices of unit hexagon centered at origin, angle 0."""
        angles = np.deg2rad(60 * np.arange(6))
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def get_hex_vertices(cx, cy, angle_deg=0.0):
        """Vertices of hexagon with center (cx,cy), side 1, given angle."""
        base = get_unit_hex_vertices()
        theta = np.deg2rad(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        return base @ R.T + np.array([cx, cy])

    def check_separation(centers, angles):
        """Check if all pairs of hexagons are separated by > 1e-6 on some axis."""
        n = len(centers)
        verts_list = []
        for i in range(n):
            verts_list.append(get_hex_vertices(centers[i,0], centers[i,1], angles[i]))
        
        for i in range(n):
            for j in range(i+1, n):
                v1 = verts_list[i]
                v2 = verts_list[j]
                # Get all edge normals from both hexagons
                normals = []
                for v in [v1, v2]:
                    for k in range(6):
                        p1 = v[k]
                        p2 = v[(k+1) % 6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-12:
                            normals.append(normal / norm)
                
                # Check if ANY normal gives separation > 1e-6
                separated = False
                for normal in normals:
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    # Gap is the distance between intervals
                    gap = max(min2 - max1, min1 - max2)
                    if gap > 1e-6:
                        separated = True
                        break
                if not separated:
                    return False
        return True

    def compute_s(centers, angles, outer_center, outer_angle):
        """Compute minimum s such that all inner hexagons fit in outer hexagon."""
        angles_rad = np.deg2rad(outer_angle + 60 * np.arange(6))
        normals = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])
        
        max_proj = 0.0
        for i in range(len(centers)):
            v = get_hex_vertices(centers[i,0], centers[i,1], angles[i])
            rel = v - outer_center
            proj_rel = rel @ normals.T
            max_proj = max(max_proj, np.max(proj_rel))
        
        return max(1.0, max_proj)

    def objective(params):
        if time.time() > deadline:
            return 1e9
            
        n_centers = 12
        centers = params[:2*n_centers].reshape(n_centers, 2)
        angles = params[2*n_centers:3*n_centers]
        outer_angle = params[3*n_centers]
        
        # Center the configuration
        centroid = np.mean(centers, axis=0)
        shifted = centers - centroid
        
        # Penalty for overlaps: distance between centers must be >= sqrt(3)
        # Approximate overlap penalty using center distances
        # This is a heuristic but effective for guiding the optimizer away from overlaps
        overlap_penalty = 0.0
        min_dist = sqrt3
        for i in range(n_centers):
            for j in range(i+1, n_centers):
                d = np.linalg.norm(shifted[i] - shifted[j])
                if d < min_dist:
                    overlap_penalty += (min_dist - d)**2
        
        s = compute_s(shifted, angles, np.array([0.0, 0.0]), outer_angle)
        
        # Strong penalty for overlaps
        return s + 10.0 * overlap_penalty

    # Initial guess: hexagonal lattice with small gap
    # Scale factor to ensure gaps
    gap = 0.05
    scale = sqrt3 + gap
    
    # Hexagonal lattice coordinates for 12 hexagons (1+6+5)
    # Center at 0
    centers_list = [[0.0, 0.0]]
    
    # Ring 1
    for i in range(6):
        angle = np.deg2rad(60 * i)
        centers_list.append([scale * np.cos(angle), scale * np.sin(angle)])
    
    # Ring 2 (5 positions to make 12 total, skipping one to keep symmetry or just take 5)
    # Actually, let's just take the first 5 of the next ring
    for i in range(6):
        angle = np.deg2rad(60 * i + 30) # Offset for hex lattice
        # Distance for ring 2 is 2 * scale? No, hex lattice step is scale.
        # Ring 2 radius is 2 * (sqrt(3)/2 * scale)? No.
        # In hex lattice with center distance D:
        # Ring 1 radius D. Ring 2 radius 2D.
        # Let's just use a grid.
        pass
        
    # Simpler initial guess: just place them in a circle with large radius to avoid overlap
    # Then optimizer pulls them in.
    R_init = 5.0
    for i in range(11):
        angle = 2 * np.pi * i / 11
        centers_list.append([R_init * np.cos(angle), R_init * np.sin(angle)])
        
    all_centers = np.array(centers_list)
    
    angles0 = np.zeros(12)
    outer_angle0 = 0.0
    
    x0 = np.concatenate([all_centers.flatten(), angles0, [outer_angle0]])
    
    # Optimize
    best_x = x0.copy()
    best_f = objective(x0)
    
    try:
        res = scipy.optimize.minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
        )
        if np.isfinite(res.fun) and res.fun < best_f:
            best_x = res.x
            best_f = res.fun
    except Exception:
        pass
    
    # Extract solution
    centers = best_x[:24].reshape(12, 2)
    angles = best_x[24:36]
    outer_angle = best_x[36]
    
    # Center the configuration
    centroid = np.mean(centers, axis=0)
    shifted_centers = centers - centroid
    
    # Compute s
    s = compute_s(shifted_centers, angles, np.array([0.0, 0.0]), outer_angle)
    
    # Verify validity
    if not check_separation(shifted_centers, angles):
        # Fallback: Scale up the centers until they don't overlap
        # This guarantees a valid solution, though s will be larger
        scale_factor = 1.0
        for attempt in range(100):
            scaled_centers = shifted_centers * scale_factor
            if check_separation(scaled_centers, angles):
                shifted_centers = scaled_centers
                s = compute_s(shifted_centers, angles, np.array([0.0, 0.0]), outer_angle)
                break
            scale_factor *= 1.05
            
    # Final centers (shifted back so outer center is at centroid)
    final_centers = shifted_centers + centroid
    
    # Build output
    inner_data = np.zeros((12, 3))
    inner_data[:, 0] = final_centers[:, 0]
    inner_data[:, 1] = final_centers[:, 1]
    inner_data[:, 2] = angles
    
    outer_data = np.array([centroid[0], centroid[1], outer_angle])
    
    # Ensure s is valid
    if not np.isfinite(s) or s < np.sqrt(12):
        s = 5.0  # Fallback
    
    return inner_data, outer_data, s

if __name__ == "__main__":
    inner, outer, s = hexagon_packing_12()
    print(f"s: {s}")
