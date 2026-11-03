import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    def lattice_to_cart(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    core_coords = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Candidate shell sets
    shell_candidates = [
        [(2,0), (0,2), (-2,0), (0,-2)], # Cross
        [(2,0), (1,1), (0,2), (-1,1)], # Clump
        [(2,0), (-2,0), (1,1), (-1,-1)], # Pairs
        [(2,0), (0,2), (-1,1), (1,-1)] # Mix
    ]
    
    hex_verts = np.column_stack((np.cos(np.deg2rad(np.arange(0, 360, 60))), 
                                 np.sin(np.deg2rad(np.arange(0, 360, 60)))))
    
    container_normals = np.column_stack((np.cos(np.deg2rad(30 + np.arange(0, 360, 60))), 
                                         np.sin(np.deg2rad(30 + np.arange(0, 360, 60)))))
    
    best_S = float('inf')
    best_angle = 0.0
    best_centers = None
    
    for shell in shell_candidates:
        all_coords = core_coords + shell
        centers = np.array([lattice_to_cart(q, r) for q, r in all_coords])
        
        for angle_deg in np.arange(0, 60, 0.5):
            angle_rad = np.deg2rad(angle_deg)
            R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
            rotated_centers = centers @ R.T
            rotated_hex_verts = hex_verts @ R.T
            
            max_proj = 0.0
            for c in rotated_centers:
                verts = c + rotated_hex_verts
                projs = container_normals @ verts.T
                current_max = np.max(np.abs(projs))
                if current_max > max_proj:
                    max_proj = current_max
            
            inradius_req = max_proj
            S_req = inradius_req * 2.0 / np.sqrt(3)
            
            if S_req < best_S:
                best_S = S_req
                best_angle = angle_deg
                best_centers = rotated_centers

    inner = []
    for c in best_centers:
        inner.append([c[0], c[1], best_angle])
        
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
