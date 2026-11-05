import numpy as np
from scipy.optimize import minimize_scalar
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Define hexagon vertices for a unit hexagon centered at origin
    # Orientation: Pointy-right (vertex at (1,0))
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Container normals (for a hexagon with flat sides horizontal? No, normals at 0, 60, etc)
    # This corresponds to a container with vertices at 30, 90, etc?
    # If normals are at 0, 60, 120..., the sides are perpendicular to them.
    # Side 1 normal (1,0) -> Side is vertical line x=R.
    # This is a "pointy-top" container?
    # Let's just use these 6 normals.
    container_normals = np.array([
        [np.cos(k * np.pi / 3), np.sin(k * np.pi / 3)] for k in range(6)
    ])
    
    # Define candidate clusters in axial coordinates (q, r)
    # 7-flower base
    base_7 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Cluster A: Cross extension
    cluster_A = base_7 + [(2,0), (-2,0), (0,2), (0,-2)]
    
    # Cluster B: Top/Bottom extension
    cluster_B = base_7 + [(1,1), (-1,2), (-2,1), (2,-1)] # Just guessing some symmetric ones
    
    # Cluster C: Linear-ish
    cluster_C = base_7 + [(2,0), (1,-1), (2,-1), (3,-1)] # Unlikely to be better
    
    # We'll just use Cluster A and maybe a few variations if time permits.
    candidates = [cluster_A]
    
    best_side = float('inf')
    best_inner = None
    best_angle = 0.0
    
    for cluster in candidates:
        if time.time() > deadline:
            break
            
        # Convert axial to Cartesian
        centers = []
        for q, r in cluster:
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            centers.append([x, y])
        centers = np.array(centers)
        
        # Generate all vertices for the cluster
        # Each hexagon has 6 vertices
        # We will rotate the whole cluster, so we can just store centers and apply rotation to centers and verts
        # Actually, since all hexagons are parallel, we can just rotate the centers and the local verts.
        
        # Function to calculate min side length for a given rotation alpha
        def calc_side(alpha):
            # Rotate centers
            cos_a = np.cos(alpha)
            sin_a = np.sin(alpha)
            rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            rotated_centers = centers @ rot_mat.T
            
            # Max projection of any vertex onto any container normal
            max_proj = 0.0
            for rc in rotated_centers:
                # Vertices of this hexagon: rc + rotated(hex_verts)
                # Since hex_verts are rotated by the same alpha
                # local_verts = hex_verts @ rot_mat.T
                # global_verts = local_verts + rc
                # proj = global_verts @ normal
                # max_proj = max(max_proj, |proj|)
                
                local_verts = hex_verts @ rot_mat.T
                global_verts = local_verts + rc
                projs = np.abs(global_verts @ container_normals.T)
                current_max = np.max(projs)
                if current_max > max_proj:
                    max_proj = current_max
                    
            # Side length S from inradius R (max_proj)
            # R = S * sqrt(3) / 2
            S = 2.0 * max_proj / np.sqrt(3)
            return S
        
        # Optimize alpha
        # Due to symmetry, we only need to search [0, pi/6]
        result = minimize_scalar(calc_side, bounds=(0, np.pi/6), method='bounded', options={'xatol': 1e-6})
        
        if result.success:
            best_alpha = result.x
            best_S_for_cluster = result.fun
            if best_S_for_cluster < best_side:
                best_side = best_S_for_cluster
                best_angle = best_alpha
                # Prepare inner list
                cos_a = np.cos(best_alpha)
                sin_a = np.sin(best_alpha)
                rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                rotated_centers = centers @ rot_mat.T
                inner = []
                for rc in rotated_centers:
                    inner.append([rc[0], rc[1], np.degrees(best_alpha)])
                best_inner = inner
                
    # If no optimization succeeded or time ran out, return a default
    if best_inner is None:
        # Fallback to seed-like behavior or simple grid
        # Just return the unrotated cluster A with estimated side
        centers = []
        for q, r in cluster_A:
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            centers.append([x, y])
        centers = np.array(centers)
        # Estimate side
        max_x = np.max(centers[:, 0]) + 1
        max_y = np.max(centers[:, 1]) + 1
        # Rough estimate
        S_est = 2 * max(max_x, max_y) / np.sqrt(3)
        inner = [[c[0], c[1], 0.0] for c in centers]
        return inner, (0.0, 0.0), S_est, 0.0
        
    return best_inner, (0.0, 0.0), best_side, 0.0

# EVOLVE-BLOCK-START
# The code above is the implementation
# EVOLVE-BLOCK-END
