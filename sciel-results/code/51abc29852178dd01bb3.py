import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 11
D = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _outer_hex_normals(angle_deg):
    """
    Return the 6 outward unit normals of a regular hexagon with the given rotation.
    """
    base = np.deg2rad(angle_deg + 30)
    normals = []
    for k in range(6):
        theta = base + k * np.pi / 3
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    return np.array(normals)

def _min_enclosing_side(centers, inner_angles, outer_angle_deg):
    """
    Compute the minimum side length of the outer hexagon (centered at origin) 
    such that all inner hexagons are contained.
    """
    normals = _outer_hex_normals(outer_angle_deg)
    max_proj = 0.0
    for i in range(len(centers)):
        cx, cy = centers[i]
        a = inner_angles[i]
        verts = _hex_vertices(cx, cy, a)
        # Project all vertices onto all normals
        dots = verts @ normals.T
        abs_dots = np.abs(dots)
        m = np.max(abs_dots)
        if m > max_proj:
            max_proj = m
    
    return 2.0 * max_proj / np.sqrt(3)

def _check_no_overlap(centers, min_dist=np.sqrt(3) - 1e-9):
    """Check that no two inner hexagons overlap (center distance >= sqrt(3))."""
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                return False
    return True

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    
    # Core 7 hexagons: 1 center, 6 at distance sqrt(3)
    # Lattice vectors: v1=(sqrt(3), 0), v2=(sqrt(3)/2, 3/2)
    # First ring centers:
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) in lattice coords
    # Let's use explicit coords for the 7 core
    core_centers = [
        [0.0, 0.0],
        [np.sqrt(3), 0.0],
        [np.sqrt(3)/2, 1.5],
        [-np.sqrt(3)/2, 1.5],
        [-np.sqrt(3), 0.0],
        [-np.sqrt(3)/2, -1.5],
        [np.sqrt(3)/2, -1.5]
    ]
    
    # Second ring centers (distance 3):
    # (1,1), (0,2), (-1,2), (-2,1), (-1,-1), (1,-2) in lattice coords?
    # Let's use the angles derived: 30, 90, 150, 210, 270, 330 degrees at radius 3.
    # Wait, the lattice vectors gave:
    # (1,1) -> 30 deg
    # (-1,2) -> 90 deg
    # (-2,1) -> 150 deg
    # (-1,-1) -> 210 deg
    # (1,-2) -> 270 deg
    # (2,-1) -> 330 deg
    second_ring_centers = []
    for k in range(6):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0
        second_ring_centers.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Generate 3 distinct topologies of choosing 4 from 6
    # Indices 0..5
    topologies = [
        [0, 1, 2, 3],  # Missing 4,5 (adjacent)
        [0, 1, 2, 4],  # Missing 3,5 (separated by 1? 3 and 5 have 4 between? No. 3,5 diff 2. 0,1,2,4. Missing 3,5. Gap 4. So 1 gap between missing? No. 3 and 5 are separated by index 4. So yes.)
        [0, 1, 3, 4]   # Missing 2,5. Diff 3. Opposite? 0 and 3 are opposite. 2 and 5 are opposite.
                       # Wait, 0,1,3,4. Missing 2,5.
                       # 2 and 5 are opposite.
    ]
    
    # Actually, let's just iterate all 15 subsets to be safe, it's cheap.
    import itertools
    all_subsets = list(itertools.combinations(range(6), 4))
    
    best_side = float('inf')
    best_centers = None
    best_inner_angles = None
    best_outer_angle = 0.0
    
    # Inner angles: try 0 and 30 degrees relative to the cluster axes
    # The cluster axes are defined by the lattice.
    # If we define the lattice with v1 along X, the hexagons at (1,0) etc.
    # A hexagon at (1,0) with angle 0 has vertices at 0, 60...
    # The "flat" side is perpendicular to X?
    # Let's just try 0 and 30.
    
    for subset in all_subsets:
        if time.time() > DEADLINE - safety_margin:
            break
            
        # Build centers for this subset
        centers_list = core_centers.copy()
        for idx in subset:
            centers_list.append(second_ring_centers[idx])
        
        centers = np.array(centers_list)
        
        # Check overlap (should be fine for lattice points, but just in case)
        if not _check_no_overlap(centers):
            continue
            
        for inner_angle in [0.0, 30.0]:
            if time.time() > DEADLINE - safety_margin:
                break
                
            inner_angles = np.array([inner_angle] * 11)
            
            # Optimize rotation of the cluster
            # We rotate the centers and the inner angles
            def objective(phi_deg):
                if time.time() > DEADLINE - 2:
                    return 1e9
                phi_rad = np.deg2rad(phi_deg)
                rot_matrix = np.array([
                    [np.cos(phi_rad), -np.sin(phi_rad)],
                    [np.sin(phi_rad), np.cos(phi_rad)]
                ])
                rotated_centers = centers @ rot_matrix.T
                rotated_angles = inner_angles + phi_deg
                
                return _min_enclosing_side(rotated_centers, rotated_angles, 0.0)
            
            try:
                res = minimize_scalar(objective, bounds=(0, 60), method='bounded',
                                       options={'xatol': 1e-8})
                if res.fun < best_side:
                    best_side = res.fun
                    phi_rad = np.deg2rad(res.x)
                    rot_matrix = np.array([
                        [np.cos(phi_rad), -np.sin(phi_rad)],
                        [np.sin(phi_rad), np.cos(phi_rad)]
                    ])
                    best_centers = centers @ rot_matrix.T
                    best_inner_angles = inner_angles + res.x
                    best_outer_angle = 0.0
            except Exception:
                pass
    
    # Fallback if no valid result found
    if best_centers is None:
        best_centers = np.array(core_centers[:7] + second_ring_centers[:4])
        best_inner_angles = np.array([30.0] * 11)
        best_side = _min_enclosing_side(best_centers, best_inner_angles, 30.0)
        best_outer_angle = 30.0
    
    # Final verification
    if not _check_no_overlap(best_centers):
        # Fallback to a known good config (seed)
        best_centers, best_inner_angles = _seed_configuration()
        best_side = _min_enclosing_side(best_centers, best_inner_angles, 30.0)
        best_outer_angle = 30.0
    
    # Build the inner list: [x, y, angle_degrees]
    inner = []
    for i in range(11):
        inner.append([best_centers[i, 0], best_centers[i, 1], best_inner_angles[i]])
    
    return inner, (0.0, 0.0), best_side, best_outer_angle

def _seed_configuration():
    """
    Return the standard concentric ring configuration.
    """
    eps = 1e-10
    centers = []
    angles = []
    
    centers.append([0.0, 0.0])
    angles.append(30.0)
    
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        centers.append([r * np.cos(angle), r * np.sin(angle)])
        angles.append(30.0)
    
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + eps
        centers.append([r * np.cos(angle), r * np.sin(angle)])
        angles.append(30.0)
    
    return np.array(centers), np.array(angles)
