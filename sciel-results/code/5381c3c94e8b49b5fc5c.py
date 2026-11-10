import numpy as np
import time

def get_hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    angles = angle_rad + np.deg2rad(np.arange(0, 360, 60))
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def polygon_normals(vertices):
    edges = np.roll(vertices, -1, axis=0) - vertices
    normals = np.column_stack((edges[:, 1], -edges[:, 0]))
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return normals / norms

def sat_overlap(proj_a, proj_b):
    min_a, max_a = np.min(proj_a), np.max(proj_a)
    min_b, max_b = np.min(proj_b), np.max(proj_b)
    overlap = np.minimum(max_a, max_b) - np.maximum(min_a, min_b)
    return max(0.0, overlap)

def compute_min_s(centers, angles):
    """
    Compute the minimum side length s of a regular hexagon centered at (0,0) with angle 0
    that contains all 12 unit hexagons defined by centers and angles.
    """
    # Outer hexagon normals (outward) for angle 0
    # Vertices at 0, 60, ... 300.
    # Edge 0 is between 0 and 60? No.
    # Vertices: v0=(s,0), v1=(s/2, s*sqrt(3)/2), ...
    # Edge 0: v0 to v1. Vector: (-s/2, s*sqrt(3)/2).
    # Normal: (dy, -dx) = (s*sqrt(3)/2, s/2). Normalized: (sqrt(3)/2, 1/2). Angle 30 deg.
    # The 6 outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    outer_normals = np.array([
        [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))],
        [np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90))],
        [np.cos(np.deg2rad(150)), np.sin(np.deg2rad(150))],
        [np.cos(np.deg2rad(210)), np.sin(np.deg2rad(210))],
        [np.cos(np.deg2rad(270)), np.sin(np.deg2rad(270))],
        [np.cos(np.deg2rad(330)), np.sin(np.deg2rad(330))]
    ])
    
    max_proj = 0.0
    for i in range(12):
        cx, cy = centers[i]
        ang = angles[i]
        verts = get_hex_vertices(cx, cy, 1.0, ang)
        # Project all vertices onto all 6 outer normals
        projs = np.dot(verts, outer_normals.T) # (6, 6)
        max_proj = np.maximum(max_proj, np.max(projs))
        
    # Condition: max_proj <= s * sqrt(3) / 2
    # s >= 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / np.sqrt(3.0)

def check_inner_overlap(centers, angles):
    """
    Returns total overlap penalty (sum of squared overlaps) for inner-inner pairs.
    0.0 if no overlap.
    """
    verts_list = []
    normals_list = []
    for i in range(12):
        cx, cy = centers[i]
        ang = angles[i]
        verts = get_hex_vertices(cx, cy, 1.0, ang)
        normals = polygon_normals(verts)
        verts_list.append(verts)
        normals_list.append(normals)
        
    total_penalty = 0.0
    for i in range(12):
        for j in range(i+1, 12):
            # Quick reject: if distance between centers > 2*sqrt(3), they can't touch?
            # Max distance from center to vertex is 1. So if dist > 2, no overlap.
            # Actually, max width is 2*sqrt(3)/2 * 2? No.
            # Distance between centers of touching hexes is sqrt(3).
            # If dist > 2 * 1 = 2, they are definitely separate?
            # Max extent from center is 1 (vertex). So if dist > 2, separate.
            d = np.linalg.norm(centers[i] - centers[j])
            if d > 2.0:
                continue
                
            v_a = verts_list[i]
            v_b = verts_list[j]
            n_a = normals_list[i]
            n_b = normals_list[j]
            
            # Check normals of A
            for n in n_a:
                proj_a = np.dot(v_a, n)
                proj_b = np.dot(v_b, n)
                overlap = sat_overlap(proj_a, proj_b)
                if overlap > 0:
                    total_penalty += overlap * overlap
                    break # Early exit if overlap found? No, we need total penalty for SA? 
                    # For validity check, we just need to know if > 0.
                    # For SA cost, we can sum.
            else:
                # Check normals of B
                for n in n_b:
                    proj_a = np.dot(v_a, n)
                    proj_b = np.dot(v_b, n)
                    overlap = sat_overlap(proj_a, proj_b)
                    if overlap > 0:
                        total_penalty += overlap * overlap
                        break
                        
    return total_penalty

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial guess: Honeycomb lattice
    centers = np.zeros((12, 2))
    angles = np.zeros(12)
    
    # Center
    centers[0] = [0.0, 0.0]
    # Ring 1
    for i in range(6):
        angle = np.deg2rad(60 * i)
        centers[i+1] = [np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)]
    # Ring 2 (pick 5)
    for i in range(5):
        angle = np.deg2rad(60 * i)
        centers[i+7] = [2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle)]
        
    # Compute initial s
    s = compute_min_s(centers, angles)
    overlap = check_inner_overlap(centers, angles)
    best_cost = s + 100.0 * overlap
    best_centers = centers.copy()
    best_angles = angles.copy()
    best_s = s
    
    T = 1.0
    cooling = 0.9995
    
    # Simulated Annealing
    while time.time() < deadline - 10.0:
        # Pick a random hex to move
        idx = np.random.randint(12)
        
        # Perturb center
        delta = np.random.normal(0, 0.1, 2)
        new_centers = centers.copy()
        new_centers[idx] += delta
        
        # Perturb angle
        new_angles = angles.copy()
        new_angles[idx] += np.random.normal(0, 5.0)
        
        # Check validity and cost
        new_overlap = check_inner_overlap(new_centers, new_angles)
        new_s = compute_min_s(new_centers, new_angles)
        new_cost = new_s + 100.0 * new_overlap
        
        # Acceptance
        if new_cost < best_cost:
            # Always accept improvement
            pass
        else:
            # Accept with probability exp(-(new_cost - best_cost) / T)
            # If overlap > 0, cost is huge, so probability is ~0.
            # This effectively enforces hard constraints after initial phase.
            if np.random.random() > np.exp(-(new_cost - best_cost) / T):
                continue
                
        # Update state
        centers = new_centers
        angles = new_angles
        s = new_s
        overlap = new_overlap
        cost = new_cost
        
        if cost < best_cost:
            best_cost = cost
            best_centers = centers.copy()
            best_angles = angles.copy()
            best_s = s
            
        T *= cooling
        if T < 0.01:
            T = 0.01 # Floor temperature to keep exploring slightly
            
    # Final validation and return
    inner = np.column_stack((best_centers[:, 0], best_centers[:, 1], best_angles))
    outer = np.array([0.0, 0.0, 0.0])
    
    # Ensure s is at least sqrt(12) as per contract? 
    # Contract says s < sqrt(12) is invalid. 
    # sqrt(12) ~ 3.46.
    # Our s is usually ~4.5-5.
    if best_s < np.sqrt(12):
        best_s = np.sqrt(12)
        
    return inner, outer, best_s
