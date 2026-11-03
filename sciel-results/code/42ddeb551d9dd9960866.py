import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s limit
    
    # Generate vertices of a unit regular hexagon (flat-top)
    hex_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    unit_hex_verts = np.stack([np.cos(hex_angles), np.sin(hex_angles)], axis=1)
    
    def get_hex_vertices(center, angle_deg):
        """Get vertices of a unit hexagon at given center and rotation."""
        theta = np.radians(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        return center + unit_hex_verts @ R.T
    
    def container_side_length(positions, angles, outer_center, outer_angle):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        outer_theta = np.radians(outer_angle)
        R_outer = np.array([[np.cos(outer_theta), -np.sin(outer_theta)],
                            [np.sin(outer_theta), np.cos(outer_theta)]])
        
        # Face normals of the outer hexagon (pointing inward)
        # For a regular hexagon, face normals are at angles 30, 90, 150, 210, 270, 330
        # relative to the hexagon's orientation
        face_normal_angles = (30 + np.arange(6) * 60) * np.pi / 180
        face_normals = np.stack([np.cos(face_normal_angles), np.sin(face_normal_angles)], axis=1)
        # Rotate face normals by outer_angle
        face_normals = face_normals @ R_outer.T
        
        max_proj = 0.0
        for i in range(N):
            verts = get_hex_vertices(positions[i], angles[i])
            # Translate to outer center frame
            verts_rel = verts - outer_center
            # Project onto each face normal
            for fn in face_normals:
                proj = np.max(verts_rel @ fn)
                if proj > max_proj:
                    max_proj = proj
        
        # Inradius = s * sqrt(3) / 2, so s = 2 * inradius / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3.0)
    
    def soft_max(values, temperature=0.1):
        """Smooth approximation of max using log-sum-exp."""
        shifted = values - np.max(values)
        return np.max(values) + temperature * np.log(np.sum(np.exp(shifted / temperature)))
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        # Unpack parameters
        pos_flat = params[:2*N]
        ang_flat = params[2*N:3*N]
        oc = params[3*N:3*N+2]
        oa = params[3*N+2]
        
        pos = pos_flat.reshape(N, 2)
        ang = ang_flat
        oc_vec = oc
        
        # Compute the side length needed
        s = container_side_length(pos, ang, oc_vec, oa)
        
        # Add penalty for overlapping hexagons
        penalty = 0.0
        min_dist = 2.0 * np.sqrt(3.0)  # minimum distance between centers
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    overlap = min_dist - d
                    penalty += overlap ** 2
        
        # Add penalty if any hexagon is not properly contained
        # (this should be handled by the side length calculation, but add extra safety)
        outer_theta = np.radians(oa)
        R_outer = np.array([[np.cos(outer_theta), -np.sin(outer_theta)],
                            [np.sin(outer_theta), np.cos(outer_theta)]])
        face_normal_angles = (30 + np.arange(6) * 60) * np.pi / 180
        face_normals = np.stack([np.cos(face_normal_angles), np.sin(face_normal_angles)], axis=1)
        face_normals = face_normals @ R_outer.T
        
        containment_penalty = 0.0
        inradius = s * np.sqrt(3.0) / 2.0
        for i in range(N):
            verts = get_hex_vertices(pos[i], ang[i])
            verts_rel = verts - oc_vec
            for fn in face_normals:
                proj = np.max(verts_rel @ fn)
                if proj > inradius + 1e-6:
                    containment_penalty += (proj - inradius) ** 2
        
        return s + 10.0 * penalty + 100.0 * containment_penalty
    
    # Initial configuration: honeycomb lattice
    hex_dist = 2.0 * np.sqrt(3.0)  # distance between adjacent hexagon centers
    
    # 7-hexagon cluster positions
    base_positions = [(0.0, 0.0)]
    for k in range(6):
        angle = k * 60.0 * np.pi / 180
        base_positions.append((hex_dist * np.cos(angle), hex_dist * np.sin(angle)))
    
    # Add 4 more hexagons at the next ring positions
    extra_positions = []
    for k in range(12):
        angle = (30 + k * 30) * np.pi / 180
        extra_positions.append((2 * hex_dist * np.cos(angle), 2 * hex_dist * np.sin(angle)))
    
    # Take first 4 extra positions
    positions = np.array(base_positions + extra_positions[:4], dtype=float)
    angles = np.zeros(N)
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    # Initial parameters
    init_params = np.concatenate([
        positions.flatten(),
        angles,
        outer_center,
        [outer_angle]
    ])
    
    best_params = init_params
    best_obj = objective(init_params)
    
    # Try optimization from multiple starting points
    start_configs = [init_params]
    
    # Add some perturbed starting points
    rng = np.random.RandomState(42)
    for trial in range(5):
        perturbed = init_params.copy()
        perturbed[:2*N] += rng.normal(0, 0.5, size=(2*N,))
        start_configs.append(perturbed)
    
    for sc in start_configs:
        if time.time() > deadline - 25:
            break
        try:
            res = minimize(objective, sc, method='L-BFGS-B', 
                          options={'maxiter': 300, 'maxfun': 1000, 'ftol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Extract best solution
    pos_flat = best_params[:2*N]
    ang_flat = best_params[2*N:3*N]
    oc = best_params[3*N:3*N+2]
    oa = best_params[3*N+2]
    
    pos = pos_flat.reshape(N, 2)
    ang = ang_flat
    
    # Final verification: ensure all hexagons are contained
    # If not, increase the side length slightly
    s = container_side_length(pos, ang, oc, oa)
    
    # Verify containment
    outer_theta = np.radians(oa)
    R_outer = np.array([[np.cos(outer_theta), -np.sin(outer_theta)],
                        [np.sin(outer_theta), np.cos(outer_theta)]])
    face_normal_angles = (30 + np.arange(6) * 60) * np.pi / 180
    face_normals = np.stack([np.cos(face_normal_angles), np.sin(face_normal_angles)], axis=1)
    face_normals = face_normals @ R_outer.T
    
    inradius = s * np.sqrt(3.0) / 2.0
    max_violation = 0.0
    for i in range(N):
        verts = get_hex_vertices(pos[i], ang[i])
        verts_rel = verts - oc
        for fn in face_normals:
            proj = np.max(verts_rel @ fn)
            if proj > inradius:
                max_violation = max(max_violation, proj - inradius)
    
    if max_violation > 1e-8:
        # Increase side length to fix containment
        s = s + 2.0 * max_violation / np.sqrt(3.0) + 0.01
    
    # Convert to the required format
    inner = []
    for i in range(N):
        inner.append([pos[i, 0], pos[i, 1], np.degrees(ang[i])])
    
    return inner, (oc[0], oc[1]), s, np.degrees(oa)
