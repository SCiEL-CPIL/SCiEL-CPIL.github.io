# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _edge_normals(angle_deg):
    a = np.radians(angle_deg)
    normal_angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])


def _sat_overlap(c1, a1, c2, a2):
    """Check if two unit hexagons overlap using SAT. Returns (overlap, min_push_vec) or (0, None)."""
    n1 = _edge_normals(a1)
    n2 = _edge_normals(a2)
    all_normals = np.vstack([n1, n2])
    
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    proj1 = v1 @ all_normals.T
    proj2 = v2 @ all_normals.T
    
    p1_min, p1_max = proj1.min(axis=0), proj1.max(axis=0)
    p2_min, p2_max = proj2.min(axis=0), proj2.max(axis=0)
    
    min_overlap = np.inf
    min_push = None
    
    for i in range(12):
        overlap = min(p1_max[i], p2_max[i]) - max(p1_min[i], p2_min[i])
        if overlap <= 0:
            return 0, None  # Separated on this axis
        if overlap < min_overlap:
            min_overlap = overlap
            # Determine push direction
            if p1_min[i] < p2_min[i]:
                # hex1 is to the left, push hex1 left / hex2 right
                push = all_normals[i] * overlap
            else:
                push = -all_normals[i] * overlap
            min_push = push
    
    return min_overlap, min_push


def _compute_s_and_center(verts, rot_deg):
    """Compute min outer hex side s and center for given rotation containing all vertices."""
    sqrt3 = np.sqrt(3)
    rot = np.radians(rot_deg)
    normal_angles = rot + np.radians(60 * np.arange(6) + 30)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    projections = verts @ normals.T
    p_min = projections.min(axis=0)
    p_max = projections.max(axis=0)
    
    b = (p_min + p_max) / 2.0
    center, _, _, _ = np.linalg.lstsq(normals, b, rcond=None)
    
    center_projections = normals @ center
    s = 0.0
    for i in range(6):
        d_proj = projections[:, i] - center_projections[i]
        s = max(s, np.abs(d_proj).max() * 2.0 / sqrt3)
    
    return s, center[0], center[1]


def _resolve_collisions(centers, angles, max_iter=500, push_factor=0.5):
    """Iteratively resolve all pairwise collisions."""
    centers = centers.copy()
    for _ in range(max_iter):
        moved = False
        for i in range(12):
            for j in range(i + 1, 12):
                overlap, push = _sat_overlap(centers[i], angles[i], centers[j], angles[j])
                if overlap > 1e-9 and push is not None:
                    # Push both hexagons apart
                    half_push = push * push_factor
                    centers[i] -= half_push
                    centers[j] += half_push
                    moved = True
        if not moved:
            break
    return centers


def _validate(inner, outer, s, eps=1e-6):
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    ox, oy, oang = outer
    outer_verts = _hex_vertices(ox, oy, s, oang)
    
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    for i in range(12):
        for j in range(i + 1, 12):
            overlap, _ = _sat_overlap(inner[i, :2], inner[i, 2], inner[j, :2], inner[j, 2])
            if overlap > 1e-7:
                return False
    
    for i in range(12):
        cx, cy, aang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, aang)
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False
    
    return True


def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590.0
    sqrt3 = np.sqrt(3)
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    def try_config(centers, angles):
        nonlocal best_s, best_inner, best_outer
        if time.time() > deadline - 5:
            return
        
        # Resolve collisions first
        centers_resolved = _resolve_collisions(centers, angles)
        
        # Check if resolved config is valid (no overlaps)
        valid = True
        for i in range(12):
            for j in range(i + 1, 12):
                overlap, _ = _sat_overlap(centers_resolved[i], angles[i], centers_resolved[j], angles[j])
                if overlap > 1e-7:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            return
        
        all_verts = np.zeros((72, 2))
        for i in range(12):
            all_verts[6*i:6*(i+1)] = _hex_vertices(centers_resolved[i, 0], centers_resolved[i, 1], 1.0, angles[i])
        
        # Try multiple outer rotations
        for rot in np.linspace(0, 60, 25):
            if time.time() > deadline - 5:
                break
            s, ox, oy = _compute_s_and_center(all_verts, rot)
            
            if s < best_s:
                inner = np.column_stack([centers_resolved, angles])
                outer = np.array([ox, oy, rot])
                if _validate(inner, outer, s):
                    best_s = s
                    best_inner = inner.copy()
                    best_outer = outer.copy()
    
    # Define seed configurations
    e1_std = np.array([2.0, 0.0])
    e2_std = np.array([1.0, sqrt3])
    
    # Seed 1: Center + ring of 6 + partial second ring
    lattice1 = [(0, 0),
                (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                (-1, 2), (-2, 1), (2, -1), (1, -2), (0, 2)]
    
    # Seed 2: 2-3-4-3 rows
    lattice2 = [(-1, 0), (0, 0),
                (-1, 1), (0, 1), (1, 1),
                (-2, 2), (-1, 2), (0, 2), (1, 2),
                (-1, 3), (0, 3), (1, 3)]
    
    # Seed 3: 3-4-3-2 rows
    lattice3 = [(-1, 0), (0, 0), (1, 0),
                (-2, 1), (-1, 1), (0, 1), (1, 1),
                (-1, 2), (0, 2), (1, 2),
                (0, 3), (1, 3)]
    
    # Seed 4: Compact triangular
    lattice4 = [(0, 0), (1, 0), (-1, 0),
                (0, 1), (1, 1), (-1, 1), (0, -1),
                (0, 2), (1, 2), (-1, 2),
                (2, 0), (-2, 0)]
    
    # Seed 5: Alternative compact
    lattice5 = [(0, 0), (2, 0), (-2, 0),
                (1, 1), (-1, 1), (1, -1), (-1, -1),
                (0, 2), (0, -2), (2, 1), (-2, -1), (2, -1), (-2, 1)]
    
    seeds = [lattice1, lattice2, lattice3, lattice4, lattice5]
    
    # Phase 1: Try all seeds with collision resolution
    for lattice in seeds:
        if time.time() > deadline - 100:
            break
        centers = np.array([c[0] * e1_std + c[1] * e2_std for c in lattice])
        angles = np.zeros(12)
        try_config(centers, angles)
    
    # Phase 2: Perturbed seeds - jitter positions and resolve
    if time.time() < deadline - 80:
        rng = np.random.default_rng(42)
        for lattice in seeds[:3]:
            for trial in range(30):
                if time.time() > deadline - 60:
                    break
                centers = np.array([c[0] * e1_std + c[1] * e2_std for c in lattice])
                # Add small random perturbation
                centers += rng.normal(0, 0.3, size=(12, 2))
                angles = np.zeros(12)
                try_config(centers, angles)
    
    # Phase 3: Vary inner hexagon angles
    if time.time() < deadline - 50:
        for lattice in seeds[:2]:
            for trial in range(20):
                if time.time() > deadline - 30:
                    break
                centers = np.array([c[0] * e1_std + c[1] * e2_std for c in lattice])
                # Random small angle perturbations
                angles = rng.uniform(-10, 10, size=12)
                try_config(centers, angles)
    
    # Phase 4: Local search - move one hex at a time
    if time.time() < deadline - 20 and best_inner is not None:
        current_inner = best_inner.copy()
        current_s = best_s
        
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            
            improved = False
            for i in range(12):
                if time.time() > deadline - 10:
                    break
                # Try moving hex i in small steps
                for dx in [-0.2, -0.1, 0, 0.1, 0.2]:
                    for dy in [-0.2, -0.1, 0, 0.1, 0.2]:
                        new_centers = current_inner[:, :2].copy()
                        new_centers[i] += [dx, dy]
                        new_angles = current_inner[:, 2].copy()
                        
                        # Quick check: does this hex still fit?
                        centers_resolved = _resolve_collisions(new_centers, new_angles, max_iter=100)
                        
                        # Check validity quickly
                        valid = True
                        for k in range(12):
                            for l in range(k + 1, 12):
                                overlap, _ = _sat_overlap(centers_resolved[k], new_angles[k], centers_resolved[l], new_angles[l])
                                if overlap > 1e-7:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if not valid:
                            continue
                        
                        all_verts = np.zeros((72, 2))
                        for k in range(12):
                            all_verts[6*k:6*(k+1)] = _hex_vertices(centers_resolved[k, 0], centers_resolved[k, 1], 1.0, new_angles[k])
                        
                        best_local_s = np.inf
                        for rot in np.linspace(0, 60, 13):
                            s, _, _ = _compute_s_and_center(all_verts, rot)
                            best_local_s = min(best_local_s, s)
                        
                        if best_local_s < current_s - 1e-8:
                            # Accept improvement
                            current_inner[:, :2] = centers_resolved
                            current_s = best_local_s
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        if current_s < best_s:
            # Final validation
            centers_final = current_inner[:, :2]
            angles_final = current_inner[:, 2]
            all_verts = np.zeros((72, 2))
            for i in range(12):
                all_verts[6*i:6*(i+1)] = _hex_vertices(centers_final[i, 0], centers_final[i, 1], 1.0, angles_final[i])
            
            best_rot = 0
            best_final_s = np.inf
            for rot in np.linspace(0, 60, 50):
                s, ox, oy = _compute_s_and_center(all_verts, rot)
                if s < best_final_s:
                    best_final_s = s
                    best_rot = rot
            
            inner_final = np.column_stack([centers_final, angles_final])
            outer_final = np.array([ox, oy, best_rot])
            if _validate(inner_final, outer_final, best_final_s):
                best_s = best_final_s
                best_inner = inner_final
                best_outer = outer_final
    
    # Fallback
    if best_inner is None:
        centers = np.array([c[0] * e1_std * 1.001 + c[1] * e2_std * 1.001 for c in lattice1])
        angles = np.zeros(12)
        centers = _resolve_collisions(centers, angles)
        all_verts = np.zeros((72, 2))
        for i in range(12):
            all_verts[6*i:6*(i+1)] = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        s, ox, oy = _compute_s_and_center(all_verts, 0)
        best_s = s
        best_inner = np.column_stack([centers, angles])
        best_outer = np.array([ox, oy, 0.0])
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
