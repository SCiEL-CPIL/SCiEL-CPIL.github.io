import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, a):
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    """
    Compute the minimum side length of a regular hexagon with the given angle
    that encloses all unit hexagons at the given positions.
    Assumes inner hexagons are aligned with angle 0.
    """
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    
    # All inner hexagon vertices
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    a_rad = np.deg2rad(angle_deg)
    
    # Outward normals of the outer hexagon edges
    # Edge i connects vertex i and i+1.
    # Outward normal angle is a + 60i + 30 degrees.
    normals = []
    for i in range(6):
        n_angle = a_rad + i * np.pi/3 + np.pi/6
        n_unit = np.array([np.cos(n_angle), np.sin(n_angle)])
        normals.append(n_unit)
    normals = np.array(normals)
    
    # Project all vertices onto normals
    projections = all_verts_flat @ normals.T # (36, 6)
    max_projections = np.max(projections, axis=0) # (6,)
    
    # s * sqrt(3)/2 >= max_proj  =>  s >= 2 * max_proj / sqrt(3)
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    
    return min_s

def _check_validity(inner, outer, s):
    """
    Validates the solution according to the task contract.
    Returns True if valid, False otherwise.
    """
    # Check finite
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    
    if s < np.sqrt(12) - 1e-9:
        return False
        
    # Check outer vertices distinct
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
                
    # Check unit edge length of inner hexagons
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False

    # Check separation (SAT)
    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_vertices(cx, cy, 1, ang))
        
    for i in range(12):
        for j in range(i+1, 12):
            va = inner_verts_list[i]
            vb = inner_verts_list[j]
            
            separated = False
            # Check axes from A
            for k in range(6):
                v1 = va[k]
                v2 = va[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if separated: continue
            
            # Check axes from B
            for k in range(6):
                v1 = vb[k]
                v2 = vb[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if not separated:
                return False
                
    # Check inner vertices inside outer
    for i in range(12):
        for v in inner_verts_list[i]:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                e = p2 - p1
                cross_val = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
                
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580 # Safety margin
    
    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = None
    
    # Initial guess: Hexagonal lattice arrangement
    h = np.sqrt(3)/2
    init_pos = np.array([
        [0, 0],
        [2*h, 0],
        [h, 1.5],
        [-h, 1.5],
        [-2*h, 0],
        [-h, -1.5],
        [h, -1.5],
        [3*h, 1.5],
        [0, 3],
        [-3*h, 1.5],
        [-3*h, -1.5],
        [3*h, -1.5]
    ])
    
    # We optimize 12 positions (24 vars) + 1 outer angle (1 var) = 25 vars
    # Inner angles are fixed to 0 for simplicity and symmetry, which is often optimal for hexagon packing.
    
    def objective(x):
        positions = x[:24].reshape(12, 2)
        angle_deg = x[24]
        
        # Compute required s
        s = _min_outer_side(positions, angle_deg)
        
        # Penalty for overlaps
        # We check if any two inner hexagons overlap.
        # A cheap proxy for overlap in hexagonal lattice contexts is center distance.
        # However, for arbitrary positions, we need a better check.
        # Let's use a smooth penalty based on the minimum distance between vertices of different hexagons.
        # This is expensive. 
        # Alternative: Use a simpler geometric constraint.
        # If centers are closer than a certain value, they *might* overlap.
        # For unit hexagons, if distance < 1.5 (approx), they definitely overlap?
        # Actually, the minimum distance between centers of two non-overlapping unit hexagons 
        # aligned at 0 is 1.5 (touching edges) or sqrt(3) (touching vertices)?
        # Wait, if they are aligned, edge-to-edge touching distance is 1.5.
        # Vertex-to-vertex touching distance is sqrt(3) ~ 1.732.
        # So if distance < 1.5, they overlap.
        # If distance is between 1.5 and 1.732, it depends on angle.
        # Since we fix angle to 0, we can use a smooth penalty for d < 1.5.
        
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(positions[i] - positions[j])
                # Smooth step function for penalty
                if d < 1.5:
                    penalty += (1.5 - d)**2
                # Also penalize if they are too close even if not strictly overlapping
                # to encourage separation
                elif d < 1.6:
                    penalty += 0.1 * (1.6 - d)**2
        
        # We want to minimize s, but maximize score 1/s.
        # Minimizing s is equivalent.
        # Add penalty to s to discourage overlaps.
        return s + penalty * 10.0

    # Multi-start Nelder-Mead
    num_restarts = 20
    np.random.seed(42)
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial guess
        if r == 0:
            x0 = np.concatenate([init_pos.flatten(), [0.0]])
        else:
            # Perturb the initial guess
            noise_pos = np.random.randn(12, 2) * 0.5
            noise_angle = np.random.uniform(-10, 10)
            x0 = np.concatenate([(init_pos + noise_pos).flatten(), [noise_angle]])
        
        # Run Nelder-Mead
        try:
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
        except Exception:
            continue
            
        if res.success or res.fun < 100: # Accept if it found something reasonable
            x_opt = res.x
            positions = x_opt[:24].reshape(12, 2)
            angle_deg = x_opt[24]
            
            # Construct candidate
            inner_candidate = np.zeros((12, 3))
            inner_candidate[:, :2] = positions
            inner_candidate[:, 2] = 0.0
            
            outer_candidate = np.array([0, 0, angle_deg])
            
            # Recompute s exactly for this configuration
            s_candidate = _min_outer_side(positions, angle_deg)
            
            # Check validity
            if _check_validity(inner_candidate, outer_candidate, s_candidate):
                score_candidate = 1.0 / s_candidate
                if score_candidate > best_score:
                    best_score = score_candidate
                    best_inner = inner_candidate
                    best_outer = outer_candidate
                    best_s = s_candidate
            else:
                # If invalid, try to repair by slightly increasing s?
                # Or just discard.
                # Since Nelder-Mead minimizes s, it might have pushed into an overlap.
                # We can try a local refinement with a stronger penalty?
                # For now, discard.
                pass

    # If we found a valid solution, return it
    if best_inner is not None:
        return best_inner, best_outer, best_s
    else:
        # Fallback to a known valid configuration
        inner_fallback = np.array([
            [0, 0, 0],
            [2.5, 0, 0],
            [-2.5, 0, 0],
            [1.25, 2.165, 0],
            [-1.25, 2.165, 0],
            [1.25, -2.165, 0],
            [-1.25, -2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, -2.165, 0],
            [-3.75, -2.165, 0],
            [0, 4.33, 0]
        ])
        outer_fallback = np.array([0, 0, 0])
        s_fallback = _min_outer_side(inner_fallback[:, :2], 0)
        
        if s_fallback < np.sqrt(12):
            s_fallback = np.sqrt(12)
            
        return inner_fallback, outer_fallback, s_fallback

# EVOLVE-BLOCK-END
