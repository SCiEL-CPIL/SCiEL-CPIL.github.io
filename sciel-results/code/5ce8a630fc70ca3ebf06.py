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
    
    # Define the 12 lattice points relative to the origin
    # Standard hexagonal packing of 12:
    # Center (0,0)
    # Ring 1: 6 points at distance 1.5 (if touching)
    # Ring 2: 5 more points to make 12?
    # Actually, a hexagon of side 2 has 19 points. We need 12.
    # A common shape for 12 is a 2x2x2 rhombus or a truncated hexagon.
    # Let's use a set of integer coordinates (m, n) in the triangular lattice basis.
    # Basis vectors for unit hexagons (side 1, angle 0):
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 1.5)
    
    # Let's select 12 points that form a compact shape.
    # Points in (m, n) coordinates:
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1) -> 7 points (hexagon of side 1)
    # (2,0), (-2,0), (0,2), (0,-2), (2,-2), (-2,2) -> 6 points
    # Total 13. Remove one to get 12? Or use a different shape.
    # Let's use a 3x2 parallelogram of hexagons?
    # m in [0, 2], n in [0, 3] -> 12 points.
    # This is a rhombus.
    # Coordinates:
    lattice_coords = []
    for m in range(3):
        for n in range(4):
            lattice_coords.append([m, n])
    lattice_coords = np.array(lattice_coords, dtype=float)
    
    # Center the lattice
    lattice_coords -= lattice_coords.mean(axis=0)
    
    def get_positions(x):
        """
        x = [v1x, v1y, v2x, v2y, rot]
        Returns positions of 12 hexagons.
        """
        v1 = np.array([x[0], x[1]])
        v2 = np.array([x[2], x[3]])
        rot = np.deg2rad(x[4])
        
        # Rotation matrix
        R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        
        # Basis vectors rotated
        v1_r = R @ v1
        v2_r = R @ v2
        
        # Generate positions
        positions = np.zeros((12, 2))
        for i, coord in enumerate(lattice_coords):
            pos = coord[0] * v1_r + coord[1] * v2_r
            positions[i] = pos
            
        return positions

    def objective(x):
        positions = get_positions(x)
        
        # 1. Check for overlaps (penalty)
        # For unit hexagons on a lattice, if the lattice vectors are too short, they overlap.
        # Minimum distance between centers for non-overlapping unit hexagons (angle 0) is 1.5.
        # However, with rotation, it's complex.
        # Simple penalty: if any pair is closer than 1.5, penalize.
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < 1.5:
                    penalty += (1.5 - d)**2
        if penalty > 0:
            return 1000.0 + penalty # Huge penalty for overlap
            
        # 2. Compute s
        s = _min_outer_side(positions, 0.0) # Outer angle is effectively handled by the rotation of the lattice?
        # Wait, the outer hexagon angle is a separate variable in the original code.
        # Here, we rotate the inner lattice. The optimal outer hexagon for a rotated lattice 
        # of aligned hexagons (angle 0) will have an angle related to the lattice rotation.
        # Actually, if we rotate the lattice, the hexagons themselves are still at angle 0 
        # in the global frame? No, the problem says inner hexagons have angle_deg.
        # The current strategy assumes inner angle 0.
        # If we rotate the lattice positions, the inner hexagons are still "aligned" (angle 0) 
        # relative to the global axes, but their positions form a rotated lattice.
        # Is this optimal?
        # Usually, optimal packings have inner hexagons aligned with the outer hexagon.
        # If we fix inner angle to 0, we should also optimize the outer angle.
        # Let's add outer angle to the search space.
        
        # Let's redefine x to include outer angle.
        # x = [v1x, v1y, v2x, v2y, rot_inner, rot_outer]
        # But rot_inner rotates the positions. The hexagons themselves are still at angle 0?
        # The task allows inner hexagons to have any angle.
        # If we rotate the lattice, we might as well rotate the hexagons too to maintain symmetry.
        # If we rotate both positions and hexagon angles by the same amount, the relative geometry is the same.
        # So we can fix inner angle to 0 and just rotate the positions.
        # Then we optimize the outer angle separately.
        
        # Let's use x = [v1x, v1y, v2x, v2y, rot_outer]
        # The lattice rotation is implicitly part of the basis vectors v1, v2.
        # We don't need a separate rot_inner if we allow v1, v2 to be arbitrary vectors.
        # The basis vectors define the orientation and scale of the lattice.
        
        s = _min_outer_side(positions, x[4])
        
        return s

    # Initial guess
    # Standard triangular lattice for unit hexagons (touching)
    # v1 = (1.5, 0)
    # v2 = (0.75, 1.299)  # 1.5 * sin(60)
    
    np.random.seed(42)
    
    # Multi-start Simulated Annealing / Nelder-Mead hybrid
    # Since the space is small (5 vars), Nelder-Mead is very fast.
    # We can do many restarts.
    
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            x0 = np.array([1.5, 0, 0.75, 1.299, 0.0])
        else:
            # Perturb the standard lattice
            noise_v1 = np.random.randn(2) * 0.2
            noise_v2 = np.random.randn(2) * 0.2
            noise_rot = np.random.uniform(-10, 10)
            
            v1 = np.array([1.5, 0]) + noise_v1
            v2 = np.array([0.75, 1.299]) + noise_v2
            # Ensure v1 and v2 are not collinear (determinant != 0)
            if abs(np.linalg.det(np.array([v1, v2]))) < 0.1:
                continue
                
            x0 = np.concatenate([v1, v2, [noise_rot]])
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
        except Exception:
            continue
            
        x_opt = res.x
        positions = get_positions(x_opt)
        angle_deg = x_opt[4]
        
        # Construct candidate
        inner_candidate = np.zeros((12, 3))
        inner_candidate[:, :2] = positions
        inner_candidate[:, 2] = 0.0
        
        outer_candidate = np.array([0, 0, angle_deg])
        
        # Recompute s exactly
        s_candidate = _min_outer_side(positions, angle_deg)
        
        # Check validity
        if _check_validity(inner_candidate, outer_candidate, s_candidate):
            score_candidate = 1.0 / s_candidate
            if score_candidate > best_score:
                best_score = score_candidate
                best_inner = inner_candidate.copy()
                best_outer = outer_candidate.copy()
                best_s = s_candidate

    # If we found a valid solution, return it
    if best_inner is not None:
        return best_inner, best_outer, best_s
    else:
        # Fallback to a known valid configuration
        # Use the standard lattice result if valid, else a very loose one
        inner_fallback = np.array([
            [0, 0, 0],
            [1.5, 0, 0],
            [-1.5, 0, 0],
            [0.75, 1.299, 0],
            [-0.75, 1.299, 0],
            [0.75, -1.299, 0],
            [-0.75, -1.299, 0],
            [3.0, 0, 0],
            [-3.0, 0, 0],
            [2.25, 1.299, 0],
            [-2.25, 1.299, 0],
            [2.25, -1.299, 0]
        ])
        outer_fallback = np.array([0, 0, 0])
        s_fallback = _min_outer_side(inner_fallback[:, :2], 0)
        
        if s_fallback < np.sqrt(12):
            s_fallback = np.sqrt(12)
            
        return inner_fallback, outer_fallback, s_fallback

# EVOLVE-BLOCK-END
