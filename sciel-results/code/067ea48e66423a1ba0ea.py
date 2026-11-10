import time
import numpy as np
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _edge_normals(V):
    n = np.empty((6, 2))
    for i in range(6):
        e = V[(i + 1) % 6] - V[i]
        L = np.hypot(e[0], e[1])
        if L == 0:
            n[i] = [0, 0]
        else:
            n[i] = np.array([e[1], -e[0]]) / L
    return n

def _gap_and_margin(A, B):
    """Return (best_margin, active_normal). best_margin>0 means separated."""
    nA = _edge_normals(A)
    nB = _edge_normals(B)
    best = -np.inf
    best_n = nA[0]
    for n in nA:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
            best_n = n
    for n in nB:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
            best_n = n
    return best, best_n

def _all_valid(inner, outer, side):
    """Full validity check. inner (11,3), outer (3,), side float."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(side):
        return False
    if side < np.sqrt(11.0) - 1e-9:
        return False
    # separation
    Vs = [_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(11)]
    for i in range(11):
        for j in range(i + 1, 11):
            m, _ = _gap_and_margin(Vs[i], Vs[j])
            if m <= 1e-6:
                return False
    # containment
    cx, cy, ca = outer[0], outer[1], outer[2]
    OV = _hex_vertices(cx, cy, side, ca)
    for V in Vs:
        for i in range(6):
            p1 = OV[i]
            p2 = OV[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (V[:, 1] - p1[1]) - (p2[1] - p1[1]) * (V[:, 0] - p1[0])
            if np.any(cross < -1e-6):
                return False
    # edge length check after shifting by outer center
    for V in Vs:
        Vs2 = V - np.array([cx, cy])
        for i in range(6):
            L = np.hypot(*(Vs2[(i + 1) % 6] - Vs2[i]))
            if abs(L - 1.0) > 1e-8:
                return False
    return True

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 590.0  # Safety margin

    def _outer_side_for_points(points, phi_deg):
        """
        Calculate the minimum side length of a regular hexagon with rotation phi_deg
        that encloses the given points. The hexagon is centered optimally.
        """
        phi = np.deg2rad(phi_deg)
        # Normals of the 6 sides of the hexagon
        # The hexagon is defined by 6 half-planes: n_k . (p - c) <= apothem
        # n_k are unit vectors pointing outward from center to side midpoints.
        # For a regular hexagon with side s, apothem = s * sqrt(3) / 2.
        # The angle of the normal for side k is phi + 60*k.
        angles = phi + np.deg2rad(60.0) * np.arange(6)
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # Project points onto normals
        # proj[i, k] = points[i] . normals[k]
        proj = points @ normals.T
        
        # For a given orientation, the minimal enclosing hexagon centered at c
        # requires apothem >= max_k ( max_i (proj[i,k] - c . n_k) )
        # AND apothem >= max_k ( -min_i (proj[i,k] - c . n_k) ) = max_k ( c . n_k - min_i proj[i,k] )
        
        # To minimize the side length (and thus apothem), we can choose c.
        # However, the problem asks for a specific outer hexagon (x, y, angle).
        # In the optimization, we will treat the outer hexagon's center as variables too?
        # No, the task returns (inner, outer, side).
        # The objective is to minimize side.
        # For a fixed set of points and fixed orientation phi, the minimal side length
        # is determined by the width of the point set in the 6 normal directions.
        # Specifically, the hexagon must cover the range of projections.
        # Let R_k = max_i proj[i,k] - min_i proj[i,k].
        # The hexagon's "width" in direction n_k is 2 * apothem.
        # So 2 * apothem >= R_k for all k.
        # apothem >= max_k (R_k / 2).
        # side = 2 * apothem / sqrt(3) = max_k (R_k) / sqrt(3).
        
        ranges = proj.max(axis=0) - proj.min(axis=0)
        max_range = ranges.max()
        return max_range / np.sqrt(3.0)

    def _separation_penalty(Vs):
        """
        Calculate a penalty for overlaps between inner hexagons.
        We want the gap to be > 0.
        """
        penalty = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                m, _ = _gap_and_margin(Vs[i], Vs[j])
                if m < 1e-6:
                    # Penalty for being too close or overlapping
                    # Use a quadratic penalty for m < 0 (overlap) and linear for 0 <= m < 1e-6
                    if m < 0:
                        penalty += 100.0 * (m ** 2)
                    else:
                        penalty += 10.0 * (1e-6 - m)
        return penalty

    def _objective(params):
        # params: [x1, y1, a1, ..., x11, y11, a11, phi_outer]
        # Total 34 parameters.
        inner_params = params[:33].reshape(11, 3)
        phi_outer = params[33]
        
        # Unpack inner
        cx = inner_params[:, 0]
        cy = inner_params[:, 1]
        ca = inner_params[:, 2]
        
        # Compute vertices for penalty and side calculation
        Vs = []
        for i in range(11):
            Vs.append(_hex_vertices(cx[i], cy[i], 1.0, ca[i]))
        
        # 1. Separation Penalty
        pen = _separation_penalty(Vs)
        
        # 2. Outer Side Length
        # We need the points that define the cluster.
        # The cluster is the union of the hexagons.
        # The bounding hexagon must enclose all vertices of all inner hexagons.
        all_vertices = np.vstack(Vs)
        
        side = _outer_side_for_points(all_vertices, phi_outer)
        
        # Objective: Minimize side, but penalize overlaps heavily.
        # We want side to be small, but if there's overlap, the side calculation
        # might be artificially small or the configuration invalid.
        # By adding the penalty, we push the optimizer to find a valid configuration
        # with minimal side.
        
        return side + pen

    # Initial guess: A hexagonal flower of 7 + 4 extensions
    # Axial to Cartesian
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        return np.array([x, y])
    
    coords_A = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (1,1), (-1,1), (0,2), (0,-2)
    ]
    
    init_points = np.array([axial_to_cartesian(q, r) for q, r in coords_A])
    # Center it
    init_points -= init_points.mean(axis=0)
    
    # Initialize angles to 0
    init_angles = np.zeros(11)
    
    # Flatten for optimizer
    init_params = np.zeros(34)
    init_params[:33] = np.column_stack((init_points, init_angles)).flatten()
    init_params[33] = 0.0 # Outer rotation
    
    best_val = -np.inf
    best_result = None
    
    # Run Nelder-Mead
    # Nelder-Mead is derivative-free and robust for non-smooth penalties (though our penalty is somewhat smooth)
    # We will run multiple restarts with slight perturbations
    
    for trial in range(10):
        if time.time() > deadline - 10.0:
            break
            
        if trial == 0:
            x0 = init_params
        else:
            # Perturb initial positions and angles
            rng = np.random.default_rng(trial)
            x0 = init_params.copy()
            # Perturb centers
            x0[0::3] += rng.normal(0, 0.1, 11)
            x0[1::3] += rng.normal(0, 0.1, 11)
            # Perturb angles
            x0[2::3] += rng.normal(0, 10, 11)
            # Perturb outer rotation
            x0[33] += rng.normal(0, 10)
            
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        except Exception:
            continue
            
        if time.time() > deadline - 5.0:
            break
            
        # Reconstruct result
        inner_params = res.x[:33].reshape(11, 3)
        phi_outer = res.x[33]
        
        inner = inner_params.copy()
        
        # Calculate optimal side length and center for the outer hexagon
        # The _objective used a simplified side calculation assuming the hexagon
        # is centered to minimize the range. But the task requires returning
        # a specific outer hexagon (x, y, angle).
        # We need to find the center (ox, oy) that minimizes the side length
        # for the given orientation phi_outer and the final cluster.
        
        all_vertices = np.vstack([_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(11)])
        
        # For a fixed orientation, the minimal enclosing regular hexagon's center
        # is not trivially the centroid. However, we can approximate it or solve for it.
        # Let's use a local minimization for the center (ox, oy) given the orientation.
        
        def _side_with_center(ox_oy):
            ox, oy = ox_oy
            phi = np.deg2rad(phi_outer)
            angles = phi + np.deg2rad(60.0) * np.arange(6)
            normals = np.column_stack((np.cos(angles), np.sin(angles)))
            
            # Shift points
            shifted = all_vertices - np.array([ox, oy])
            proj = shifted @ normals.T
            
            # The hexagon is defined by -apothem <= n_k . p <= apothem
            # So we need apothem >= max_k ( max_i proj[i,k] ) AND apothem >= max_k ( -min_i proj[i,k] )
            max_pos = proj.max(axis=0)
            max_neg = -proj.min(axis=0)
            max_apothem = np.maximum(max_pos, max_neg).max()
            
            side = 2 * max_apothem / np.sqrt(3.0)
            return side
            
        # Optimize center
        # Start with centroid
        ox0, oy0 = all_vertices.mean(axis=0)
        res_center = minimize(_side_with_center, [ox0, oy0], method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
        
        ox_opt, oy_opt = res_center.x
        side_opt = res_center.fun
        
        outer = np.array([ox_opt, oy_opt, phi_outer])
        
        # Check validity
        if _all_valid(inner, outer, side_opt):
            val = 1.0 / side_opt
            if val > best_val:
                best_val = val
                best_result = (inner, outer, side_opt)
                
    if best_result is None:
        # Fallback to the seed's fallback or a known valid configuration
        # Let's return a valid, though not optimal, configuration
        # Use the initial cluster with a large enough outer hexagon
        inner = init_params[:33].reshape(11, 3)
        all_vertices = np.vstack([_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(11)])
        ox0, oy0 = all_vertices.mean(axis=0)
        side_opt = _side_with_center([ox0, oy0])
        outer = np.array([ox0, oy0, 0.0])
        return inner, outer, side_opt

    return best_result

# EVOLVE-BLOCK-END
