# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _check_validity(inner, outer, side, tol=1e-6):
    """Check if packing is valid. Returns True if valid."""
    # All finite
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer))):
        return False
    if not np.isfinite(side) or side < np.sqrt(11) - 1e-9:
        return False

    # Shift by outer center
    ox, oy, oa = outer
    inner_shifted = inner.copy()
    inner_shifted[:, 0] -= ox
    inner_shifted[:, 1] -= oy

    # Check edge lengths of inner hexagons (should be 1 within 1e-8)
    for i in range(len(inner_shifted)):
        cx, cy, ang = inner_shifted[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for j in range(6):
            v1 = verts[j]
            v2 = verts[(j + 1) % 6]
            d = np.linalg.norm(v2 - v1)
            if abs(d - 1.0) > 1e-8:
                return False

    # Check all inner hexagons are inside outer hexagon
    outer_verts = _hex_vertices(0, 0, side, oa)  # shifted to origin
    n_outer = len(outer_verts)
    for i in range(n_outer):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % n_outer]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        # For CCW polygon, interior is to the left, so cross(p-p1, edge) >= 0
        # Actually: cross(edge, p - p1) >= 0 for CCW
        for k in range(len(inner_shifted)):
            cx, cy, ang = inner_shifted[k]
            verts = _hex_vertices(cx, cy, 1.0, ang)
            for v in verts:
                cross_val = np.cross(edge, v - p1)
                if cross_val < -tol:
                    return False

    # Check pairwise separation of inner hexagons
    n_inner = len(inner_shifted)
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            ci = inner_shifted[i]
            cj = inner_shifted[j]
            vi = _hex_vertices(ci[0], ci[1], 1.0, ci[2])
            vj = _hex_vertices(cj[0], cj[1], 1.0, cj[2])
            
            # Check if they are separated on at least one of the 12 edge normals
            separated = False
            # Edge normals of hex i
            for k in range(6):
                e = vi[(k + 1) % 6] - vi[k]
                nrm = np.array([-e[1], e[0]])
                nrm = nrm / np.linalg.norm(nrm)
                
                proj_i = np.dot(vi, nrm)
                proj_j = np.dot(vj, nrm)
                
                min_i = np.min(proj_i)
                max_i = np.max(proj_i)
                min_j = np.min(proj_j)
                max_j = np.max(proj_j)
                
                # Check if intervals are apart (gap > tol)
                if max_i + tol < min_j or max_j + tol < min_i:
                    separated = True
                    break
            
            if not separated:
                # Also check normals of hex j
                for k in range(6):
                    e = vj[(k + 1) % 6] - vj[k]
                    nrm = np.array([-e[1], e[0]])
                    nrm = nrm / np.linalg.norm(nrm)
                    
                    proj_i = np.dot(vi, nrm)
                    proj_j = np.dot(vj, nrm)
                    
                    min_i = np.min(proj_i)
                    max_i = np.max(proj_i)
                    min_j = np.min(proj_j)
                    max_j = np.max(proj_j)
                    
                    if max_i + tol < min_j or max_j + tol < min_i:
                        separated = True
                        break
            
            if not separated:
                return False

    return True


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    deadline = time.time() + 600.0
    safety_margin = 10.0  # seconds

    # Start with a known valid packing
    # Honeycomb arrangement: 1 center + 6 surrounding + 4 more
    # For unit hexagons in honeycomb, the distance between centers of adjacent hexagons is sqrt(3)
    # Let's use a standard honeycomb layout
    
    sqrt3 = np.sqrt(3)
    
    # Initial guess: honeycomb pattern
    # Center at origin, 6 around it at distance sqrt(3)
    # Then 4 more at the "corners"
    
    # Positions for honeycomb of 11 hexagons
    # Row 1 (top): 3 hexagons
    # Row 2 (middle): 5 hexagons  
    # Row 3 (bottom): 3 hexagons
    # But offset for hexagonal grid
    
    # Actually, let's use a proper hexagonal grid
    # In a hex grid with spacing d = sqrt(3) between centers:
    # Axial coordinates (q, r) -> (x, y) = (d*(q + r/2), d*sqrt(3)/2*r)
    
    # 11 hexagons in a compact arrangement:
    # (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # plus (2,-1), (-2,1), (1,1), (-1,-1) -- need to verify
    
    # Let me use a known good starting point and optimize
    # Start with the grid from the seed but with better side estimate
    
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ]
    )
    
    outer_hex_data = np.array([0, 0, 0.0])
    outer_hex_side_length = 8.0
    
    # Try to find a tighter packing using optimization
    # We'll optimize the positions and the outer hexagon side
    
    def compute_min_side(inner_pos, outer_angle_deg):
        """Compute minimum side length of outer hexagon to contain all inner hexagons."""
        oa = np.deg2rad(outer_angle_deg)
        # Outer hexagon edges: 6 edges, each perpendicular to a direction
        # The constraint is that all vertices of all inner hexagons must be inside
        
        # For a regular hexagon centered at origin with side s and angle a,
        # the half-width in direction theta is s * cos(min_angle_to_edge_normal)
        # Actually, the support function: for direction u, max of dot(v, u) over hex = s * cos(delta)
        # where delta is angle between u and nearest edge normal
        
        # Edge normals are at angles a + 30 + 60*k for k=0..5 (pointing outward)
        # Wait, for a hexagon with vertices at angles a + 60*i, the edge normals
        # (outward) are at angles a + 30 + 60*i
        
        # The hexagon is the intersection of 6 half-planes: dot(v, n_k) <= s * cos(0) = s
        # No wait. Let me think again.
        
        # Vertices at angle a + 60*i, radius s (circumradius = s for regular hexagon)
        # Edge between vertex i and i+1 has midpoint at angle a + 60*i + 30, distance from center = s*cos(30) = s*sqrt(3)/2
        # The outward normal to that edge points in direction a + 60*i + 30
        # The constraint is: dot(v, n_k) <= s * sqrt(3)/2 for all k
        
        # So for each direction n_k, we need max over all inner vertices of dot(v, n_k) <= s * sqrt(3)/2
        # Thus s >= max_k (max_v dot(v, n_k)) / (sqrt(3)/2)
        
        # Get all vertices of all inner hexagons
        all_verts = []
        for i in range(len(inner_pos)):
            cx, cy, ang = inner_pos[i]
            verts = _hex_vertices(cx, cy, 1.0, ang)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # Edge normal directions
        normal_angles = oa + np.deg2rad(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # For each normal, find max dot product
        max_dots = np.max(all_verts @ normals.T, axis=0)
        
        # s * sqrt(3)/2 >= max_dots, so s >= max_dots / (sqrt(3)/2)
        min_s = np.max(max_dots) / (np.sqrt(3) / 2)
        
        return min_s


    def objective(params):
        """Objective function: minimize outer hexagon side length."""
        # params: 11*2 positions + 11 angles + 1 outer angle = 22 + 11 + 1 = 34
        inner_pos = params[:22].reshape(11, 2)
        inner_angles = params[22:33]
        outer_angle = params[33]
        
        # Check pairwise separation
        for i in range(11):
            for j in range(i + 1, 11):
                vi = _hex_vertices(inner_pos[i, 0], inner_pos[i, 1], 1.0, inner_angles[i])
                vj = _hex_vertices(inner_pos[j, 0], inner_pos[j, 1], 1.0, inner_angles[j])
                
                # Quick check: center distance
                d = np.linalg.norm(inner_pos[i] - inner_pos[j])
                if d < 1.7:  # minimum distance for non-overlapping unit hexagons
                    return 1e6  # penalty
                
                # Check separation on edge normals
                separated = False
                for k in range(6):
                    e = vi[(k + 1) % 6] - vi[k]
                    nrm = np.array([-e[1], e[0]])
                    nrm = nrm / np.linalg.norm(nrm)
                    
                    proj_i = np.dot(vi, nrm)
                    proj_j = np.dot(vj, nrm)
                    
                    if np.max(proj_i) + 1e-6 < np.min(proj_j) or np.max(proj_j) + 1e-6 < np.min(proj_i):
                        separated = True
                        break
                
                if not separated:
                    for k in range(6):
                        e = vj[(k + 1) % 6] - vj[k]
                        nrm = np.array([-e[1], e[0]])
                        nrm = nrm / np.linalg.norm(nrm)
                        
                        proj_i = np.dot(vi, nrm)
                        proj_j = np.dot(vj, nrm)
                        
                        if np.max(proj_i) + 1e-6 < np.min(proj_j) or np.max(proj_j) + 1e-6 < np.min(proj_i):
                            separated = True
                            break
                
                if not separated:
                    return 1e6  # penalty for overlap
        
        # Compute minimum outer side
        s = compute_min_side(inner_pos, outer_angle)
        
        return s


    # Try optimization with limited time
    best_result = (inner_hex_data, outer_hex_data, outer_hex_side_length)
    best_score = 1.0 / outer_hex_side_length

    # Try a few random restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - safety_margin:
            break
        
        # Create a perturbed initial guess
        if restart == 0:
            # Use the seed arrangement
            inner_init = inner_hex_data[:, :2].copy()
            angles_init = inner_hex_data[:, 2].copy()
            outer_angle_init = 0.0
        else:
            # Random perturbation around a honeycomb-like arrangement
            np.random.seed(restart * 42)
            # Honeycomb base
            base_positions = np.array([
                [0, 0],
                [np.sqrt(3), 0],
                [-np.sqrt(3), 0],
                [np.sqrt(3)/2, 1.5],
                [-np.sqrt(3)/2, 1.5],
                [np.sqrt(3)/2, -1.5],
                [-np.sqrt(3)/2, -1.5],
                [3*np.sqrt(3)/2, 1.5],
                [-3*np.sqrt(3)/2, 1.5],
                [3*np.sqrt(3)/2, -1.5],
                [-3*np.sqrt(3)/2, -1.5],
            ])
            inner_init = base_positions + np.random.randn(11, 2) * 0.5
            angles_init = np.zeros(11)
            outer_angle_init = np.random.uniform(0, 60)
        
        params_init = np.concatenate([inner_init.flatten(), angles_init, [outer_angle_init]])
        
        # Use Nelder-Mead for optimization (no gradient needed)
        try:
            result = minimize(
                objective,
                params_init,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            
            if result.fun < 1e5:  # Valid solution
                inner_pos_opt = result.x[:22].reshape(11, 2)
                inner_angles_opt = result.x[22:33]
                outer_angle_opt = result.x[33]
                
                # Build inner hex data
                inner_opt = np.column_stack([inner_pos_opt, inner_angles_opt])
                outer_opt = np.array([0, 0, outer_angle_opt])
                side_opt = result.fun
                
                # Verify validity
                if _check_validity(inner_opt, outer_opt, side_opt):
                    score = 1.0 / side_opt
                    if score > best_score:
                        best_score = score
                        best_result = (inner_opt, outer_opt, side_opt)
        except Exception:
            pass

    return best_result


# EVOLVE-BLOCK-END
