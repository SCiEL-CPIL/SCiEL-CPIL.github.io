# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


def _hex_vertices(cx, cy, d, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _all_inner_vertices(inner_data, side=1.0):
    """Get all 72 vertices of 12 inner hexagons."""
    verts = []
    for row in inner_data:
        cx, cy, ang = row
        verts.append(_hex_vertices(cx, cy, side, ang))
    return verts


def _separation_violation(inner_data, side=1.0, tol=1e-6):
    """
    Check separating axis test between all pairs of inner hexagons.
    Returns max violation (positive means overlap).
    """
    n = len(inner_data)
    verts = _all_inner_vertices(inner_data, side)
    max_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            vi = verts[i]
            vj = verts[j]
            # Test all 12 edge normals (6 from each hex)
            for k in range(6):
                # Edge normal from hex i
                e = vi[(k + 1) % 6] - vi[k]
                nrm = np.array([-e[1], e[0]])
                nrm_len = np.linalg.norm(nrm)
                if nrm_len < 1e-12:
                    continue
                nrm = nrm / nrm_len
                proj_i = np.dot(vi, nrm)
                proj_j = np.dot(vj, nrm)
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                # Overlap if intervals intersect; separation if gap > 0
                # Violation if they overlap (gap < 0)
                gap = max(min_i - max_j, min_j - max_i)
                if gap < -tol:
                    max_viol = max(max_viol, -gap)
                # Also test normal from hex j
                e2 = vj[(k + 1) % 6] - vj[k]
                nrm2 = np.array([-e2[1], e2[0]])
                nrm2_len = np.linalg.norm(nrm2)
                if nrm2_len < 1e-12:
                    continue
                nrm2 = nrm2 / nrm2_len
                proj_i2 = np.dot(vi, nrm2)
                proj_j2 = np.dot(vj, nrm2)
                min_i2, max_i2 = proj_i2.min(), proj_i2.max()
                min_j2, max_j2 = proj_j2.min(), proj_j2.max()
                gap2 = max(min_i2 - max_j2, min_j2 - max_i2)
                if gap2 < -tol:
                    max_viol = max(max_viol, -gap2)
    return max_viol


def _outer_containment_violation(inner_data, outer_data, outer_side, tol=1e-6):
    """
    Check that all inner vertices are inside the outer hexagon.
    Returns max violation (positive means outside).
    """
    cx, cy, ang = outer_data
    outer_verts = _hex_vertices(cx, cy, outer_side, ang)
    max_viol = 0.0
    for row in inner_data:
        ix, iy, iang = row
        inner_verts = _hex_vertices(ix, iy, 1.0, iang)
        for v in inner_verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                edge = p2 - p1
                # cross product: should be >= 0 for point inside (CCW)
                cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
                if cross < -tol:
                    max_viol = max(max_viol, -cross)
    return max_viol


def _objective(params):
    """
    Objective function: minimize outer side length + penalties for violations.
    params: 12*2 positions + 3 outer (cx, cy, angle) + 1 outer side = 28 params
    """
    deadline = time.time() + 500  # safety margin
    
    positions = params[:24].reshape(12, 2)
    outer_cx, outer_cy, outer_ang = params[24:27]
    outer_side = params[27]
    
    inner_data = np.zeros((12, 3))
    inner_data[:, 0] = positions[:, 0]
    inner_data[:, 1] = positions[:, 1]
    inner_data[:, 2] = 0.0  # keep inner hexagons axis-aligned for simplicity
    
    outer_data = np.array([outer_cx, outer_cy, outer_ang])
    
    # Penalty for separation violations
    sep_viol = _separation_violation(inner_data)
    # Penalty for containment violations
    cont_viol = _outer_containment_violation(inner_data, outer_data, outer_side)
    
    # Also ensure outer side is reasonable
    if outer_side < np.sqrt(12) - 0.1:
        return 1e10 + outer_side
    
    penalty = 100.0 * (sep_viol ** 2 + cont_viol ** 2)
    
    return outer_side + penalty


def _initial_ring_config():
    """
    Generate initial configuration using hexagonal lattice rings.
    Ring 0: 1 hex at center
    Ring 1: 6 hexagons at distance 2*sqrt(3) (tangent)
    Ring 2: 5 hexagons selected from the 12 possible ring-2 positions
    """
    d = 2.0 * np.sqrt(3)  # distance between centers of tangent hexagons
    
    positions = np.zeros((12, 2))
    # Center
    positions[0] = [0, 0]
    
    # Ring 1: 6 hexagons
    for i in range(6):
        angle = np.radians(60 * i)
        positions[i + 1] = [d * np.cos(angle), d * np.sin(angle)]
    
    # Ring 2: pick 5 from 12 possible positions
    # Ring 2 positions are at distances: d*2 (along axes) and d*sqrt(3) (between)
    # Let's use the 6 "corner" positions at distance 2*d and 5 of the 6 "edge" positions
    r2_corner = 2.0 * d
    r2_edge = d * np.sqrt(3)
    
    # 6 corner positions (along the 6 main directions)
    corner_positions = []
    for i in range(6):
        angle = np.radians(60 * i)
        corner_positions.append([r2_corner * np.cos(angle), r2_corner * np.sin(angle)])
    
    # 6 edge positions (between corners, at 30, 90, 150, 210, 270, 330 degrees)
    edge_positions = []
    for i in range(6):
        angle = np.radians(30 + 60 * i)
        edge_positions.append([r2_edge * np.cos(angle), r2_edge * np.sin(angle)])
    
    # Use all 6 corners + ... wait, that's 1+6+6=13. We need 12.
    # Use 1 center + 6 ring1 + 5 from ring2
    # Let's use 5 of the 6 edge positions (skip one for symmetry breaking)
    # Actually, let's use 6 corners and skip one edge: 1+6+6=13, too many.
    # Use 1 center + 6 ring1 + 5 ring2: pick 5 edge positions
    for i in range(5):
        positions[i + 7] = edge_positions[i]
    
    inner_data = np.zeros((12, 3))
    inner_data[:, 0] = positions[:, 0]
    inner_data[:, 1] = positions[:, 1]
    inner_data[:, 2] = 0.0
    
    # Estimate outer hex size
    max_r = np.max(np.sqrt(positions[:, 0]**2 + positions[:, 1]**2))
    outer_side_est = max_r + 1.0  # rough estimate
    
    outer_data = np.array([0.0, 0.0, 0.0])
    
    return inner_data, outer_data, outer_side_est


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    # Get initial configuration
    inner_data, outer_data, outer_side_est = _initial_ring_config()
    
    # Build initial params
    positions = inner_data[:, :2]
    params_init = np.concatenate([positions.flatten(), outer_data, [outer_side_est]])
    
    # Define bounds for differential evolution
    # Positions: roughly -10 to 10
    # Outer center: -5 to 5
    # Outer angle: 0 to 60 (hex has 60-degree symmetry)
    # Outer side: sqrt(12) to 15
    bounds = [(-10, 10)] * 24 + [(-5, 5), (-5, 5), (0, 60), (np.sqrt(12), 15)]
    
    best_result = None
    best_score = -1.0
    best_inner = inner_data.copy()
    best_outer = outer_data.copy()
    best_side = outer_side_est
    
    # Update best with initial
    sep_v = _separation_violation(best_inner)
    cont_v = _outer_containment_violation(best_inner, best_outer, best_side)
    if sep_v < 1e-6 and cont_v < 1e-6:
        best_score = 1.0 / best_side
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=200,
            tol=1e-8,
            popsize=15,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        if time.time() < deadline:
            params_best = result.x
            positions_best = params_best[:24].reshape(12, 2)
            outer_cx, outer_cy, outer_ang = params_best[24:27]
            outer_side = params_best[27]
            
            inner_best = np.zeros((12, 3))
            inner_best[:, 0] = positions_best[:, 0]
            inner_best[:, 1] = positions_best[:, 1]
            inner_best[:, 2] = 0.0
            
            outer_best = np.array([outer_cx, outer_cy, outer_ang])
            
            # Verify validity
            sep_v = _separation_violation(inner_best)
            cont_v = _outer_containment_violation(inner_best, outer_best, outer_side)
            
            if sep_v < 1e-6 and cont_v < 1e-6 and outer_side >= np.sqrt(12) - 1e-8:
                score = 1.0 / outer_side
                if score > best_score:
                    best_score = score
                    best_inner = inner_best
                    best_outer = outer_best
                    best_side = outer_side
            
    except Exception as e:
        pass
    
    # If DE didn't improve, try a local refinement on the initial config
    if time.time() < deadline - 60:
        try:
            from scipy.optimize import minimize
            
            # Refine outer side only, keeping positions fixed
            def obj_outer_side(s):
                outer_test = np.array([best_outer[0], best_outer[1], best_outer[2]])
                # Check containment
                cx, cy, ang = outer_test
                outer_verts = _hex_vertices(cx, cy, s, ang)
                max_viol = 0.0
                for row in best_inner:
                    ix, iy, iang = row
                    inner_verts = _hex_vertices(ix, iy, 1.0, iang)
                    for v in inner_verts:
                        for k in range(6):
                            p1 = outer_verts[k]
                            p2 = outer_verts[(k + 1) % 6]
                            edge = p2 - p1
                            cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
                            if cross < -1e-6:
                                max_viol = max(max_viol, -cross)
                return s + 100.0 * max_viol**2
            
            res = minimize(obj_outer_side, [best_side], method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if time.time() < deadline:
                s_refined = res.x[0]
                if s_refined >= np.sqrt(12) - 1e-8:
                    # Verify
                    outer_refined = np.array([best_outer[0], best_outer[1], best_outer[2]])
                    cont_v = _outer_containment_violation(best_inner, outer_refined, s_refined)
                    if cont_v < 1e-6:
                        score = 1.0 / s_refined
                        if score > best_score:
                            best_score = score
                            best_side = s_refined
                            
        except Exception:
            pass
    
    # Final fallback: ensure we return something valid
    if best_score <= 0:
        # Use the seed's solution as fallback
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
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    return best_inner, best_outer, best_side


# EVOLVE-BLOCK-END
