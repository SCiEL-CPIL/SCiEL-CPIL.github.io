# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _unit_hex_vertices():
    """Vertices of a unit regular hexagon centered at origin, angle 0."""
    angles = np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _rigid_cluster_offsets():
    """
    Fixed hexagonal-lattice cluster: 1 center + 6 ring-1 + 5 ring-2 hexagons.
    Returns (12, 2) array of offsets from cluster centroid.
    """
    sqrt3 = np.sqrt(3.0)
    # Center
    pts = np.array([[0.0, 0.0]])
    # Ring 1: 6 hexagons at distance 2*sqrt(3) in 60-degree increments
    for i in range(6):
        ang = np.deg2rad(60.0 * i)
        pts = np.vstack([pts, [2.0 * sqrt3 * np.cos(ang), 2.0 * sqrt3 * np.sin(ang)]])
    # Ring 2: select 5 of the 12 positions to keep compact
    # Ring 2 positions are at:
    #   distance 2*sqrt(3) at angles 30, 90, 150, 210, 270, 330 (6 positions)
    #   distance 4*sqrt(3) at angles 0, 60, 120, 180, 240, 300 (6 positions)
    # We pick the 6 at distance 2*sqrt(3) (the "corner" positions) and drop one
    # to get 5, keeping the cluster as symmetric as possible.
    # Actually, let's use the 6 "corner" positions (distance 2*sqrt(3) at 30+60k)
    # and drop one to make 11 total, then... wait, we need exactly 12.
    # Center(1) + Ring1(6) = 7. Need 5 more from Ring2.
    # Use the 6 positions at distance 2*sqrt(3) at angles 30+60k, drop one.
    for i in range(5):  # take 5 of the 6
        ang = np.deg2rad(30.0 + 60.0 * i)
        pts = np.vstack([pts, [2.0 * sqrt3 * np.cos(ang), 2.0 * sqrt3 * np.sin(ang)]])
    
    # Center the cluster
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    return pts


def _min_enclosing_hex_side_for_orientation(offsets, outer_angle_deg):
    """
    Given fixed offsets of 12 unit hexagons (all at angle 0 relative to cluster),
    compute the minimum side length s of a regular hexagon at the given outer angle
    that contains all 12 hexagons (cluster centered at origin).
    
    The outer hexagon is centered at the cluster centroid (origin after centering).
    We compute the support function in each of the 6 normal directions.
    """
    outer_angle = np.deg2rad(outer_angle_deg)
    # Outward normal directions of the outer hexagon
    # For a hexagon with vertices at angles (outer_angle + 60k), the edge normals
    # are at angles (outer_angle + 30 + 60k)
    normal_angles = outer_angle + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    # For each of the 12 hexagons, compute the max projection of its vertices
    # onto each normal direction
    unit_verts = _unit_hex_vertices()  # (6, 2)
    
    max_support = 0.0
    for i in range(len(offsets)):
        # Vertices of hexagon i: offsets[i] + unit_verts
        # Projection onto each normal
        # proj = (offsets[i] + unit_verts) @ normals.T  -> (6, 6)
        # We need max over vertices for each normal
        proj = unit_verts @ normals.T + offsets[i] @ normals.T  # (6, 6)
        max_proj_per_normal = np.max(proj, axis=0)  # (6,)
        max_support = max(max_support, np.max(max_proj_per_normal))
    
    # The apothem of the outer hexagon must be >= max_support
    # apothem = s * sqrt(3) / 2
    s = 2.0 * max_support / np.sqrt(3.0)
    return s


def _check_separation(centers, angles_deg, tol=1e-6):
    """Check if all pairs of unit hexagons are separated using SAT."""
    n = len(centers)
    normal_angles_base = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals_base = np.column_stack([np.cos(normal_angles_base), np.sin(normal_angles_base)])
    
    all_verts = []
    for i in range(n):
        a = np.deg2rad(angles_deg[i])
        angles = a + np.deg2rad(60.0 * np.arange(6))
        verts = centers[i] + np.column_stack([np.cos(angles), np.sin(angles)])
        all_verts.append(verts)
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            
            # SAT: check all edge normals of both hexagons
            for k in range(6):
                # Normal of edge k of hex i
                a_i = np.deg2rad(angles_deg[i])
                n_angle_i = a_i + np.deg2rad(30.0 + 60.0 * k)
                n_i = np.array([np.cos(n_angle_i), np.sin(n_angle_i)])
                
                proj_i = vi @ n_i
                proj_j = vj @ n_i
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                if max_i < min_j - tol or max_j < min_i - tol:
                    break
            else:
                # Check normals of hex j
                for k in range(6):
                    a_j = np.deg2rad(angles_deg[j])
                    n_angle_j = a_j + np.deg2rad(30.0 + 60.0 * k)
                    n_j = np.array([np.cos(n_angle_j), np.sin(n_angle_j)])
                    
                    proj_i = vi @ n_j
                    proj_j = vj @ n_j
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    
                    if max_i < min_j - tol or max_j < min_i - tol:
                        break
                else:
                    return False
    return True


def _compute_s_and_centers(outer_angle_deg, outer_cx, outer_cy, offsets):
    """
    Given outer hexagon parameters and cluster offsets, compute:
    - s: minimum side length
    - centers: (12, 2) array of inner hexagon centers
    """
    # The cluster is centered at (outer_cx, outer_cy)
    centers = offsets + np.array([outer_cx, outer_cy])
    s = _min_enclosing_hex_side_for_orientation(offsets, outer_angle_deg)
    return s, centers


def _objective_rigid(params, offsets):
    """
    Objective for rigid body optimization.
    params: [outer_angle_deg, outer_cx, outer_cy]
    """
    outer_angle_deg = params[0]
    outer_cx = params[1]
    outer_cy = params[2]
    
    s = _min_enclosing_hex_side_for_orientation(offsets, outer_angle_deg)
    return s


def _refine_individual_angles(centers, outer_angle_deg, outer_cx, outer_cy, s, n_iters=500):
    """
    Try to refine individual hexagon angles to reduce the required s.
    This is a local optimization that may help if the rigid assumption is suboptimal.
    """
    angles_deg = np.zeros(12)
    
    def _obj_angles(angle_params):
        angles = angle_params
        new_centers = centers  # centers fixed for now
        # Recompute s with individual angles
        # This is expensive; we'll use a simplified version
        # Actually, for the rigid body, angles are all 0.
        # Let's just return the current s as a placeholder.
        return s
    
    # For now, skip individual angle refinement and return as-is
    return angles_deg


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    offsets = _rigid_cluster_offsets()
    
    best_s = np.inf
    best_params = None
    
    # Phase 1: Grid search over outer angle
    # The outer hexagon has 6-fold symmetry, so we only need to search 0 to 60 degrees
    # But due to the cluster's asymmetry, we search the full range
    angle_grid = np.linspace(0.0, 60.0, 61)  # 1-degree resolution
    
    for angle in angle_grid:
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        s = _min_enclosing_hex_side_for_orientation(offsets, angle)
        if s < best_s:
            best_s = s
            best_params = np.array([angle, 0.0, 0.0])
    
    # Phase 2: Nelder-Mead refinement of the 3 parameters
    if best_params is not None:
        elapsed = time.time() - _START_TIME
        if elapsed < _DEADLINE - _SAFETY_MARGIN:
            try:
                result = minimize(
                    _objective_rigid,
                    best_params,
                    args=(offsets,),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'maxfev': 50000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
                if result is not None:
                    s_cand = result.fun
                    if s_cand < best_s:
                        best_s = s_cand
                        best_params = result.x
            except Exception:
                pass
    
    # Phase 3: Try different cluster topologies
    # Alternative cluster: center + 6 ring-1 + 5 ring-2 at distance 4*sqrt(3)
    sqrt3 = np.sqrt(3.0)
    alternative_clusters = []
    
    # Cluster A: original (center + 6 ring-1 + 5 ring-2 at 2*sqrt(3))
    alternative_clusters.append(offsets)
    
    # Cluster B: center + 6 ring-1 + 5 ring-2 at 4*sqrt(3) (the "far" positions)
    pts_b = np.array([[0.0, 0.0]])
    for i in range(6):
        ang = np.deg2rad(60.0 * i)
        pts_b = np.vstack([pts_b, [2.0 * sqrt3 * np.cos(ang), 2.0 * sqrt3 * np.sin(ang)]])
    for i in range(5):
        ang = np.deg2rad(60.0 * i)
        pts_b = np.vstack([pts_b, [4.0 * sqrt3 * np.cos(ang), 4.0 * sqrt3 * np.sin(ang)]])
    centroid_b = np.mean(pts_b, axis=0)
    alternative_clusters.append(pts_b - centroid_b)
    
    # Cluster C: center + 6 ring-1 + 3 ring-2 at 2*sqrt(3) + 2 ring-2 at 4*sqrt(3)
    pts_c = np.array([[0.0, 0.0]])
    for i in range(6):
        ang = np.deg2rad(60.0 * i)
        pts_c = np.vstack([pts_c, [2.0 * sqrt3 * np.cos(ang), 2.0 * sqrt3 * np.sin(ang)]])
    for i in range(3):
        ang = np.deg2rad(30.0 + 60.0 * i)
        pts_c = np.vstack([pts_c, [2.0 * sqrt3 * np.cos(ang), 2.0 * sqrt3 * np.sin(ang)]])
    for i in range(2):
        ang = np.deg2rad(60.0 * i)
        pts_c = np.vstack([pts_c, [4.0 * sqrt3 * np.cos(ang), 4.0 * sqrt3 * np.sin(ang)]])
    centroid_c = np.mean(pts_c, axis=0)
    alternative_clusters.append(pts_c - centroid_c)
    
    for cluster in alternative_clusters:
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Quick grid search for this cluster
        for angle in np.linspace(0.0, 60.0, 31):
            s = _min_enclosing_hex_side_for_orientation(cluster, angle)
            if s < best_s:
                best_s = s
                best_params = np.array([angle, 0.0, 0.0])
                offsets = cluster
    
    # Final: compute centers and verify
    if best_params is None:
        best_params = np.array([0.0, 0.0, 0.0])
    
    outer_angle_deg = best_params[0]
    outer_cx = best_params[1]
    outer_cy = best_params[2]
    
    centers = offsets + np.array([outer_cx, outer_cy])
    s_final = _min_enclosing_hex_side_for_orientation(offsets, outer_angle_deg)
    
    # Verify separation
    angles_deg = np.zeros(12)
    if not _check_separation(centers, angles_deg):
        # Fallback: use a known-valid spread configuration
        sqrt3 = np.sqrt(3.0)
        centers = np.array([
            [0.0, 0.0],
            [2.5, 0.0],
            [-2.5, 0.0],
            [1.25, 2.165],
            [-1.25, 2.165],
            [1.25, -2.165],
            [-1.25, -2.165],
            [3.75, 2.165],
            [-3.75, 2.165],
            [3.75, -2.165],
            [-3.75, -2.165],
            [0.0, -4.33],
        ])
        # Compute s for this configuration
        # Use the general method
        unit_verts = _unit_hex_vertices()
        all_verts = []
        for i in range(12):
            all_verts.append(centers[i] + unit_verts)
        all_verts = np.array(all_verts).reshape(-1, 2)
        
        # Try multiple outer angles
        best_s_fb = np.inf
        for angle in np.linspace(0, 60, 61):
            outer_angle = np.deg2rad(angle)
            normal_angles = outer_angle + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            projections = normals @ all_verts.T
            max_proj = np.max(np.max(projections, axis=1))
            s_candidate = 2.0 * max_proj / np.sqrt(3.0)
            if s_candidate < best_s_fb:
                best_s_fb = s_candidate
                outer_angle_deg = angle
        
        s_final = best_s_fb
        outer_cx = 0.0
        outer_cy = 0.0
    
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([outer_cx, outer_cy, outer_angle_deg])
    outer_hex_side_length = s_final
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
