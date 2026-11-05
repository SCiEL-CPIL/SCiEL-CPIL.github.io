"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 145.0  # safety margin before 150s kill

# Cache for MDS coordinates and principal axis
_mds_cache = {}


def _get_mds_coords(distance_matrix):
    """Compute 2D MDS coordinates from distance matrix, cached by matrix id."""
    key = id(distance_matrix)
    if key in _mds_cache:
        return _mds_cache[key]
    
    n = distance_matrix.shape[0]
    D = distance_matrix.astype(np.float64)
    D2 = D ** 2
    
    # MDS: double centering
    one_n = np.ones((n, n)) / n
    B = -0.5 * (one_n @ D2 @ one_n - one_n @ D2 - D2 @ one_n + D2 @ one_n)
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Take top 2 dimensions
    coords = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    
    _mds_cache[key] = coords
    return coords


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    n = len(unvisited_nodes)
    if n == 0:
        return int(depot)

    # Feasible: demand fits in remaining capacity
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)

    feas = unvisited_arr[feas_mask]
    k = len(feas)

    if k == 1:
        return int(feas[0])

    # Check time budget
    if (time.time() - _T0) > _DEADLINE:
        # Fallback: nearest feasible node
        d_cur = distance_matrix[current_node, feas]
        return int(feas[int(np.argmin(d_cur))])

    # Get MDS coordinates
    coords = _get_mds_coords(distance_matrix)
    
    # Current node and depot coordinates
    cur_pos = coords[current_node]
    depot_pos = coords[depot]
    
    # Compute heading vector: from depot through current node (direction of travel)
    heading = cur_pos - depot_pos
    heading_norm = np.linalg.norm(heading)
    
    if heading_norm < 1e-9:
        # At depot, use a default heading
        heading = np.array([1.0, 0.0])
        heading_norm = 1.0
    else:
        heading = heading / heading_norm
    
    # Get coordinates of feasible nodes
    feas_pos = coords[feas]
    
    # Vector from current node to each feasible candidate
    vecs = feas_pos - cur_pos
    vec_norms = np.linalg.norm(vecs, axis=1)
    
    # Handle zero-length vectors (shouldn't happen but be safe)
    vec_norms_safe = np.maximum(vec_norms, 1e-9)
    vecs_unit = vecs / vec_norms_safe[:, np.newaxis]
    
    # Compute angle between heading and each candidate direction
    # cos(angle) = dot product of unit vectors
    cos_angles = vecs_unit @ heading
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles)  # 0 to pi
    
    # Angular deviation from heading (smaller is better - stay on course)
    angular_score = angles
    
    # Also compute distance to depot for each feasible node
    d_depot = distance_matrix[feas, depot]
    
    # Distance from current to each feasible
    d_cur = distance_matrix[current_node, feas]
    
    # Composite score: prioritize staying on the sweep (angular) + compactness
    # Normalize distances to comparable scale
    d_cur_norm = d_cur / (d_cur.max() + 1e-9)
    d_depot_norm = d_depot / (d_depot.max() + 1e-9)
    
    # Weighted combination: angular deviation (sweep) + immediate distance + return distance
    # The sweep angle encourages visiting nodes in a coherent direction
    # The distance terms keep the route compact
    alpha_ang = 0.5    # weight for angular deviation
    alpha_dist = 0.3   # weight for immediate distance
    alpha_ret = 0.2    # weight for return-to-depot distance
    
    scores = alpha_ang * angular_score + alpha_dist * d_cur_norm + alpha_ret * d_depot_norm
    
    idx = int(np.argmin(scores))
    return int(feas[idx])
# EVOLVE-BLOCK-END
