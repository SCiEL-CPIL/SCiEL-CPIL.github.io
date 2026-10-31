"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We cache some per-instance state to avoid recomputation across calls.
# The evaluator likely calls select_next_node sequentially to build one tour per instance.
# We detect a new instance by the shape/hash of the distance matrix.

_cache = {
    'dm_hash': None,
    'coords': None,       # (N, 2) Euclidean coordinates recovered from DM
    'dm': None,
    'n': 0,
    'start_time': None,
    'deadline': None,
}

def _recover_coords(distance_matrix):
    """
    Recover approximate 2D coordinates from a Euclidean distance matrix using
    classical MDS (double centering + eigendecomposition).
    This is done once per instance.
    """
    n = distance_matrix.shape[0]
    d2 = distance_matrix ** 2
    # Double centering
    one_n = np.ones((n, n)) / n
    B = -0.5 * (one_n @ d2 @ one_n - one_n @ d2 - d2 @ one_n + d2 @ one_n)
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # Take top 2 eigenvalues (largest)
    idx = np.argsort(eigenvalues)[::-1]
    idx = idx[:2]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Ensure non-negative eigenvalues
    eigenvalues = np.maximum(eigenvalues, 0)
    coords = eigenvectors * np.sqrt(eigenvalues)
    return coords


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a geometric heuristic that combines:
    1. Angle-based smoothness (minimize deviation from direction toward destination)
    2. Distance to candidate
    3. A penalty for nodes that would create long 'spikes' back to the destination
    
    This uses recovered 2D coordinates to compute angles via cross/dot products.
    """
    global _cache
    
    # Deadline guard
    if _cache['start_time'] is None:
        _cache['start_time'] = time.time()
        _cache['deadline'] = _cache['start_time'] + 115.0  # 5s safety margin
    
    # Check for new instance
    dm_hash = hash(distance_matrix.tobytes())
    if dm_hash != _cache['dm_hash']:
        _cache['dm_hash'] = dm_hash
        _cache['dm'] = distance_matrix
        _cache['n'] = distance_matrix.shape[0]
        _cache['coords'] = _recover_coords(distance_matrix)
        _cache['start_time'] = time.time()
        _cache['deadline'] = _cache['start_time'] + 115.0
    
    coords = _cache['coords']
    dm = _cache['dm']
    
    if len(unvisited_nodes) == 0:
        return -1
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Time check
    if time.time() > _cache['deadline'] - 1.0:
        # Emergency: just return nearest
        return int(unvisited_nodes[np.argmin(dm[current_node, unvisited_nodes])])
    
    # Compute vectors
    cur_pos = coords[current_node]
    dest_pos = coords[destination_node]
    cand_pos = coords[unvisited_nodes]
    
    # Vector from current to destination
    cur_to_dest = dest_pos - cur_pos
    # Vectors from current to each candidate
    cur_to_cand = cand_pos - cur_pos
    
    # Compute angles using cross and dot products (2D)
    # cross = ax*by - ay*bx
    cross = cur_to_dest[0] * cur_to_cand[1] - cur_to_dest[1] * cur_to_cand[0]
    dot = cur_to_dest[0] * cur_to_cand[0] + cur_to_dest[1] * cur_to_cand[1]
    
    # Magnitudes
    cur_to_dest_mag = np.linalg.norm(cur_to_dest)
    cur_to_cand_mag = np.linalg.norm(cur_to_cand, axis=1)
    
    # Avoid division by zero
    cur_to_cand_mag_safe = np.maximum(cur_to_cand_mag, 1e-10)
    cur_to_dest_mag_safe = max(cur_to_dest_mag, 1e-10)
    
    # sin of angle = |cross| / (|a||b|)
    sin_angle = np.abs(cross) / (cur_to_dest_mag_safe * cur_to_cand_mag_safe)
    
    # cos of angle = dot / (|a||b|)
    cos_angle = dot / (cur_to_dest_mag_safe * cur_to_cand_mag_safe)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Angle in [0, pi]
    angle = np.arccos(cos_angle)
    
    # Distance from current to candidate
    dist_cur_cand = dm[current_node, unvisited_nodes]
    
    # Distance from candidate to destination
    dist_cand_dest = dm[unvisited_nodes, destination_node]
    
    # Heuristic score:
    # - Primary: distance from current to candidate (we want short steps)
    # - Secondary: angle deviation (we want to head toward destination)
    # - Tertiary: avoid candidates that are far from destination (spike penalty)
    
    # Normalize components for weighting
    # Scale distances by typical scale
    dist_scale = np.median(dist_cur_cand) if len(dist_cur_cand) > 0 else 1.0
    dist_scale = max(dist_scale, 1e-10)
    
    # Score = w1 * (dist_cur_cand / scale) + w2 * angle + w3 * (dist_cand_dest / scale)
    # We want to minimize this.
    
    w1 = 1.0   # distance weight
    w2 = 0.8   # angle weight (angle is in [0, pi], so max ~3.14)
    w3 = 0.5   # spike penalty weight
    
    score = w1 * (dist_cur_cand / dist_scale) + w2 * angle + w3 * (dist_cand_dest / dist_scale)
    
    # Pick the candidate with minimum score
    best_idx = np.argmin(score)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
