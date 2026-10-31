"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to persist across calls within a single tour construction
_state = {
    'start_node': None,
    'visited': None,
    'coords': None,
    'deadline': None,
    'tour': None,
}


def _reset_state(start_node, distance_matrix):
    _state['start_node'] = start_node
    _state['visited'] = set()
    _state['visited'].add(start_node)
    _state['coords'] = None
    _state['deadline'] = time.time() + 115.0  # Safety margin before 120s limit
    _state['tour'] = [start_node]


def _get_coords(distance_matrix):
    """Recover approximate 2D coordinates from distance matrix using MDS-like approach."""
    if _state['coords'] is not None:
        return _state['coords']
    
    n = distance_matrix.shape[0]
    # For small n, we can do a simple embedding
    # Use double centering with the first few points as reference
    # This is a heuristic approximation of Euclidean coordinates
    
    # Simple approach: pick 3 reference points and solve for coordinates
    if n < 3:
        _state['coords'] = np.zeros((n, 2))
        return _state['coords']
    
    # Pick reference points: 0, 1, 2 (or available)
    ref_indices = [0, 1, 2] if n >= 3 else list(range(n))
    
    # Place reference 0 at origin, reference 1 on x-axis
    coords = np.zeros((n, 2))
    d01 = distance_matrix[ref_indices[0], ref_indices[1]]
    coords[ref_indices[1], 0] = d01
    
    if n >= 3:
        d02 = distance_matrix[ref_indices[0], ref_indices[2]]
        d12 = distance_matrix[ref_indices[1], ref_indices[2]]
        # Solve for x2, y2
        x2 = (d02**2 - d12**2 + d01**2) / (2 * d01) if d01 > 1e-10 else 0.0
        y2_sq = d02**2 - x2**2
        y2 = np.sqrt(max(y2_sq, 0.0))
        coords[ref_indices[2], 0] = x2
        coords[ref_indices[2], 1] = y2
    
    # For remaining points, solve using distances to reference points
    for i in range(n):
        if i in ref_indices:
            continue
        d0i = distance_matrix[ref_indices[0], i]
        d1i = distance_matrix[ref_indices[1], i]
        # Solve for xi, yi using distances to ref 0 and ref 1
        xi = (d0i**2 - d1i**2 + d01**2) / (2 * d01) if d01 > 1e-10 else 0.0
        yi_sq = d0i**2 - xi**2
        yi = np.sqrt(max(yi_sq, 0.0))
        
        # Determine sign of yi using distance to ref 2 if available
        if n >= 3:
            d2i = distance_matrix[ref_indices[2], i]
            # Check which sign gives better fit to d2i
            xi_candidate = xi
            yi_positive = yi
            yi_negative = -yi
            
            dist_pos = np.sqrt((xi_candidate - coords[ref_indices[2], 0])**2 + 
                              (yi_positive - coords[ref_indices[2], 1])**2)
            dist_neg = np.sqrt((xi_candidate - coords[ref_indices[2], 0])**2 + 
                              (yi_negative - coords[ref_indices[2], 1])**2)
            
            if abs(dist_pos - d2i) < abs(dist_neg - d2i):
                yi = yi_positive
            else:
                yi = yi_negative
        
        coords[i, 0] = xi
        coords[i, 1] = yi
    
    _state['coords'] = coords
    return _state['coords']


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Initialize state if this is the first call
    if _state['start_node'] is None:
        _reset_state(current_node, distance_matrix)
    
    # Check deadline
    if time.time() > _state['deadline']:
        # Fallback to nearest neighbor if over time
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Get coordinates
    coords = _get_coords(distance_matrix)
    
    current_pos = coords[current_node]
    dest_pos = coords[destination_node]
    
    # Compute alignment score: how well each candidate lies on the path toward destination
    # score = dist(current, candidate) + dist(candidate, dest) - dist(current, dest)
    # Lower score means better alignment (closer to straight line)
    
    cand_indices = unvisited_nodes
    if len(cand_indices) == 0:
        return int(current_node)  # Should not happen
    
    # Compute distances from current to all candidates
    dist_current_to_cand = distance_matrix[current_node, cand_indices]
    
    # Compute distances from candidates to destination
    dist_cand_to_dest = distance_matrix[cand_indices, destination_node]
    
    # Distance from current to destination
    dist_current_to_dest = distance_matrix[current_node, destination_node]
    
    # Alignment score (lower is better - indicates straighter path)
    alignment_scores = dist_current_to_cand + dist_cand_to_dest - dist_current_to_dest
    
    # Also consider the direct distance to destination as a tiebreaker
    # Combine: primarily alignment, secondarily distance to destination
    # Weight: 70% alignment, 30% distance to destination
    max_align = np.max(alignment_scores) if len(alignment_scores) > 0 else 1.0
    max_dest_dist = np.max(dist_cand_to_dest) if len(dist_cand_to_dest) > 0 else 1.0
    
    # Normalize
    align_norm = alignment_scores / max_align if max_align > 1e-10 else alignment_scores
    dest_norm = dist_cand_to_dest / max_dest_dist if max_dest_dist > 1e-10 else dist_cand_to_dest
    
    combined_score = 0.7 * align_norm + 0.3 * dest_norm
    
    # Select the candidate with minimum combined score
    best_idx = np.argmin(combined_score)
    next_node = int(cand_indices[best_idx])
    
    # Track visited
    _state['visited'].add(next_node)
    _state['tour'].append(next_node)
    
    return next_node
# EVOLVE-BLOCK-END
