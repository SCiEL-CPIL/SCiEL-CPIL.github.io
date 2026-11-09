"""Combined constructive TSP heuristic: phase-aware NN with lookahead and destination bias."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guarding
_T0 = None
_DEADLINE = 145.0  # Safety margin before 150s kill
_SAFETY = 5.0

def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return time.time() - _T0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in a TSP tour.
    
    Strategy:
    - Phase-aware blend of nearest-neighbor, lookahead (min distance to other unvisited),
      and destination-proximity.
    - Early phase: pure nearest-neighbor.
    - Mid phase: blend in lookahead to avoid isolated nodes.
    - Late phase: blend in destination proximity to avoid long closing edge.
    
    We estimate progress by the number of unvisited nodes remaining.
    With 50 cities total, progress = (50 - n_unvisited) / 49.
    """
    unvisited_nodes = np.asarray(unvisited_nodes)
    n_unvisited = unvisited_nodes.size
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Hard deadline guard: if we're close to the limit, use simple NN
    if _elapsed() > _DEADLINE - _SAFETY:
        try:
            d = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[np.argmin(d)])
        except Exception:
            return int(unvisited_nodes[0])
    
    try:
        # 1. Immediate distances from current node to all unvisited
        d_cur = distance_matrix[current_node, unvisited_nodes]
        
        # 2. Distances from all unvisited to destination
        d_dest = distance_matrix[unvisited_nodes, destination_node]
        
        # 3. Estimate future local cost: min distance from each candidate to other unvisited nodes
        # This is O(N^2) but N is small (<=50).
        D_sub = distance_matrix[unvisited_nodes, unvisited_nodes].copy()
        np.fill_diagonal(D_sub, np.inf)
        d_min_next = np.min(D_sub, axis=1)
        
        # 4. Phase-aware weighting
        # Total nodes is typically 50. We estimate progress based on remaining nodes.
        N_total = distance_matrix.shape[0]
        if N_total <= 1:
            return int(unvisited_nodes[0])
        
        # Progress: 0 at start, 1 at end
        progress = (N_total - n_unvisited) / max(N_total - 1, 1)
        progress = float(np.clip(progress, 0.0, 1.0))
        
        # Weights:
        # - lambda_: weight for lookahead local cost. Start at 0.3, increase to 0.6.
        # - mu: weight for destination. Should be 0 early, 1 late.
        
        # Lambda increases linearly with progress
        lambda_ = 0.3 + 0.3 * progress
        
        # Mu increases quadratically to keep destination bias low until late
        mu = progress ** 2
        
        # Calculate score
        # Score = d_cur + lambda * d_min_next + mu * d_dest
        # We need to ensure scales are comparable. d_cur, d_min_next, d_dest are all Euclidean distances.
        # So direct addition is valid.
        
        score = d_cur + lambda_ * d_min_next + mu * d_dest
        
        best_idx = int(np.argmin(score))
        
        return int(unvisited_nodes[best_idx])
        
    except Exception:
        # Fallback to simple NN
        try:
            d = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[np.argmin(d)])
        except Exception:
            return int(unvisited_nodes[0])
# EVOLVE-BLOCK-END
