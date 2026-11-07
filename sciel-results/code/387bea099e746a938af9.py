"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix (larger is more attractive) of the same shape.
    We use a combination of:
      1) Inverse distance (strong local preference)
      2) Degree-based adjustment: edges that connect to nodes with few nearby neighbors
         are slightly boosted to avoid trapping ants in dense clusters.
      3) A soft normalization to keep values in a reasonable range.
    """
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    
    # Ensure no zeros cause division issues
    safe_dm = np.where(dm < 1e-12, 1e-9, dm)
    
    # Base heuristic: inverse distance
    base = 1.0 / safe_dm
    
    # Compute degree-like measure: for each node, how many other nodes are 'close'
    # Use a threshold relative to median distance
    median_dist = np.median(dm)
    close_mask = dm < 1.5 * median_dist
    close_count = np.sum(close_mask, axis=1)  # number of close neighbors per node
    
    # Normalize close_count to [0, 1]
    if close_count.max() > 0:
        close_norm = close_count / close_count.max()
    else:
        close_norm = np.zeros(n)
    
    # Boost edges going to nodes with fewer close neighbors (to spread out)
    # Factor: 1 + alpha * (1 - close_norm[j])
    alpha = 0.5
    boost = 1.0 + alpha * (1.0 - close_norm[np.newaxis, :])
    
    # Also add a small symmetric boost for both directions
    boost_sym = 1.0 + alpha * (1.0 - close_norm[:, np.newaxis])
    boost_combined = boost * boost_sym  # element-wise product gives per-edge boost
    
    heuristic = base * boost_combined
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Soft normalization: divide by mean to keep scale reasonable
    mean_h = np.mean(heuristic)
    if mean_h > 0 and np.isfinite(mean_h):
        heuristic = heuristic / mean_h
    
    return heuristic
# EVOLVE-BLOCK-END
