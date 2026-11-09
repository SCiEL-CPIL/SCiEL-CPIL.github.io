"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, QhullError


def _recover_coords_2d(D):
    """Recover 2D coordinates from a Euclidean distance matrix using classical MDS."""
    n = D.shape[0]
    sq = D ** 2
    ones = np.ones((n, n))
    J = np.eye(n) - ones / n
    B = -0.5 * J @ sq @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvals = np.maximum(eigvals[:2], 0)
    coords = eigvecs[:, :2] * np.sqrt(eigvals)
    return coords


def _build_voronoi_adjacency(coords):
    """
    Build adjacency information from Voronoi diagram.
    Returns:
        same_region: (n, n) bool matrix, True if two points share a Voronoi edge
        (i.e., their Voronoi cells are adjacent)
    """
    n = len(coords)
    same_region = np.zeros((n, n), dtype=bool)
    
    try:
        vor = Voronoi(coords)
        
        # Map each ridge to the two points whose Voronoi cells it separates
        for ridge in vor.ridges:
            p1_idx, p2_idx = ridge[0], ridge[1]
            # -1 means the Voronoi cell is unbounded; skip those
            if p1_idx >= 0 and p2_idx >= 0:
                same_region[p1_idx, p2_idx] = True
                same_region[p2_idx, p1_idx] = True
        
    except (QhullError, Exception):
        pass
    
    return same_region


def heuristics(distance_matrix):
    """
    Compute a heuristic edge-attractiveness matrix for ACO on TSP.
    
    Approach: Voronoi-based topological heuristic
    - Recover 2D coordinates via classical MDS from the distance matrix.
    - Compute Voronoi diagram to identify spatially adjacent regions.
    - Edges between points with adjacent Voronoi cells (shared Voronoi edge)
      are considered "local" and get moderate attractiveness.
    - Edges between points with non-adjacent Voronoi cells (disjoint regions)
      that are relatively short get boosted as "bridge" edges.
    - Long edges within the same local cluster get penalized.
    - Base attractiveness is smooth inverse distance.
    
    Returns: matrix of same shape as distance_matrix, larger = more attractive
    """
    n = distance_matrix.shape[0]
    if n < 2:
        return np.ones_like(distance_matrix)
    
    D = distance_matrix
    off_diag = ~np.eye(n, dtype=bool)
    med_d = np.median(D[off_diag]) if np.any(off_diag) else 1.0
    if med_d <= 0:
        med_d = 1.0
    
    # Base heuristic: smooth inverse distance with exponential decay
    sigma = 0.35
    H_base = np.exp(-D / (sigma * med_d))
    H_base = np.where(off_diag, H_base, 0.0)
    
    try:
        coords = _recover_coords_2d(D)
        same_region = _build_voronoi_adjacency(coords)
        
        # Check if Voronoi gave us useful info
        if not np.any(same_region):
            raise ValueError("No Voronoi adjacency found")
        
        # Classify edges:
        # 1. Voronoi-adjacent (same local region): moderate boost, short edges preferred
        # 2. Non-adjacent but short: bridge edges, high boost
        # 3. Non-adjacent and long: penalize
        
        # Voronoi-adjacent edges: boost short ones
        vor_adj_mask = same_region & off_diag
        # For Voronoi-adjacent, use a stronger inverse distance
        # Boost factor based on how much shorter than median
        vor_boost = np.ones((n, n))
        with np.errstate(divide='ignore', invalid='ignore'):
            # Shorter Voronoi-adjacent edges get higher boost
            vor_ratio = med_d / (D + 1e-9)
            vor_boost_vals = 5.0 * (vor_ratio ** 0.6)
        vor_boost_vals = np.clip(vor_boost_vals, 1.0, 50.0)
        vor_boost = np.where(vor_adj_mask, vor_boost_vals, 1.0)
        H_base *= vor_boost
        
        # Bridge edges: non-Voronoi-adjacent but relatively short
        # These connect different local clusters and are important for TSP
        non_vor_mask = (~same_region) & off_diag
        # Threshold: edges shorter than ~1.5x median are potential bridges
        bridge_threshold = 1.5 * med_d
        bridge_mask = non_vor_mask & (D < bridge_threshold)
        
        # Boost bridge edges significantly
        bridge_boost = np.ones((n, n))
        with np.errstate(divide='ignore', invalid='ignore'):
            # The shorter the bridge (relative to threshold), the more attractive
            bridge_ratio = bridge_threshold / (D + 1e-9)
            bridge_boost_vals = 15.0 * (bridge_ratio ** 0.8)
        bridge_boost_vals = np.clip(bridge_boost_vals, 1.0, 100.0)
        bridge_boost = np.where(bridge_mask, bridge_boost_vals, 1.0)
        H_base *= bridge_boost
        
        # Penalize long non-adjacent edges (likely crossing edges)
        long_non_vor_mask = non_vor_mask & (D > 2.0 * med_d)
        with np.errstate(divide='ignore', invalid='ignore'):
            long_falloff = 1.0 / (1.0 + (D / (2.0 * med_d)) ** 3)
        long_penalty = np.where(long_non_vor_mask, long_falloff, 1.0)
        H_base *= long_penalty
        
        # Additional: penalize very long Voronoi-adjacent edges (shouldn't happen much)
        long_vor_mask = vor_adj_mask & (D > 1.8 * med_d)
        with np.errstate(divide='ignore', invalid='ignore'):
            long_vor_falloff = 1.0 / (1.0 + (D / (1.8 * med_d)) ** 2)
        long_vor_penalty = np.where(long_vor_mask, long_vor_falloff, 1.0)
        H_base *= long_vor_penalty
        
    except (QhullError, Exception):
        # Fallback: smooth distance-based with long-edge penalty
        long_mask = off_diag & (D > 3.0 * med_d)
        with np.errstate(divide='ignore', invalid='ignore'):
            falloff = 1.0 / (1.0 + (D / (3.0 * med_d)) ** 3)
        H_base *= np.where(long_mask, falloff, 1.0)
    
    # Zero out diagonal
    np.fill_diagonal(H_base, 0.0)
    
    # Ensure positive values (off-diagonal should be positive)
    H_base = np.where(off_diag, np.maximum(H_base, 1e-9), 0.0)
    
    # Symmetrize
    H_base = (H_base + H_base.T) / 2
    
    return H_base
# EVOLVE-BLOCK-END
