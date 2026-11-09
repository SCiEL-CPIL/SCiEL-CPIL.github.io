"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, QhullError


def _recover_coords_2d(D):
    """Recover 2D coordinates from a Euclidean distance matrix using classical MDS."""
    n = D.shape[0]
    # Center the squared distance matrix
    sq = D ** 2
    # Double-centering: B = -1/2 * J * sq * J
    ones = np.ones((n, n))
    J = np.eye(n) - ones / n
    B = -0.5 * J @ sq @ J
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Take top 2 eigenvalues (should be non-negative for Euclidean distances)
    eigvals = np.maximum(eigvals[:2], 0)
    coords = eigvecs[:, :2] * np.sqrt(eigvals)
    return coords


def heuristics(distance_matrix):
    """
    Compute a heuristic edge-attractiveness matrix for ACO on TSP.
    
    Approach: Delaunay-based topological heuristic with refined adaptive boost
    - Recover 2D coordinates via classical MDS from the distance matrix.
    - Compute Delaunay triangulation to identify natural short, non-crossing edges.
    - Boost Delaunay edges with a stronger adaptive factor for short edges.
    - Apply a more aggressive smooth penalty for long non-Delaunay edges.
    - Combine with a smooth distance-based attractiveness component.
    
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
    # H_base = exp(-d / (sigma * med_d)) gives smooth preference for short edges
    sigma = 0.3
    H_base = np.exp(-D / (sigma * med_d))
    H_base = np.where(off_diag, H_base, 0.0)
    
    # Try to recover coordinates and compute Delaunay
    try:
        coords = _recover_coords_2d(D)
        tri = Delaunay(coords)
        
        # Extract edges from Delaunay triangulation
        is_delaunay = np.zeros((n, n), dtype=bool)
        delaunay_lengths = []
        for simplex in tri.simplices:
            u, v, w = simplex
            is_delaunay[u, v] = True
            is_delaunay[v, u] = True
            is_delaunay[v, w] = True
            is_delaunay[w, v] = True
            is_delaunay[w, u] = True
            is_delaunay[u, w] = True
            delaunay_lengths.append(D[u, v])
            delaunay_lengths.append(D[v, w])
            delaunay_lengths.append(D[w, u])
        
        delaunay_lengths = np.array(delaunay_lengths)
        median_dl = np.median(delaunay_lengths) if len(delaunay_lengths) > 0 else med_d
        if median_dl <= 0:
            median_dl = med_d
        
        # Stronger adaptive boost for Delaunay edges
        # Boost factor = base_boost * (median_dl / edge_length)^alpha
        alpha = 0.8
        base_boost = 20.0
        max_boost = 200.0
        
        dl_boost = np.ones((n, n))
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = median_dl / (D + 1e-9)
            boost_vals = base_boost * (ratio ** alpha)
        boost_vals = np.clip(boost_vals, 1.0, max_boost)
        dl_boost = np.where(is_delaunay & off_diag, boost_vals, 1.0)
        
        H_base *= dl_boost
        
        # Aggressive penalty for long non-Delaunay edges
        # Use a lower threshold and sharper falloff
        long_threshold = 2.5 * median_dl
        penalty = np.ones((n, n))
        long_mask = (~is_delaunay) & off_diag & (D > long_threshold)
        with np.errstate(divide='ignore', invalid='ignore'):
            falloff = 1.0 / (1.0 + (D / long_threshold) ** 3)
        penalty = np.where(long_mask, falloff, 1.0)
        
        H_base *= penalty
        
        # Additional boost for very short Delaunay edges (nearest-neighbor-like)
        very_short_mask = is_delaunay & off_diag & (D < 0.5 * median_dl)
        H_base[very_short_mask] *= 1.5
        
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
