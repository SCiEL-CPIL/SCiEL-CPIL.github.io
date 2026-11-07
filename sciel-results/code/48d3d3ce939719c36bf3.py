"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right
    after item i in the same bin. Larger is more attractive.
    
    Uses a spectral embedding of the item compatibility graph where edges are
    weighted by the negative residual space (tighter fits are more attractive).
    Additionally incorporates a bias towards placing larger items in tight fits
    (First Fit Decreasing principle).
    """
    n = len(demands)
    demands = np.asarray(demands, dtype=np.float64)
    
    # Build compatibility matrix: item j fits after item i if d_i + d_j <= capacity
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    
    # Residual space: how much room is left after placing both items
    # Smaller residual = tighter fit = more attractive
    residual = capacity - pair_sum  # >= 0 where fits
    
    # Edge weight for compatibility graph: prefer tight fits
    # Use negative residual so tighter fits have larger weights
    # Only valid where fits is True
    edge_weight = np.where(fits, -residual, 0.0)
    
    # Build adjacency matrix for spectral embedding
    # Normalize edge weights to [0, 1] range for stability
    max_w = np.max(edge_weight)
    if max_w > 0:
        adj = edge_weight / max_w
    else:
        adj = np.zeros_like(edge_weight)
    
    # Zero out diagonal (self-edges)
    np.fill_diagonal(adj, 0.0)
    
    # Compute spectral embedding using top eigenvectors
    # Use symmetric matrix for real eigenvalues
    adj_sym = (adj + adj.T) / 2.0
    
    # Get eigenvalues and eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(adj_sym)
    except np.linalg.LinAlgError:
        # Fallback to simple heuristic
        pair = demands[:, None] + demands[None, :]
        return np.where(pair <= capacity, pair / capacity, 1e-9)
    
    # Use the top k eigenvectors (k=4 for a 120-item problem)
    k = min(4, n)
    # eigvecs columns are eigenvectors, sorted by ascending eigenvalue
    # We want the largest eigenvalues, so take from the end
    top_eigvecs = eigvecs[:, -k:]
    
    # Embedding: each item gets a k-dimensional vector
    embedding = top_eigvecs  # shape (n, k)
    
    # Compute pairwise distances in embedding space
    # Items that are close in embedding space are more compatible
    # Use squared Euclidean distance
    diff = embedding[:, None, :] - embedding[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Convert distance to similarity: smaller distance = higher similarity
    # Use exponential decay
    sigma = np.median(dist_sq[dist_sq > 0]) if np.any(dist_sq > 0) else 1.0
    similarity = np.exp(-dist_sq / (2 * sigma))
    
    # Combine spectral similarity with local gap metric
    # Local gap: prefer tight fits (smaller residual)
    gap_score = np.where(fits, 1.0 / (1.0 + residual), 0.0)
    
    # Bias towards larger items in tight fits (FFD principle)
    # Scale by the size of item j (the item being placed)
    size_bias = demands[None, :] / capacity  # shape (1, n), broadcast to (n, n)
    
    # Combine all components
    # spectral_similarity * gap_score * size_bias
    # Ensure positivity
    heuristic = similarity * gap_score * size_bias
    
    # Where items don't fit, set to 1e-9
    heuristic = np.where(fits, heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
