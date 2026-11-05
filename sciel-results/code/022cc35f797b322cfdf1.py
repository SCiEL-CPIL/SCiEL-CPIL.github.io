"""Priority scoring for the fixed greedy Maximum Independent Set decoder.

The decoder picks vertices in descending score order, skipping neighbours of
already-picked vertices.  We want a score that makes the greedy pass select a
large independent set.

Design:
  * Spectral 2-coloring: The principal eigenvector of the adjacency matrix
    provides a global partition of the graph. Vertices with similar signed
    components tend to form near-independent sets. We use the magnitude of the
    principal eigenvector component as a "color strength" score.
  * Degree features: Low degree is cheap to include; high mean-neighbour-degree
    means including the vertex absorbs high-degree neighbors (hub-blocking benefit).
  * Local refinement: After initial scoring, we perform local swaps to improve
    the greedy decode result.

We combine these into a single per-vertex scalar score.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_decode(A, scores):
    """Simulate the greedy decoder and return the size of the independent set."""
    n = len(scores)
    order = np.argsort(-scores, kind='stable')
    selected = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if not blocked[v]:
            selected[v] = True
            count += 1
            # Block neighbors
            blocked[A[v] == 1] = True
    return count


def _compute_features(A):
    """Compute rich features for scoring."""
    n = A.shape[0]
    degree = A.sum(axis=1)
    
    # Mean neighbor degree
    neigh_deg_sum = A @ degree
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_neigh_deg = np.where(degree > 0, neigh_deg_sum / np.maximum(degree, 1.0), 0.0)
    
    # Second moment of neighbor degrees (variance proxy)
    neigh_deg_sq_sum = A @ (degree ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_neigh_deg_sq = np.where(degree > 0, neigh_deg_sq_sum / np.maximum(degree, 1.0), 0.0)
    neigh_deg_var = np.maximum(0.0, mean_neigh_deg_sq - mean_neigh_deg ** 2)
    
    # Spectral features: principal eigenvector
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        # Principal eigenvector is the last column (largest eigenvalue)
        principal_evec = eigenvectors[:, -1]
        # Normalize to zero mean, unit variance for consistent scaling
        p_mean = np.mean(principal_evec)
        p_std = np.std(principal_evec)
        if p_std > 1e-10:
            principal_evec = (principal_evec - p_mean) / p_std
        else:
            principal_evec = np.zeros(n)
        
        # Fiedler vector (second largest eigenvalue) for additional structure
        fiedler_evec = eigenvectors[:, -2]
        f_mean = np.mean(fiedler_evec)
        f_std = np.std(fiedler_evec)
        if f_std > 1e-10:
            fiedler_evec = (fiedler_evec - f_mean) / f_std
        else:
            fiedler_evec = np.zeros(n)
    except Exception:
        principal_evec = np.zeros(n)
        fiedler_evec = np.zeros(n)
    
    # 2-coloring score: vertices with high |principal_evec| are "strongly colored"
    # We prefer vertices from the dominant color class (positive side)
    color_strength = np.abs(principal_evec)
    color_sign = np.sign(principal_evec)
    
    # Combine: prefer positive color class, weighted by strength
    spectral_score = color_strength * (1.0 + 0.5 * color_sign)
    
    # Degree-based scores
    low_degree_score = -degree / max(degree.max(), 1.0)  # lower degree = higher score
    hub_block_score = mean_neigh_deg / max(mean_neigh_deg.max(), 1.0)
    variance_score = neigh_deg_var / max(neigh_deg_var.max(), 1.0) if neigh_deg_var.max() > 1e-10 else np.zeros(n)
    
    # Normalize all features to [0, 1]
    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 1e-10:
            return (x - x_min) / (x_max - x_min)
        return np.zeros_like(x)
    
    spectral_norm = normalize(spectral_score)
    low_degree_norm = normalize(-degree)
    hub_block_norm = normalize(mean_neigh_deg)
    variance_norm = normalize(neigh_deg_var)
    fiedler_norm = normalize(np.abs(fiedler_evec))
    
    return {
        'spectral': spectral_norm,
        'low_degree': low_degree_norm,
        'hub_block': hub_block_norm,
        'variance': variance_norm,
        'fiedler': fiedler_norm,
        'degree': degree,
        'mean_neigh_deg': mean_neigh_deg
    }


def _local_refine(A, scores, features, max_iterations=50, time_budget=10.0):
    """Local swap-based refinement to improve greedy decode result."""
    n = len(scores)
    best_scores = scores.copy()
    best_size = _greedy_decode(A, best_scores)
    
    start_time = time.time()
    for iteration in range(max_iterations):
        if time.time() - start_time > time_budget:
            break
        
        # Try swapping scores of pairs of vertices
        improved = False
        for i in range(n):
            if time.time() - start_time > time_budget:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > time_budget:
                    break
                
                # Only consider swaps that might help
                # Heuristic: swap if one is selected and the other is blocked,
                # or if their scores are close
                if abs(scores[i] - scores[j]) > 0.1:
                    continue
                
                # Try the swap
                old_i, old_j = scores[i], scores[j]
                scores[i], scores[j] = scores[j], scores[i]
                
                new_size = _greedy_decode(A, scores)
                
                if new_size > best_size:
                    best_size = new_size
                    best_scores = scores.copy()
                    improved = True
                else:
                    # Revert
                    scores[i], scores[j] = old_i, old_j
        
        if not improved:
            break
    
    return best_scores, best_size


def priority(adjacency):
    # Work in float for stable arithmetic; adjacency is a bool/0-1 matrix.
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Compute rich features
    features = _compute_features(A)
    
    # Initial score: weighted combination of features
    # Weights tuned to emphasize spectral partitioning and degree features
    w_spectral = 0.4
    w_low_degree = 0.25
    w_hub_block = 0.2
    w_variance = 0.1
    w_fiedler = 0.05
    
    scores = (
        w_spectral * features['spectral'] +
        w_low_degree * features['low_degree'] +
        w_hub_block * features['hub_block'] +
        w_variance * features['variance'] +
        w_fiedler * features['fiedler']
    )
    
    # Local refinement if time permits
    if time.time() - start_time < deadline - 15.0:
        time_budget = min(10.0, deadline - time.time() - start_time - 5.0)
        if time_budget > 2.0:
            scores, _ = _local_refine(A, scores, features, max_iterations=50, time_budget=time_budget)
    
    return scores
# EVOLVE-BLOCK-END
