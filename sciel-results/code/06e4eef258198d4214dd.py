"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute priority scores for vertices using a refined spectral approach.
    Uses the Fiedler vector with damped fixed-point iteration, adaptive
    spectral weighting, and local neighborhood refinement.
    
    Improvements:
    1. Damped fixed-point iteration for smoother convergence
    2. Adaptive spectral weighting based on score distribution
    3. Local neighborhood refinement penalizing contested vertices
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Compute degree
    degree = A.sum(axis=1)
    
    # Compute Laplacian: L = D - A
    D = np.diag(degree)
    L = D - A
    
    try:
        # Get eigenvalues and eigenvectors of the Laplacian
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Fiedler vector (second smallest eigenvalue)
        fiedler_vec = eigenvectors[:, 1]
        fiedler_mag = np.abs(fiedler_vec)
        
        # Normalize to [0, 1]
        fmin, fmax = fiedler_mag.min(), fiedler_mag.max()
        if fmax > fmin + 1e-12:
            fiedler_norm = (fiedler_mag - fmin) / (fmax - fmin)
        else:
            fiedler_norm = np.zeros(n)
        
        # Normalize degree to [0, 1]
        dmin, dmax = degree.min(), degree.max()
        if dmax > dmin + 1e-12:
            degree_norm = (degree - dmin) / (dmax - dmin)
        else:
            degree_norm = np.zeros(n)
        
        # Base scores: blend of low-degree preference and peripherality
        alpha = 0.5
        beta = 0.5
        base_scores = alpha * (1.0 - degree_norm) + beta * fiedler_norm
        
        # Damped fixed-point iteration for smoothing
        # Iterate to find a stable safety score
        scores = base_scores.copy()
        damping = 0.7
        max_iter = 50
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Compute neighbor influence: average score of neighbors
            # A @ scores gives sum of neighbor scores, divide by degree
            neighbor_sum = A @ scores
            neighbor_avg = np.where(degree > 0, neighbor_sum / np.maximum(degree, 1), 0)
            
            # Damped update: move toward a smoothed version
            # Vertices with high neighbor scores are "contested" and should be penalized
            # New score = damping * base + (1-damping) * (1 - neighbor_avg)
            new_scores = damping * base_scores + (1 - damping) * (1.0 - neighbor_avg)
            
            # Check convergence
            if np.max(np.abs(new_scores - scores)) < 1e-6:
                scores = new_scores
                break
            scores = new_scores
        
        # Adaptive spectral weighting
        # If base scores are similar, increase weight on Fiedler signal
        score_std = np.std(base_scores)
        if score_std < 0.1:
            # Scores are similar; boost Fiedler component
            adaptive_beta = 0.6
            adaptive_alpha = 0.4
            scores = adaptive_alpha * (1.0 - degree_norm) + adaptive_beta * fiedler_norm
            # Re-apply damped iteration
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                neighbor_sum = A @ scores
                neighbor_avg = np.where(degree > 0, neighbor_sum / np.maximum(degree, 1), 0)
                new_scores = damping * (adaptive_alpha * (1.0 - degree_norm) + adaptive_beta * fiedler_norm) + (1 - damping) * (1.0 - neighbor_avg)
                if np.max(np.abs(new_scores - scores)) < 1e-6:
                    scores = new_scores
                    break
                scores = new_scores
        
        # Local neighborhood refinement
        # Penalize vertices adjacent to high-scoring neighbors (more contested)
        if time.time() < deadline:
            # Compute contestedness: how many high-scoring neighbors each vertex has
            threshold = np.mean(scores) + np.std(scores)
            high_score_mask = scores > threshold
            contested = A @ high_score_mask.astype(np.float64)
            
            # Apply penalty: reduce score for vertices with many high-scoring neighbors
            # Normalize contested to [0, 1]
            cmax = contested.max()
            if cmax > 1e-12:
                contested_norm = contested / cmax
                scores = scores - 0.2 * contested_norm
            
            # Ensure scores are bounded
            smin, smax = scores.min(), scores.max()
            if smax > smin + 1e-12:
                scores = (scores - smin) / (smax - smin)
        
        return scores.astype(np.float64)
        
    except Exception:
        # Fallback to degree-based if spectral fails
        return -degree.astype(np.float64)
# EVOLVE-BLOCK-END
