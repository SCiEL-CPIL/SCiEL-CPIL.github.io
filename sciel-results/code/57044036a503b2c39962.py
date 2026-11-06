import numpy as np
import time

def priority(adjacency):
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s kill

    n = adjacency.shape[0]
    A = np.asarray(adjacency, dtype=bool)
    A_float = A.astype(float)

    deg = A.sum(axis=1).astype(float)
    has_edge = deg > 0

    # Isolated vertices: always safe to pick -> top priority
    BIG = 1e9

    # Initialize with inverse degree
    s = 1.0 / (deg + 1.0).copy()
    s[~has_edge] = BIG

    # Fixed-point iteration: s[i] = 1 / (1 + sum_{j in nbr(i)} s[j])
    # This is a smooth surrogate for the "hard" constraint that neighbors can't be picked.
    # We use adaptive damping.
    
    # Pre-compute A_float for speed
    # Phase 1: Fast convergence
    for it in range(100):
        if time.time() > deadline - 20.0:
            break
        alpha = 0.7
        nbr_sum = A_float @ s
        new_s = 1.0 / (1.0 + nbr_sum)
        new_s[~has_edge] = BIG
        damped_s = (1 - alpha) * s + alpha * new_s
        damped_s[~has_edge] = BIG
        
        # Check convergence
        if np.allclose(damped_s, s, rtol=1e-5, atol=1e-7):
            s = damped_s
            break
        s = damped_s

    # Phase 2: High precision polish
    for it in range(50):
        if time.time() > deadline - 10.0:
            break
        alpha = 0.1
        nbr_sum = A_float @ s
        new_s = 1.0 / (1.0 + nbr_sum)
        new_s[~has_edge] = BIG
        damped_s = (1 - alpha) * s + alpha * new_s
        damped_s[~has_edge] = BIG
        
        if np.allclose(damped_s, s, rtol=1e-8, atol=1e-10):
            s = damped_s
            break
        s = damped_s

    # Compute Clustering Coefficients
    # c_i = (number of edges between neighbors of i) / (deg_i * (deg_i - 1) / 2)
    # We can approximate this or compute it exactly for n=200.
    # A^2 gives number of paths of length 2.
    # For each i, edges between neighbors = (A @ A @ A)[i, i] / 2 ? No.
    # (A^2)[i, j] is number of common neighbors.
    # Triangles containing i = sum_{j in N(i)} (A[i, j] * (A^2)[i, j]) / 2 ?
    # Actually, number of triangles involving i is 0.5 * sum_{j} A[i,j] * (A^2)[i,j]
    # Wait, (A^2)[i,j] is number of k such that A[i,k]=1 and A[k,j]=1.
    # So sum_{j in N(i)} (A^2)[i,j] counts edges (i,k) and (k,j) where j in N(i).
    # This counts each triangle (i,j,k) twice (once for edge i-j, once for i-k? No).
    # Let's just use the standard formula:
    # Triangles_i = 0.5 * np.sum(A[i, :] * (A @ A)[i, :])
    
    A2 = A_float @ A_float
    # Triangles per vertex
    tri_counts = 0.5 * np.sum(A_float * A2, axis=1)
    
    # Denominator: max possible triangles
    max_tri = deg * (deg - 1) / 2.0
    max_tri[max_tri == 0] = 1.0 # Avoid division by zero
    
    clustering = tri_counts / max_tri
    clustering[~has_edge] = 0.0
    
    # Normalize clustering to [0, 1]
    c_max = clustering.max()
    if c_max > 1e-12:
        clustering = clustering / c_max
    else:
        clustering = np.zeros(n)

    # Spectral Refinement
    # Use Laplacian L = D - A
    # Eigenvectors of L.
    # The first eigenvector is constant (0 eigenvalue).
    # The next few capture the main "cuts" or "communities".
    # Vertices with high magnitude in these eigenvectors are at the boundaries.
    
    try:
        if time.time() < deadline - 5.0:
            D = np.diag(deg)
            L = D - A_float
            
            # For n=200, full eigh is fast
            eigvals, eigvecs = np.linalg.eigh(L)
            
            # Take the first 3 non-trivial eigenvectors (indices 1, 2, 3)
            # Skip index 0 (constant)
            spectral_scores = np.zeros(n)
            num_vecs = min(3, n - 1)
            for k in range(1, num_vecs + 1):
                # Weight by eigenvalue gap? Or just sum magnitudes?
                # Summing magnitudes of the first few non-trivial vectors
                # highlights vertices that are "peripheral" to the main clusters.
                spectral_scores += np.abs(eigvecs[:, k])
            
            # Normalize spectral scores
            s_max = spectral_scores.max()
            if s_max > 1e-12:
                spectral_scores = spectral_scores / s_max
            else:
                spectral_scores = np.zeros(n)
                
            # Combine scores
            # Base score s (higher is better/safer)
            # Spectral score (higher is more peripheral, thus safer)
            # Clustering (higher is denser, thus less safe)
            
            # We want to boost s and spectral, and penalize clustering.
            # Normalize s to [0, 1] for weighting
            s_min = s[has_edge].min() if has_edge.any() else 0
            s_max_base = s[has_edge].max() if has_edge.any() else 1
            s_range = max(s_max_base - s_min, 1e-12)
            s_norm = (s - s_min) / s_range
            s_norm[~has_edge] = 1.0
            
            # Weights
            w_base = 1.0
            w_spec = 0.5
            w_clust = 0.2
            
            scores = w_base * s_norm + w_spec * spectral_scores - w_clust * clustering
            
            # Ensure isolated vertices are still top
            scores[~has_edge] = BIG
            
        else:
            scores = s.copy()
            scores[~has_edge] = BIG
    except Exception:
        scores = s.copy()
        scores[~has_edge] = BIG

    # Local Neighborhood Refinement
    # Penalize vertices with many high-scoring neighbors (contested)
    try:
        if time.time() < deadline - 2.0:
            # Normalize scores for comparison
            sc_min = scores[has_edge].min() if has_edge.any() else 0
            sc_max = scores[has_edge].max() if has_edge.any() else 1
            sc_range = max(sc_max - sc_min, 1e-12)
            scores_norm = (scores - sc_min) / sc_range
            scores_norm[~has_edge] = 0
            
            # Count neighbors with score > 0.5
            high_mask = (scores_norm > 0.5).astype(float)
            nbr_high = A_float @ high_mask
            
            # Normalize penalty
            max_high = nbr_high.max()
            if max_high > 1e-12:
                penalty = nbr_high / max_high
                scores = scores - 0.1 * penalty
            scores[~has_edge] = BIG
        else:
            scores[~has_edge] = BIG
    except Exception:
        scores[~has_edge] = BIG

    # Final Tie-breaker: Lower degree is slightly preferred
    deg_rank = -deg / (deg.max() + 1) if deg.max() > 0 else 0.0
    scores = scores + 1e-6 * deg_rank

    if time.time() > deadline:
        return scores

    return scores
