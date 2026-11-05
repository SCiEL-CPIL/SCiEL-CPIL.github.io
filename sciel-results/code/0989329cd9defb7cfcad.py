"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Score vertices for greedy independent set selection.
    Returns a score per vertex; higher score = selected earlier.
    Uses spectral centrality (principal + Fiedler eigenvectors) + degree + neighborhood features.
    Vertices with lower combined centrality score are preferred (assigned higher priority).
    """
    n = adjacency.shape[0]
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    # Convert to float for computations
    A = adjacency.astype(float)

    # Degree
    degree = A.sum(axis=1)

    # Power iteration for principal eigenvector (spectral centrality)
    max_iter = 300
    v = np.ones(n) / np.sqrt(n)
    v = v / (np.linalg.norm(v) + 1e-12)

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        v_new = A @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-12:
            break
        v_new = v_new / norm
        if np.linalg.norm(v_new - v) < 1e-10:
            v = v_new
            break
        v = v_new

    eigenvector = v  # principal eigenvector centrality

    # Fiedler vector: second-smallest eigenvalue of Laplacian
    # L = D - A, use power iteration on (L - lambda*I)^{-1} or shift-invert
    # Simpler: use power iteration on L with deflation
    D = np.diag(degree)
    L = D - A
    
    # Initialize with vector orthogonal to ones
    f = np.random.randn(n)
    f -= f.mean()
    f = f / (np.linalg.norm(f) + 1e-12)
    
    max_iter_f = 200
    for _ in range(max_iter_f):
        if time.time() > deadline:
            break
        # Apply L, then deflate (remove component along ones)
        f_new = L @ f
        f_new -= f_new.mean()
        norm = np.linalg.norm(f_new)
        if norm < 1e-12:
            break
        f_new = f_new / norm
        if np.linalg.norm(f_new - f) < 1e-10:
            f = f_new
            break
        f = f_new
    
    fiedler = f

    # Neighborhood expansion: sum of degrees of neighbors
    nbr_degree_sum = A @ degree

    # Outward expansion: average degree of neighbors
    outward_exp = np.zeros(n)
    nonzero_deg = degree > 0
    outward_exp[nonzero_deg] = nbr_degree_sum[nonzero_deg] / degree[nonzero_deg]

    # Minimum neighbor degree
    min_nbr_deg = np.zeros(n)
    for i in range(n):
        if degree[i] > 0:
            neighbors = np.where(adjacency[i])[0]
            min_nbr_deg[i] = np.min(degree[neighbors])

    # Number of neighbors that are high-degree (above median)
    median_deg = np.median(degree)
    high_deg_mask = degree > median_deg
    high_nbr_count = A @ high_deg_mask.astype(float)

    # Normalize features
    def normalize(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    norm_degree = normalize(degree)
    norm_eigenvector = normalize(eigenvector)
    norm_fiedler = normalize(np.abs(fiedler))  # Use absolute value for centrality
    norm_nbr_deg_sum = normalize(nbr_degree_sum)
    norm_outward = normalize(outward_exp)
    norm_min_nbr = normalize(min_nbr_deg)
    norm_high_nbr = normalize(high_nbr_count)

    # Weighted combination
    w1 = 0.25  # degree
    w2 = 0.25  # spectral centrality
    w3 = 0.10  # Fiedler vector
    w4 = 0.10  # neighbor degree sum
    w5 = 0.08  # outward expansion
    w6 = 0.08  # min neighbor degree
    w7 = 0.14  # high-degree neighbor count

    centrality = (
        w1 * norm_degree +
        w2 * norm_eigenvector +
        w3 * norm_fiedler +
        w4 * norm_nbr_deg_sum +
        w5 * norm_outward +
        w6 * norm_min_nbr +
        w7 * norm_high_nbr
    )

    # Priority: higher score = selected first
    priority_scores = 1.0 - centrality

    return priority_scores
# EVOLVE-BLOCK-END
