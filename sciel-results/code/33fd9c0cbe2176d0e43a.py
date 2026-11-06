"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute priority scores for vertices for a greedy Maximum Independent Set decoder.
    The decoder picks vertices in descending score order, skipping neighbors of picked vertices.
    """
    start_time = time.time()
    DEADLINE = 110.0  # seconds, safety margin before 120s kill
    
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Ensure symmetry (in case of slight numerical asymmetry)
    A = (A + A.T) / 2.0
    
    # Degrees
    deg = A.sum(axis=1)
    
    # --- Component 1: Negative degree (low degree first) ---
    # This is the baseline heuristic.
    score = -deg.copy()
    
    # --- Component 2: Neighbor Clustering / Structure ---
    # We want to pick vertices whose neighbors are densely connected to each other.
    # If neighbors of v are densely connected, picking v "wastes" fewer potential independent set members
    # because those neighbors are likely to conflict with each other anyway.
    # A = A @ A gives the number of length-2 paths between nodes.
    # (A @ A)[i, j] is the number of common neighbors of i and j.
    # The diagonal (A @ A)[i, i] is the degree of i.
    # The off-diagonal sum for row i is sum_{j != i} (A @ A)[i, j] = sum_{j} (A @ A)[i, j] - deg[i].
    # Let's compute A2 = A @ A.
    A2 = A @ A
    # Common neighbors with all neighbors of i:
    # sum_{j in N(i)} |N(i) ∩ N(j)|
    # This is roughly the "clustering" of the neighborhood.
    
    # Let's try a simpler structural metric:
    # "Effective degree" or "independent set potential" of neighbors.
    # If a vertex v has neighbors that are high-degree, picking v blocks many high-degree vertices.
    # If a vertex v has neighbors that are low-degree, picking v blocks few low-degree vertices.
    # So we prefer vertices with LOW average neighbor degree?
    # The seed uses -deg.
    # Let's look at the "inspiration" mentioned in the prompt:
    # "Combines the effective 'mean neighbor degree' signal ... with 'neighbor clustering' ...
    #  Measured: 1.015 -> 1.03."
    
    # Mean neighbor degree:
    # mnd[i] = sum_{j in N(i)} deg[j] / deg[i] (if deg[i] > 0)
    # We want to MINIMIZE this? Or maximize?
    # If I pick a vertex with high MND, I block vertices that are connected to many others.
    # This seems bad. So we want LOW MND.
    # Thus, add -mnd to the score.
    
    mnd = np.zeros(n)
    nonzero_deg = deg > 0
    # sum of neighbor degrees
    sum_nei_deg = (A @ deg)
    mnd[nonzero_deg] = sum_nei_deg[nonzero_deg] / deg[nonzero_deg]
    
    # Normalize to comparable scale
    # deg is in [0, n]
    # mnd is in [0, n]
    
    # Neighbor Clustering:
    # C[i] = (A @ A)[i, i] is deg[i].
    # Let's use the number of triangles involving i?
    # Or the density of the subgraph induced by N(i).
    # density = 2 * edges_in_N(i) / (deg[i] * (deg[i]-1))
    # edges_in_N(i) = 0.5 * sum_{j in N(i)} (A @ A)[i, j]
    # Let's compute this.
    
    # A2_off_diag_sum[i] = sum_{j != i} A2[i, j]
    # Since A2 is symmetric, A2[i, j] = |N(i) ∩ N(j)|.
    # For j in N(i), A2[i, j] is the number of common neighbors.
    # Each edge (j, k) in N(i) contributes 1 to A2[i, j] and 1 to A2[i, k].
    # So sum_{j in N(i)} A2[i, j] = 2 * edges_in_N(i).
    
    # We can compute this efficiently:
    # sum_A2_nei = (A * A2).sum(axis=1)  # Element-wise multiply A and A2, then sum rows.
    # Since A[i, j] is 1 only if j is a neighbor, this sums A2[i, j] for j in N(i).
    
    sum_A2_nei = (A * A2).sum(axis=1)
    
    # edges_in_N(i) = 0.5 * sum_A2_nei[i]
    # density[i] = edges_in_N(i) / (0.5 * deg[i] * (deg[i]-1)) if deg[i] > 1
    # density[i] = sum_A2_nei[i] / (deg[i] * (deg[i]-1))
    
    density = np.zeros(n)
    mask_deg_gt_1 = deg > 1
    density[mask_deg_gt_1] = sum_A2_nei[mask_deg_gt_1] / (deg[mask_deg_gt_1] * (deg[mask_deg_gt_1] - 1))
    
    # We want to MAXIMIZE density?
    # If neighbors are densely connected, they form a clique-like structure.
    # Picking v blocks this whole cluster.
    # But in a cluster, you can only pick 1 vertex from the cluster anyway.
    # So the "loss" of picking v is small if the neighbors are already mutually exclusive.
    # So we should PREFER high density.
    # Thus, add +density to the score.
    
    # Let's combine:
    # score = w1 * (-deg) + w2 * (-mnd) + w3 * (density)
    
    # Let's try equal weights first, or normalized.
    # Scale normalization:
    # deg mean ~ n*p.
    # mnd mean ~ n*p.
    # density mean ~ p.
    
    # Let's normalize each component to [0, 1] or mean 0 var 1.
    
    def normalize(x):
        if np.std(x) == 0:
            return np.zeros_like(x)
        return (x - np.mean(x)) / np.std(x)
    
    s1 = normalize(-deg)
    s2 = normalize(-mnd)
    s3 = normalize(density)
    
    # We need to be careful with the signs.
    # s1 is high for low degree. Good.
    # s2 is high for low mean neighbor degree. Good.
    # s3 is high for high neighbor density. Good.
    
    # Let's test a linear combination.
    # The prompt says "Combines ... mean neighbor degree ... with ... neighbor clustering".
    # It doesn't explicitly say degree is still there, but the seed is degree.
    # Let's try: score = s1 + s2 + s3
    
    new_score = s1 + s2 + s3
    
    # Check time
    if time.time() - start_time > DEADLINE:
        # Return the simplest valid score if we run out of time
        return -deg.astype(float)
    
    # Let's also consider a "spectral" component if time permits?
    # The prompt mentions spectral relaxation.
    # But the "measured" improvement came from MND + Clustering.
    # Let's stick to that family but maybe refine the weights.
    
    # What if we use a soft-max or log-sum-exp?
    # The directive says: "consider a smooth surrogate ... (soft-min/soft-max via log-sum-exp)".
    # This is for the objective function optimization, but here we are just defining a heuristic score.
    # Maybe we can combine the scores in a way that is robust?
    
    # Let's try a different combination:
    # score = -deg - 0.5 * mnd + 0.5 * density
    
    # Actually, let's look at the "fresh design" hint:
    # "Formulate the priority assignment as a Maximum Weight Independent Set ...
    #  use a fast spectral relaxation ... to generate the initial score vector."
    
    # However, the "Knowledge" section says the MND+Clustering combo worked.
    # "Do NOT switch to a different design family."
    # The current family is "local degree heuristics".
    # MND and Clustering are local structural heuristics.
    # Spectral is global.
    # The directive says "KEEP this design and investigate improvements within its family."
    # The current program is just -deg.
    # The "island best" is 1.03 (MND+Clustering).
    # So I should implement MND+Clustering.
    
    # Let's refine the MND+Clustering score.
    # Is there a better way to compute "neighbor clustering"?
    # The density metric is good.
    
    # Let's try to weight them based on degree?
    # High degree vertices have more neighbors, so their MND is more stable.
    # Low degree vertices have less information.
    
    # Let's try:
    # score = -deg + density - mnd
    
    # Let's check the time again.
    # A @ A is O(n^3) or O(n^2 * nnz). For n=200, it's very fast.
    
    # Let's try to add a small amount of randomness or tie-breaking?
    # No, the decoder is deterministic given the scores.
    
    # Let's try a weighted sum with tuned weights.
    # From the prompt: "Combines the effective 'mean neighbor degree' signal ... with the 'neighbor clustering' ... Measured: 1.015 -> 1.03."
    # This implies both signals are important.
    
    # Let's try:
    # score = s1 + 0.5 * s2 + 0.5 * s3
    
    # Or maybe just s2 + s3?
    # The seed is s1.
    # Let's try s1 + s2 + s3.
    
    # Let's also consider the "spectral" hint in the directive, but as a "hypothesis to test".
    # "Treat these as hypotheses to test. Do NOT switch to a different design family."
    # Since the measured success was MND+Clustering, I will stick to that.
    
    # One more refinement:
    # Instead of simple density, use "local clustering coefficient" of the neighborhood?
    # The density I computed is exactly the clustering coefficient of the induced subgraph on N(i).
    
    # Let's try to normalize the components differently.
    # Maybe min-max normalization?
    
    def min_max_normalize(x):
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max == x_min:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    
    s1_mm = min_max_normalize(-deg)
    s2_mm = min_max_normalize(-mnd)
    s3_mm = min_max_normalize(density)
    
    # Combine
    final_score = s1_mm + s2_mm + s3_mm
    
    # Ensure the return type is a 1D array of floats
    return final_score.astype(float)

# EVOLVE-BLOCK-END
