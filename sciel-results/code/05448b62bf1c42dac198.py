"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute priority scores for vertices to maximize the size of the
    independent set found by a greedy decoder (descending score, skip neighbors).
    
    Strategy:
    Use multiple complementary signals:
    1. Negative effective degree: prefer vertices whose neighbors are sparsely connected
       (picking them blocks fewer potential independent set members).
    2. Mean neighbor degree bonus: prefer vertices adjacent to high-degree hubs
       (removing a hub from contention preserves its other neighbors).
    3. Neighborhood sparsity ratio: prefer vertices where a large fraction of
       neighbor pairs are non-adjacent (more neighbors can coexist in the IS).
    
    All computations are vectorized with numpy for speed.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Convert to numpy array
    A = np.asarray(adjacency, dtype=bool)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
        
    if A.shape[0] != A.shape[1]:
        return np.zeros(n)
    
    # 1. Compute degrees
    degree = A.sum(axis=1).astype(float)
    
    # 2. Mean neighbor degree
    neighbor_deg_sum = A @ degree
    mean_neighbor_deg = neighbor_deg_sum / np.maximum(degree, 1.0)
    
    # 3. Effective degree: degree(v) - edges_in_N(v)
    # Vectorized computation:
    # edges_in_N(v) = 0.5 * sum_{u in N(v)} sum_{w in N(v)} A[u,w]
    #               = 0.5 * (A @ A @ degree)[v]  ... wait, let me think more carefully
    #
    # For vertex v with neighbors N(v):
    #   edges_in_N(v) = number of edges in induced subgraph on N(v)
    #   = 0.5 * sum_{u in N(v)} deg_{N(v)}(u)
    #   where deg_{N(v)}(u) = number of neighbors of u that are also in N(v)
    #   = 0.5 * sum_{u in N(v)} (A[u,:] . A[v,:])  (since A[v,:] indicates N(v))
    #   = 0.5 * (A @ (A.T @ degree_of_neighbors)) ... 
    #
    # Simpler: For each v, edges_in_N(v) = 0.5 * sum_{u} A[v,u] * (A[u,:] . A[v,:])
    #   Let M = A @ A.T  (M[i,j] = number of common neighbors of i and j)
    #   Then for vertex v: sum_{u in N(v)} (A[u,:] . A[v,:]) = sum_{u} A[v,u] * M[v,u]
    #   Wait, M[v,u] = (A @ A.T)[v,u] = sum_k A[v,k]*A[u,k] = |N(v) ∩ N(u)|
    #   But A[u,:] . A[v,:] = |N(u) ∩ N(v)| = M[v,u]
    #   So sum_{u in N(v)} M[v,u] = sum_u A[v,u] * M[v,u] = (A * M)[v,:].sum()
    #   Actually: sum_{u} A[v,u] * M[v,u] is the row sum of element-wise product.
    #   That's (A * M)[v,:].sum() = (A * M) @ ones, but more simply:
    #   For each v: s_v = sum_u A[v,u] * M[v,u]
    #   This is the diagonal of A @ M? No.
    #   s_v = sum_u A[v,u] * M[v,u] = (A * M)[v].sum()
    #   We can compute S = (A * M).sum(axis=1)
    #   Then edges_in_N(v) = 0.5 * S[v]
    
    # M = A @ A.T gives common neighbor counts
    # For efficiency with bool, convert to int
    A_int = A.astype(np.int32)
    M = A_int @ A_int.T  # M[i,j] = |N(i) ∩ N(j)|
    
    # S[v] = sum_u A[v,u] * M[v,u]
    S = (A_int * M).sum(axis=1).astype(float)
    
    # edges_in_N(v) = 0.5 * S[v]
    edges_in_N = 0.5 * S
    
    # effective_degree(v) = degree(v) - edges_in_N(v)
    effective_deg = degree - edges_in_N
    
    # 4. Neighborhood sparsity ratio: fraction of neighbor pairs that are NON-adjacent
    #    For vertex v with d_v neighbors:
    #    total_pairs = d_v * (d_v - 1) / 2
    #    adjacent_pairs = edges_in_N(v)
    #    sparsity_ratio = 1 - adjacent_pairs / total_pairs  (if total_pairs > 0)
    
    d_v = degree
    total_pairs = d_v * (d_v - 1.0) / 2.0
    # Avoid division by zero
    sparsity_ratio = np.where(total_pairs > 0, 
                              1.0 - edges_in_N / np.maximum(total_pairs, 1.0), 
                              1.0)
    
    # 5. Combine scores
    # Primary signal: negative effective degree (lower is better to pick)
    # Secondary: sparsity ratio (higher means neighbors can coexist more)
    # Tertiary: mean neighbor degree (higher means we're blocking a hub)
    
    # Normalize each component to [0, 1] for stable weighting
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    norm_eff_deg = normalize(effective_deg)       # 0 = low eff deg (good), 1 = high (bad)
    norm_sparsity = normalize(sparsity_ratio)      # 0 = dense nbrs (bad), 1 = sparse (good)
    norm_mnd = normalize(mean_neighbor_deg)        # 0 = low (bad), 1 = high (good)
    
    # Weighted combination:
    # -1 * norm_eff_deg: prefer low effective degree
    # + w1 * norm_sparsity: prefer sparse neighborhoods
    # + w2 * norm_mnd: prefer high mean neighbor degree
    
    w1 = 0.3
    w2 = 0.15
    score = -1.0 * norm_eff_deg + w1 * norm_sparsity + w2 * norm_mnd
    
    # Deadline check
    if time.time() > deadline:
        # Return what we have (already computed)
        pass
    
    return score
# EVOLVE-BLOCK-END
