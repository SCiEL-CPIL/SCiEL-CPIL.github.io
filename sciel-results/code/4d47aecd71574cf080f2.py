"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Return a score per vertex for the greedy independent set decoder.
    
    Heuristic:
    We want to pick vertices that are "cheap" to pick.
    1. Low degree is good (fewer neighbors blocked).
    2. If neighbors are densely connected (high clustering), blocking them is "cheaper" 
       in terms of opportunity cost because they couldn't be picked together anyway.
       So we reward high clustering among neighbors.
       
    Score = -degree + beta * clustering_coeff
    """
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    
    if n == 0:
        return np.array([])
        
    # 1. Degrees
    degrees = adjacency.sum(axis=1).astype(float)
    
    # 2. Clustering Coefficient of the neighborhood
    # Number of triangles containing v:
    # common_neighbors = A @ A
    # triangles_v = 0.5 * sum(common_neighbors[v, u] for u in N(v))
    common_neighbors = adjacency @ adjacency
    
    # Mask with adjacency to only sum over neighbors
    # Note: common_neighbors[v, u] is the number of common neighbors of v and u.
    # If u is a neighbor of v, then common_neighbors[v, u] is the number of w such that
    # v-w and u-w exist. This forms a triangle v-u-w.
    # Each triangle is counted twice (once for each edge from v in the triangle? 
    # Actually, for a fixed v, summing over u in N(v) counts each triangle v-u-w 
    # once for the pair (u, w) if we were summing all pairs, but here we sum 
    # common_neighbors[v, u] which is the count of w.
    # Let's verify: Triangle v-u-w.
    # u is in N(v). common_neighbors[v, u] includes w.
    # w is in N(v). common_neighbors[v, w] includes u.
    # So summing common_neighbors[v, u] for u in N(v) counts each triangle twice.
    triangles_v = 0.5 * (common_neighbors * adjacency).sum(axis=1)
    
    # Max possible triangles for a vertex with degree d is d*(d-1)/2
    max_triangles = 0.5 * degrees * (degrees - 1)
    
    # Clustering coefficient
    # If degree < 2, clustering is 0 (or undefined, set to 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        clustering_coeff = np.where(max_triangles > 0, triangles_v / max_triangles, 0.0)
        
    # 3. Combine scores
    # Primary signal: Low degree is better -> -degrees
    # Secondary signal: High clustering is better -> + clustering_coeff
    # We scale the clustering term to be comparable to degree.
    # Max degree can be N. Max clustering is 1.
    # So we might need to scale clustering up.
    # Let's try scaling by average degree or just a constant.
    # Since degree ranges 0..N, and clustering 0..1, a weight of ~N/10 might be good?
    # Or just use the raw clustering if we normalize degree?
    # Let's try: score = -degrees + 2.0 * clustering_coeff * (degrees.mean() + 1)
    # This makes the clustering term proportional to the scale of degrees.
    
    mean_deg = degrees.mean() + 1e-9
    scale = mean_deg * 0.5  # Tuning factor: how much to trust clustering vs degree
    
    scores = -degrees + scale * clustering_coeff
    
    return scores
# EVOLVE-BLOCK-END
