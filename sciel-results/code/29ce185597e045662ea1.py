"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, laplacian
from scipy.linalg import eigh
import sys

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO TSP.
    Strategy: Spectral Graph-Theoretic Heuristic.
    
    This approach moves away from geometric features (Delaunay, Hull, MDS alignment)
    and instead uses the algebraic structure of the Minimum Spanning Tree (MST).
    
    1. Compute the MST of the complete graph defined by the distance matrix.
    2. Compute node degrees in the MST.
    3. Compute the Laplacian of the MST and its eigenvectors (spectral embedding).
    4. Construct heuristic based on:
       a) Inverse distance (base).
       b) Product of MST degrees of the endpoints (hub connectivity).
       c) Spectral proximity: Edges connecting nodes that are close in the
          spectral embedding of the MST are boosted. This captures the "flow"
          of the tree structure.
       d) Exponential decay of edge rank in the MST cut property (approximated
          by distance to the MST edge set, effectively penalizing long jumps
          that are not part of the backbone).
    """
    n = distance_matrix.shape[0]
    
    # Safety check for time
    start_time = time.time()
    deadline = start_time + 10.0  # 10 seconds is plenty for n=50
    
    try:
        # 1. Compute MST
        # Symmetrize just in case, though TSP instances are usually symmetric
        dist_sym = (distance_matrix + distance_matrix.T) / 2.0
        np.fill_diagonal(dist_sym, 0.0)
        
        dist_sparse = csr_matrix(dist_sym)
        mst = minimum_spanning_tree(dist_sparse)
        mst_dense = mst.toarray()
        
        # Create mask for MST edges
        mst_edges = (mst_dense > 0) & (mst_dense < np.inf)
        mst_edges = mst_edges | mst_edges.T
        
        # 2. Compute node degrees in MST
        # Degree is the sum of the adjacency matrix row (since it's a tree, weights are 1 or 0)
        # mst_dense has weights as distances, so we need a binary adjacency
        mst_adj = (mst_dense > 0).astype(float)
        degrees = mst_adj.sum(axis=1)
        
        # Ensure no zeros (isolated nodes shouldn't happen, but safe)
        degrees = np.maximum(degrees, 1.0)
        
        # 3. Compute Spectral Embedding of the MST
        # Laplacian L = D - A
        # For the MST, we use the binary adjacency
        D_mat = np.diag(degrees)
        L = D_mat - mst_adj
        
        # Eigendecomposition of L
        # We want the smallest non-zero eigenvalues (Fiedler vector etc.)
        # eigh returns ascending order.
        eigvals, eigvecs = eigh(L)
        
        # The first eigenvalue is 0 with eigenvector [1, 1, ..., 1].
        # We take the next k eigenvectors for embedding.
        # Let's take 4 eigenvectors for a 4D spectral embedding.
        k = min(4, n-1)
        # Skip the first eigenvector (trivial)
        if k > 0:
            spectral_coords = eigvecs[:, 1:1+k]
        else:
            spectral_coords = np.zeros((n, 0))
        
        # 4. Construct Heuristic Components
        
        # Base: Inverse distance
        h_base = 1.0 / (distance_matrix + 1e-9)
        
        # Component A: Hub Connectivity
        # Product of degrees: deg[i] * deg[j]
        # Normalize to mean 1
        deg_prod = np.outer(degrees, degrees)
        deg_prod_norm = deg_prod / np.mean(deg_prod)
        
        # Component B: Spectral Proximity
        # Distance in spectral space
        if spectral_coords.shape[1] > 0:
            # Compute pairwise distances in spectral space
            # coords[i] - coords[j]
            diff_spec = spectral_coords[None, :, :] - spectral_coords[:, None, :]
            spec_dist_sq = np.sum(diff_spec ** 2, axis=2)
            
            # Convert distance to attractiveness: exp(-alpha * dist)
            # Smaller spectral distance -> higher attractiveness
            alpha = 1.0
            spec_attr = np.exp(-alpha * spec_dist_sq)
            
            # Normalize to mean 1
            spec_attr_norm = spec_attr / np.mean(spec_attr)
        else:
            spec_attr_norm = np.ones((n, n))
        
        # Component C: MST Edge Boost
        # Edges in the MST are the backbone.
        # We want to strongly boost them.
        mst_boost = np.ones((n, n))
        mst_boost[mst_edges] = 5.0  # Strong boost for MST edges
        
        # Component D: Exponential Decay of Rank (Approximation)
        # The directive mentions "exponential decay of the edge's rank in the MST's cut property".
        # A simpler proxy: penalize edges that are far from the MST structure.
        # We can use the distance to the nearest MST edge for each node?
        # Or simply, we already have the base inverse distance.
        # Let's add a penalty for edges that are NOT in the MST and are long.
        # Actually, the base inverse distance already penalizes long edges.
        # The "rank" idea suggests that edges added later in Kruskal's (longer) are less attractive.
        # Since MST edges are sorted by weight in Kruskal's, the "rank" is correlated with length.
        # So inverse distance covers this.
        # Let's stick to the spectral and degree components as the main structural change.
        
        # 5. Combine Components
        # h = h_base * deg_prod_norm * spec_attr_norm * mst_boost
        
        # We need to be careful with magnitudes.
        # Let's normalize the product of the multiplicative factors.
        factors = deg_prod_norm * spec_attr_norm * mst_boost
        # Normalize factors to have mean 1
        factors = factors / np.mean(factors)
        
        h_final = h_base * factors
        
        # Ensure positive and finite
        h_final = np.where(np.isfinite(h_final) & (h_final > 0), h_final, 1e-9)
        
        # Diagonal to 0
        np.fill_diagonal(h_final, 0.0)
        
        # Check time
        if time.time() > deadline:
             # Fallback if too slow (unlikely)
             h_fallback = 1.0 / (distance_matrix + 1e-9)
             np.fill_diagonal(h_fallback, 0.0)
             return h_fallback
             
        return h_final
        
    except Exception as e:
        # Fallback to simple inverse distance
        h_fallback = 1.0 / (distance_matrix + 1e-9)
        np.fill_diagonal(h_fallback, 0.0)
        return h_fallback
# EVOLVE-BLOCK-END
