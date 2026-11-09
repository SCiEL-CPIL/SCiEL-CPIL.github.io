import numpy as np
import time

def priority(adjacency):
    """
    Calculates a priority score for each vertex to maximize the size of the
    greedy independent set.
    
    Strategy:
    1. Base: Prefer low degree (less blocking).
    2. Neighborhood Strength: Prefer vertices with high-degree neighbors (blocking hubs is efficient).
    3. Spectral Core: Prefer vertices that are "core" to their partition in the sparsest cut (high |Fiedler value|).
       - Vertices on the cut boundary (low |Fiedler value|) create more cross-partition conflicts.
       - The influence of this term is weighted by the spectral gap; if the graph has a clear bottleneck, the cut is more meaningful.
    4. Triangle Penalty: Deprioritize vertices in dense clusters (triangles) as they yield lower marginal gain in independent set size per blocked vertex.
    """
    start_time = time.time()
    
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    # 1. Degrees
    degrees = adj.sum(axis=1)
    
    # 2. Mean Neighbor Degree
    # Sum of degrees of neighbors
    neighbor_degree_sums = adj @ degrees
    # Avoid division by zero
    mean_neighbor_deg = neighbor_degree_sums / (degrees + 1e-9)
    
    # Base priority: Low degree is good, high neighbor degree is good
    base_priorities = -degrees + mean_neighbor_deg
    
    # Deadline check
    if time.time() - start_time > 100:
        return base_priorities.astype(float)
        
    # 3. Triangle Count
    # A^2 counts paths of length 2.
    # (A * A^2)_{vu} is non-zero only if A_{vu}=1, in which case it is the number of common neighbors.
    # Summing over u gives twice the number of triangles containing v.
    A2 = adj @ adj
    tri_sum = (adj * A2).sum(axis=1)
    tri_count = 0.5 * tri_sum
    
    # Penalty for triangles
    # Scale tri_count to be comparable to degrees.
    # Max tri_count for n=200 is ~10000, but usually much less.
    # Let's use a small weight.
    tri_penalty = 0.05 * tri_count
    
    current_priorities = base_priorities - tri_penalty
    
    # Deadline check
    if time.time() - start_time > 110:
        return current_priorities.astype(float)
        
    # 4. Spectral Heuristic
    try:
        D = np.diag(degrees)
        L = D - adj
        
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # eigvals[0] is 0 (for connected graph). eigvals[1] is the Fiedler value.
        fiedler_val = eigvals[1]
        fiedler_vec = eigvecs[:, 1]
        
        # Spectral gap
        gap = fiedler_val - eigvals[0]
        
        # Weight the spectral term based on the gap.
        # If gap is large, the cut is strong, so trust the Fiedler vector more.
        # Use a saturating function to keep weight in [0, 1].
        spectral_weight = gap / (gap + 0.5)
        
        # Core score: |fiedler_vec|
        # Normalize |fiedler_vec| to [0, 1] range for consistent scaling?
        # Max abs value of a unit norm vector is 1 (if concentrated on one node).
        # Usually much smaller.
        abs_fiedler = np.abs(fiedler_vec)
        
        # Scale factor to make spectral term comparable to degree terms.
        # Degrees are ~10-50. |fiedler| is ~0.1.
        # We want spectral contribution to be ~5-10.
        # So alpha ~ 50-100.
        alpha = 50.0
        
        spectral_term = spectral_weight * alpha * abs_fiedler
        
        current_priorities += spectral_term
        
    except Exception:
        # If spectral fails, return current best
        pass
        
    # Final deadline check
    if time.time() - start_time > 118:
        return current_priorities.astype(float)
        
    return current_priorities.astype(float)
