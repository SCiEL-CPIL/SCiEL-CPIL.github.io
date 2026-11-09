"""Rewritten program using spectral edge energy heuristic for Maximum Independent Set."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def priority(adjacency):
    """
    Assign scores to vertices using a spectral edge energy heuristic.
    
    The idea:
    1. Compute the graph Laplacian L = D - A.
    2. Find the Fiedler vector (eigenvector of the second-smallest eigenvalue).
    3. Compute the "Spectral Edge Energy" for each vertex: the sum of squared differences
       between the vertex's Fiedler value and its neighbors' Fiedler values.
    4. Vertices with high spectral edge energy are "spectrally distinct" from their neighbors,
       suggesting they are boundaries or outliers rather than part of a dense, homogeneous cluster.
       Picking such vertices minimizes the number of "similar" vertices removed.
    5. Combine this with a strong negative degree penalty. Lower degree vertices are generally
       better for independent sets.
    
    Returns:
        A numpy array of scores, one per vertex. Higher score = selected earlier.
    """
    n = adjacency.shape[0]
    A = np.asarray(adjacency, dtype=float)
    
    # Ensure symmetry
    A = (A + A.T) / 2.0
    
    # Compute degree vector
    degrees = A.sum(axis=1)
    
    # Handle case where graph is empty or trivial
    if n == 0:
        return np.array([], dtype=float)
    
    # Build Laplacian
    D = np.diag(degrees)
    L = D - A
    
    # For efficiency with larger graphs, use sparse representation
    L_sparse = sp.csr_matrix(L)
    
    # We want the second-smallest eigenvalue and its eigenvector (Fiedler vector)
    # Use shift-invert mode to find eigenvalues near 0
    try:
        # Find eigenvalues near 0 (smallest ones)
        # We need at least 2 eigenvalues; the smallest is 0 (for connected components)
        k = min(2, n)
        if n > 1:
            vals, vecs = spla.eigsh(L_sparse, k=k, sigma=0, which='LM')
        else:
            vals = np.array([0.0])
            vecs = np.array([[1.0]])
        
        # Sort by eigenvalue
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        # Fiedler vector is the eigenvector corresponding to the second-smallest eigenvalue
        if len(vals) >= 2:
            fiedler = vecs[:, 1]
        else:
            # Fallback: use degree-based scoring
            fiedler = np.zeros(n)
            
        # Normalize Fiedler vector to have zero mean and unit variance for stable scaling
        fiedler_mean = np.mean(fiedler)
        fiedler_std = np.std(fiedler)
        if fiedler_std > 1e-10:
            fiedler = (fiedler - fiedler_mean) / fiedler_std
        else:
            fiedler = np.zeros(n)
        
        # Compute Spectral Edge Energy for each vertex
        # E_i = sum_{j in N(i)} (f_i - f_j)^2
        # E_i = d_i * f_i^2 - 2 * f_i * sum_{j in N(i)} f_j + sum_{j in N(i)} f_j^2
        
        # Efficiently compute using sparse matrix multiplication
        # sum_{j in N(i)} f_j is (A @ f)
        sum_f_neighbors = A.dot(fiedler)
        
        # sum_{j in N(i)} f_j^2 is (A @ (f^2))
        sum_f_sq_neighbors = A.dot(fiedler ** 2)
        
        # Calculate energy
        # Handle division by zero or invalid values
        energy = degrees * (fiedler ** 2) - 2.0 * fiedler * sum_f_neighbors + sum_f_sq_neighbors
        
        # Ensure energy is non-negative (numerical errors might make it slightly negative)
        energy = np.maximum(energy, 0.0)
        
        # Normalize energy to be comparable to degrees
        # We want to reward high energy (distinctness) but penalize high degree.
        # Let's scale energy so its max is around the max degree, or use a fixed weight.
        max_degree = np.max(degrees) if np.max(degrees) > 0 else 1.0
        max_energy = np.max(energy) if np.max(energy) > 0 else 1.0
        
        # Weight for energy: We want it to be a secondary factor but significant.
        # If energy is high, it might override a slightly higher degree.
        # Let's try: Score = -degree + 0.5 * (energy / max_energy) * max_degree
        # This means a vertex with max energy and min degree gets the highest score.
        # A vertex with min energy and max degree gets the lowest.
        
        energy_normalized = energy / max_energy
        degree_normalized = degrees / max_degree
        
        # Combine: Lower degree is better (negative score), Higher energy is better (positive score)
        # We scale energy to have similar impact to degree differences.
        alpha = 0.5
        scores = -degree_normalized + alpha * energy_normalized
        
        # Optional: Add a tiny bit of the raw fiedler value to break ties?
        # No, energy should be sufficient.
        
    except Exception:
        # Fallback to simple degree-based scoring if spectral computation fails
        scores = -degrees.astype(float) / (np.max(degrees) + 1e-10)
    
    return scores
# EVOLVE-BLOCK-END
