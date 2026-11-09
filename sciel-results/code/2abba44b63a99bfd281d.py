"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    n = len(adjacency)
    adjacency = np.asarray(adjacency, dtype=np.int8)
    
    # Use spectral initialization: compute Fiedler vector of the graph Laplacian
    # and split by sign. This often gives a good initial cut for Max-Cut.
    # L = D - A
    degrees = adjacency.sum(axis=1).astype(float)
    # Build Laplacian
    # L = D - A
    # We want the eigenvector corresponding to the second smallest eigenvalue (Fiedler vector)
    
    # Since n=100, we can use scipy.linalg.eigh for symmetric matrices
    try:
        from scipy.linalg import eigh
        L = np.diag(degrees) - adjacency.astype(float)
        # eigh returns eigenvalues in ascending order
        # We want the second smallest eigenvalue's eigenvector (index 1)
        # For disconnected graphs, there might be multiple zero eigenvalues.
        # To be safe, we can find the eigenvector with the largest component variance 
        # among the low-frequency ones, or just take index 1 if connected.
        # A more robust approach: find the eigenvector that minimizes the Rayleigh quotient 
        # but is not the constant vector.
        
        eigenvalues, eigenvectors = eigh(L)
        
        # The first eigenvector is usually constant (or close to it) for connected graphs.
        # We pick the eigenvector with the smallest eigenvalue that is not near-zero variance.
        # Actually, for Max-Cut, the Fiedler vector (second smallest) is standard.
        # If the graph is disconnected, the first few eigenvectors correspond to components.
        # Let's just take the eigenvector with index 1, but if the graph is disconnected,
        # we might need to handle it. For 100-node random graphs, they are likely connected.
        
        # A safer bet: find the eigenvector that gives the best cut among the first few.
        # But that's expensive. Let's stick with Fiedler vector (index 1).
        # If the graph is disconnected, index 1 might be 0. Let's check variance.
        
        # Find the first eigenvector (after the constant one) that has significant variance.
        # The constant vector has zero variance.
        for i in range(1, min(10, n)):
            vec = eigenvectors[:, i]
            if np.var(vec) > 1e-10:
                fiedler = vec
                break
        else:
            fiedler = eigenvectors[:, 1] if n > 1 else np.ones(n)
        
        # Split by median or zero? Median is more robust for balanced cuts.
        # For Max-Cut, a balanced cut is often good.
        median_val = np.median(fiedler)
        labels = (fiedler >= median_val).astype(np.int8)
        
    except Exception:
        # Fallback to random or alternating
        labels = np.arange(n) % 2
    
    # Local improvement: flip vertices if it improves the cut
    # Cut size = sum over edges (u,v) of [labels[u] != labels[v]]
    # For a vertex v, the number of crossing edges if we flip v is:
    # current crossing = edges to opposite side
    # after flipping, crossing = edges to same side (which were not crossing before)
    # So we flip if edges_to_same > edges_to_opposite
    
    # Precompute neighbor lists for efficiency? Or just use matrix operations.
    # n=100, so O(n^2) per pass is fine.
    
    # We do multiple passes until no improvement or max iterations.
    for _ in range(20):
        improved = False
        for v in range(n):
            # Count edges to same and opposite
            same_count = adjacency[v, labels == labels[v]].sum()
            opp_count = adjacency[v, labels != labels[v]].sum()
            
            # If flipping improves the cut
            if same_count > opp_count:
                labels[v] = 1 - labels[v]
                improved = True
        
        if not improved:
            break
    
    return labels
# EVOLVE-BLOCK-END
