import numpy as np

def priority(adjacency):
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Handle potential zero degrees for inverse calculation
    # In ER graphs with n=200, p>0, min degree is usually > 0, but be safe.
    safe_degrees = np.maximum(degrees, 1.0)
    inv_degrees = 1.0 / safe_degrees
    
    # Sum of inverse degrees of neighbors
    # If v is connected to low-degree nodes (high inv_deg), this sum is large.
    # We want to avoid picking such v. So subtract this sum.
    neighbor_inv_sum = adjacency.dot(inv_degrees)
    
    # Score: Minimize degree and minimize destruction of low-degree neighbors
    # High score = Good candidate
    # Good candidate: Low degree, and connected to high-degree (low inv) neighbors.
    
    # Let's combine them.
    # The current program uses sum of degrees. This uses sum of inv degrees.
    # This is a distinct heuristic: "Protect low-degree nodes".
    
    scores = -(degrees + 2.0 * neighbor_inv_sum)
    
    return scores
