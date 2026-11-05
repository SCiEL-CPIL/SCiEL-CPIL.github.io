import numpy as np

def priority(adjacency):
    # Ensure float for calculations
    adj = adjacency.astype(float)
    n = adj.shape[0]
    
    # Degrees
    deg = adj.sum(axis=1)
    
    # Sum of degrees of neighbors
    # adj @ deg gives sum of degrees of neighbors for each vertex
    sum_nbr_deg = adj @ deg
    
    # Average degree of neighbors
    # Avoid division by zero
    avg_nbr_deg = np.zeros(n)
    mask = deg > 0
    avg_nbr_deg[mask] = sum_nbr_deg[mask] / deg[mask]
    
    # Score: We want to pick vertices with low degree (less blocking)
    # and high average neighbor degree (blocking hubs).
    # Higher score is picked first.
    # -deg: lower degree -> higher score (since negated)
    # +avg_nbr_deg: higher avg nbr deg -> higher score
    
    # Let's scale or tune?
    # For ER(200, p), degrees are around 20.
    # -deg is around -20.
    # avg_nbr_deg is around 20.
    # They might cancel out.
    # If they cancel, score is ~0 for all.
    # We need the hub-blocking term to dominate for leaves attached to hubs.
    # Leaf (deg 1, nbr deg 50): -1 + 50 = 49.
    # Normal (deg 20, nbr deg 20): -20 + 20 = 0.
    # This looks like a good spread.
    
    scores = -deg + avg_nbr_deg
    
    return scores
