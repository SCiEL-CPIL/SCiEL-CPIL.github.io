import numpy as np

def priority(adjacency):
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Compute degrees
    degrees = adj.sum(axis=1)
    
    # Compute sum of neighbor degrees
    # adj @ degrees gives for each vertex v, sum of degrees of its neighbors
    neighbor_degree_sum = adj @ degrees
    
    # Score: We want to pick vertices that minimize the "cost"
    # Cost = degree(v) + sum(degrees of neighbors)
    # So priority = -Cost
    scores = -(degrees + neighbor_degree_sum)
    
    return scores
