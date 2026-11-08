import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for a greedy independent set decoder.
    The decoder selects vertices in descending order of score.
    To maximize the independent set size, we want to select vertices that 
    'cost' the least in terms of blocked neighbors.
    
    Heuristic:
    The cost of selecting vertex v is the sum of the degrees of its neighbors.
    Selecting v removes v and all its neighbors from consideration.
    The 'value' lost is proportional to the degrees of the neighbors (since high-degree 
    neighbors could have covered many other vertices).
    
    Therefore, we assign a score that is the negative of the sum of neighbor degrees.
    We also subtract the vertex's own degree to break ties in favor of lower degree 
    vertices, as they are generally safer choices in early steps.
    
    score(v) = - (sum of degrees of neighbors of v) - degree(v)
    """
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # Ensure float for dot product if needed, though int is fine
    degrees = adjacency.sum(axis=1)
    
    # Sum of degrees of neighbors for each vertex
    # This is the 'weighted degree'
    neighbor_degree_sum = adjacency.dot(degrees)
    
    # Score: Higher is better.
    # We want LOW neighbor_degree_sum and LOW degree.
    # So we negate them.
    scores = -neighbor_degree_sum - degrees
    
    return scores
