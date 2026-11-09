import numpy as np

def priority(adjacency):
    adj = np.asarray(adjacency, dtype=np.float64)
    degrees = adj.sum(axis=1)
    
    # 2-hop neighborhood density
    # This penalizes vertices connected to high-degree vertices
    neighbor_degree_sums = adj @ degrees
    
    # Combine
    # To ensure the degree term is not overwhelmed if the graph is dense,
    # we can normalize the neighbor sum by the number of vertices or just use it as is.
    # In ER graphs, degrees are O(N), sums are O(N^2).
    # If we just sum, the sum dominates.
    # Is that bad?
    # If sum dominates, we are picking vertices with the lowest "neighborhood degree sum".
    # This is equivalent to picking vertices in the sparsest regions.
    # This is generally good for MIS.
    
    score = -(degrees + neighbor_degree_sums)
    
    return score
