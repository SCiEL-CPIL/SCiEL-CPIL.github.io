"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    # Calculate degree of each vertex
    degrees = adjacency.sum(axis=1)
    
    # Calculate sum of degrees of neighbors for each vertex
    # This represents the "cost" or "impact" of removing the neighbors
    neighbor_degree_sums = adjacency.dot(degrees)
    
    # Score: Prefer vertices with low degree but high neighbor-degree sum.
    # This heuristic helps in removing "hubs" (high-degree nodes) by selecting
    # their low-degree neighbors, which simplifies the remaining graph more effectively
    # than just picking the absolute lowest degree vertex.
    scores = -degrees + neighbor_degree_sums
    
    return scores.astype(float)
# EVOLVE-BLOCK-END
