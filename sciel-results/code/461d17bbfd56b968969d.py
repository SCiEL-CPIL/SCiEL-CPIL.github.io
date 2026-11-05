"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    # Convert to numpy if needed
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # Degree of each vertex
    degrees = adj.sum(axis=1)
    
    # Compute number of triangles for each vertex
    # (A^3)[i,i] = 2 * (number of triangles containing i)
    # We need the number of edges in the neighborhood of i, which is the number of triangles containing i.
    # So edges_in_neighborhood = 0.5 * (A^3)[i,i]
    
    # Efficiently compute A^3 diagonal
    # A^2 = adj @ adj
    # A^3 = A^2 @ adj
    # We only need the diagonal of A^3
    # diag(A^3) = sum over k of (A^2)[i,k] * A[k,i] = sum over k of (A^2)[i,k] * A[i,k]
    # This is equivalent to element-wise product of A^2 and A, summed along rows?
    # No. (A^3)_{ii} = \sum_{k} (A^2)_{ik} A_{ki} = \sum_{k} (A^2)_{ik} A_{ik}
    # So we can compute A2 = adj @ adj
    # Then tri_counts = (A2 * adj).sum(axis=1) / 2.0
    
    A2 = adj @ adj
    tri_counts = (A2 * adj).sum(axis=1) / 2.0
    
    # Clustering Coefficient
    # C_i = tri_counts[i] / (degrees[i] * (degrees[i] - 1) / 2)
    # Handle degrees < 2
    clustering = np.zeros(n)
    mask = degrees >= 2
    denom = degrees[mask] * (degrees[mask] - 1) / 2.0
    clustering[mask] = tri_counts[mask] / denom
    
    # Heuristic:
    # 1. Prefer vertices with LOW degree (they block fewer vertices).
    # 2. Among those, prefer vertices with HIGH clustering coefficient.
    #    Why? If neighbors are highly connected (high clustering), they form a dense subgraph (clique-like).
    #    A dense subgraph contributes few vertices to the Independent Set (at most 1 per clique).
    #    By picking the center vertex, we "remove" a dense cluster that would have contributed little anyway.
    #    In contrast, if neighbors are not connected (low clustering), they are spread out and could each be part of the MIS.
    #    Picking the center vertex blocks many "useful" vertices.
    #    Therefore, we want to pick vertices that are "hubs" of dense clusters (high clustering) but have low overall degree.
    
    score = -degrees + clustering
    
    return score
# EVOLVE-BLOCK-END
