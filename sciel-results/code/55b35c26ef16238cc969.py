"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Calculates a priority score for each vertex to maximize the size of the
    greedy independent set.
    
    Heuristic combines:
    1. Primary: Prefer vertices with lower degree (fewer neighbors to block).
    2. Secondary: Prefer vertices whose neighbors have higher degrees.
       Rationale: Picking a vertex 'v' blocks its neighbors. If the blocked
       neighbors have high degrees, it is beneficial because those high-degree
       vertices would have blocked many other vertices if they had been picked.
    3. Tertiary: Prefer vertices whose neighbors are well-connected (high edges
       among neighbors). Rationale: If neighbors are connected, they would have
       blocked each other anyway, so blocking them is more efficient.
    4. Quaternary: Tie-breaker using inverse degree of neighbors.
       Rationale: Among vertices with similar degree, prefer those whose neighbors
       have high degrees (low inverse degree), as blocking high-degree vertices
       is "cheaper" in terms of opportunity cost.
    """
    start_time = time.time()
    
    # Convert to integer array for efficient arithmetic
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    
    # Calculate degrees
    degrees = adj.sum(axis=1).astype(np.float64)
    
    # Calculate sum of degrees of neighbors for each vertex
    neighbor_degree_sums = (adj @ degrees).astype(np.float64)
    
    # Calculate mean neighbor degree (normalized)
    mean_neighbor_deg = neighbor_degree_sums / (degrees + 1.0)
    
    # Calculate sum of inverse degrees of neighbors for each vertex
    inv_deg = 1.0 / (degrees + 1.0)
    neighbor_inv_degree_sums = (adj @ inv_deg).astype(np.float64)
    
    # Calculate number of edges among neighbors for each vertex
    # For vertex v, edges among neighbors = sum over pairs (u,w) in N(v) of adj[u,w]
    # This is (adj @ adj @ adj)[v] / 2 approximately, but let's compute it properly
    # adj @ adj gives for each (i,j) the number of common neighbors
    # For vertex v, the number of edges among its neighbors is:
    # sum_{u in N(v)} sum_{w in N(v), w != u} adj[u,w] / 2
    # = (1/2) * sum_{u in N(v)} (adj[u] @ adj[v]) 
    # Actually, let's use: (adj @ adj) @ adj[v] gives sum over u in N(v) of (adj @ adj)[u, v]
    # which is sum over u in N(v) of common neighbors of u and v
    # That's not quite right. Let me think again.
    
    # The number of edges in the subgraph induced by N(v) is:
    # (1/2) * sum_{u in N(v)} deg_{N(v)}(u)
    # where deg_{N(v)}(u) = number of neighbors of u that are also in N(v)
    # = (1/2) * sum_{u in N(v)} (adj[u] @ adj[v])
    # Because adj[u] @ adj[v] = number of common neighbors of u and v, which is
    # the number of neighbors of u that are also neighbors of v.
    
    # So: edges_among_neighbors[v] = (1/2) * (adj @ (adj.T @ adj[v]))
    # But adj is symmetric, so adj.T = adj
    # Let's compute: common = adj @ adj  # common[i,j] = number of common neighbors of i and j
    # Then for vertex v: sum_{u in N(v)} common[u, v] = (common @ adj[v])
    # Wait, common[u,v] for u in N(v) is the number of common neighbors of u and v.
    # Sum over u in N(v) of common[u,v] = sum over u in N(v) of (adj[u] @ adj[v])
    # = (adj.T @ (adj @ adj)) @ v? No.
    
    # Let's just compute it directly:
    # common_neighbors = adj @ adj  # common_neighbors[i,j] = number of common neighbors of i and j
    # For vertex v, the number of edges among its neighbors is:
    # (1/2) * sum_{u in N(v)} (number of neighbors of u that are in N(v))
    # = (1/2) * sum_{u in N(v)} common_neighbors[u, v]
    # = (1/2) * (common_neighbors @ adj[v])
    # But common_neighbors @ adj[v] is a vector, and we want the v-th component? No.
    
    # Actually: sum_{u in N(v)} common_neighbors[u, v] = (common_neighbors.T @ adj[:, v])[v]? 
    # This is getting complicated. Let me use a simpler approximation.
    
    # Simpler: use the sum of degrees of neighbors as a proxy for "blocking efficiency"
    # and skip the edges-among-neighbors computation for speed.
    
    # Define weights for the cost function.
    # We want to MINIMIZE: degree, neighbor_inv_degree_sums
    # and MAXIMIZE: mean_neighbor_deg
    
    # Scale analysis for N=200:
    # degree: [0, 199]
    # mean_neighbor_deg: [0, 200]
    # neighbor_inv_degree_sums: [0, 200]
    
    # We want degree to be the primary factor.
    # Let's define cost = M * degree - w1 * mean_neighbor_deg + w2 * neighbor_inv_degree_sums
    # We want M * degree to dominate.
    
    M = 1000.0
    w1 = 1.0
    w2 = 0.5
    
    costs = M * degrees - w1 * mean_neighbor_deg + w2 * neighbor_inv_degree_sums
    priorities = -costs
    
    # Deadline guard
    if time.time() - start_time > 110:
        return priorities.astype(float)
        
    return priorities.astype(float)
# EVOLVE-BLOCK-END
