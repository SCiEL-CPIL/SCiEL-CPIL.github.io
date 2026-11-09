import numpy as np

def priority(adjacency):
    n = adjacency.shape[0]
    active = np.ones(n, dtype=bool)
    scores = np.zeros(n, dtype=float)
    
    for step in range(n):
        if not np.any(active):
            break
        
        # Calculate degrees in the active subgraph
        deg = (adjacency @ active).astype(int)
        
        # Calculate sum of neighbor degrees (MSND heuristic)
        # deg[u] is the active degree of u.
        # sum_nei_deg[v] = sum(deg[u] for u in N(v))
        sum_nei_deg = (adjacency @ deg).astype(int)
        
        # Total cost: degree + sum of neighbor degrees
        # We want to minimize this cost
        cost = deg + sum_nei_deg
        
        # Mask inactive nodes so they are not selected
        cost[~active] = n + 1
        
        # Select vertex with minimum cost
        v = np.argmin(cost)
        
        # Assign score: higher score means picked earlier by the greedy decoder
        scores[v] = n - step
        
        # Remove v and its neighbors from the active set
        active[v] = False
        active[adjacency[v]] = False
        
    return scores
