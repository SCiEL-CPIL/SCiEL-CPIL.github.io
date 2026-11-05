import numpy as np

def priority(adjacency):
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
        
    # Compute Core Numbers
    # We use a simplified iterative peeling
    # deg is current degree
    deg = adjacency.sum(axis=1).astype(int)
    core = np.zeros(n, dtype=int)
    
    # To do this efficiently without networkx:
    # We can simulate the peeling process.
    # However, a simpler proxy for "peripheralness" is just Degree.
    # Core number is monotonic with degree but captures structure.
    
    # Let's try a different distinct approach: 
    # **Weighted Degree where weight is 1/(1+degree_of_neighbor)**
    # This penalizes being connected to hubs more strongly?
    
    # Actually, let's look at the current score: -deg + mean_neigh_deg.
    # This is linear.
    # What if we use **Log** or **Square**?
    
    # Let's try **Core Number** properly.
    # It's a standard algorithm.
    
    # Batagelj-Zaversnik algorithm implementation
    # 1. Sort vertices by degree
    # 2. Use a queue for vertices with degree <= current k
    
    # This might be overkill for 200 nodes, but it's a distinct approach.
    
    # Alternative Distinct Approach: **Betweenness Centrality** (approximate)
    # Too slow.
    
    # Let's try **Local Clustering Coefficient**.
    # Vertices with low clustering are "bridges".
    # Picking bridges might be good?
    
    # Let's go with **Core Number**.
    
    # Implementation of Core Number for small n
    deg = adjacency.sum(axis=1).copy()
    core = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    
    # Simple O(n^2) peeling, fine for n=200
    # We repeatedly remove the vertex with the minimum degree among remaining vertices
    # and assign it the current minimum degree as its core number?
    # No, that's not quite right.
    
    # Correct simple peeling:
    # 1. While there are vertices:
    #    a. Find min degree d_min among remaining.
    #    b. Assign core number d_min to all remaining vertices with degree <= d_min?
    #       No, that's not standard.
    
    # Standard K-Core:
    # The k-core is the maximal subgraph where every node has degree >= k.
    # Core number of v is the largest k such that v is in the k-core.
    
    # Algorithm:
    # 1. Start with all nodes.
    # 2. Find node with min degree. Let d be its degree.
    # 3. Remove all nodes with degree <= d. Assign core number d to them.
    # 4. Update degrees of neighbors.
    # 5. Repeat.
    
    # This is O(n^2) if we just scan for min each time. n=200 is 40,000 ops. Fast.
    
    deg_current = adjacency.sum(axis=1).astype(int)
    core_number = np.zeros(n, dtype=int)
    active = np.ones(n, dtype=bool)
    
    while active.any():
        # Find min degree among active
        active_degrees = np.where(active, deg_current, np.inf)
        d_min = np.min(active_degrees)
        
        # All active nodes with degree <= d_min get core number d_min
        mask = (active) & (deg_current <= d_min)
        core_number[mask] = d_min
        active[mask] = False
        
        # Update degrees of neighbors of removed nodes
        # For each removed node, decrement degree of its active neighbors
        removed_indices = np.where(mask)[0]
        for idx in removed_indices:
            neighbors = np.where(adjacency[idx] == 1)[0]
            # Only update if neighbor is still active
            for nb in neighbors:
                if active[nb]:
                    deg_current[nb] -= 1
                    
    # Score: We want to pick low core number first.
    # So score = -core_number.
    # To break ties, use -degree.
    score = -core_number.astype(float) - 0.01 * deg_current.astype(float)
    
    return score
