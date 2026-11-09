# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph with 40 vertices and 401 edges that minimizes the largest book size.
    
    Strategy: 
    1. Start with K_{20,20} (400 edges).
    2. Add k intra-part edges forming a star in Part A.
    3. Remove k-1 bipartite edges incident to the center of the star.
    4. This balances the book sizes between intra edges and bipartite edges.
    """
    n = 40
    part_a = list(range(20))
    part_b = list(range(20, 40))
    
    # Choose k = 10 or 11 to balance max(21-k, k)
    # k=10 -> max(11, 10) = 11
    # k=11 -> max(10, 11) = 11
    # Let's try k=10
    k = 10
    
    # Start with all bipartite edges
    edges = set()
    for a in part_a:
        for b in part_b:
            edges.add((a, b))
    
    # Select center and leaves in Part A
    center = 0
    leaves = list(range(1, 1 + k))
    
    # Add intra-A edges (Star)
    for leaf in leaves:
        edges.add((center, leaf))
    
    # Remove k-1 bipartite edges incident to center
    # We need to remove exactly k-1 edges to keep total count at 401
    # Start with 400, add k, remove k-1 -> 401
    edges_to_remove = k - 1
    
    # Remove edges (center, b) for b in part_b
    # We pick the first edges_to_remove vertices in part_b
    for i in range(edges_to_remove):
        b = part_b[i]
        edges.discard((center, b))
    
    # Verify edge count
    if len(edges) != 401:
        # Fallback if logic fails
        return [[u, v] for u, v in list(edges)[:401]]
        
    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
