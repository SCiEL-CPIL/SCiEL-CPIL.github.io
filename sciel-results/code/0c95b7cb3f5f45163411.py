import time
from collections import deque

def construct():
    start_time = time.time()
    n = 16
    # We'll try to find a graph with more edges than K_{8,8} (64 edges)
    # that still has the property that every edge is essential.
    
    # Let's think about this differently. The current solution uses K_{8,8} with 64 edges.
    # Can we do better? Let's try to construct a graph with more edges.
    
    # One approach: start with a dense graph and remove edges that are not essential.
    # But that's computationally expensive.
    
    # Another approach: try specific constructions.
    
    # Let's try a graph that is almost complete but with some structure.
    
    # Actually, let me think about what makes an edge essential.
    # An edge (u,v) is essential if removing it either:
    # 1. Disconnects the graph, or
    # 2. Increases the diameter above 2.
    
    # For a graph with diameter 2, edge (u,v) is essential if after removing it,
    # either u and v become disconnected, or the shortest path between u and v becomes > 2.
    
    # In a complete graph K_n, removing any edge (u,v) gives a path of length 2 (u-w-v for any w != u,v),
    # so the diameter remains 2. Thus no edge is essential in K_n.
    
    # So we need a graph that is dense but not complete, where every edge is "critical" for maintaining diameter 2.
    
    # Let's try: take K_16 and remove a matching. Then check if every remaining edge is essential.
    # Actually, that might not work either.
    
    # Let me try a different approach: use a known construction.
    
    # Consider the complement of a perfect matching in K_16. This has 16*15/2 - 8 = 120 - 8 = 112 edges.
    # But is every edge essential? Probably not.
    
    # Let me think about this more carefully.
    
    # Actually, let's try to search. We'll use a heuristic approach.
    
    # Start with K_16, then remove edges one by one, checking if the remaining graph still has
    # the property that every edge is essential.
    
    # This is expensive, so let's be smart about it.
    
    # Actually, let me reconsider. The problem says "maximize edge count". The current answer is 64.
    # Can we do better?
    
    # Let's try: take two disjoint K_8's, and add some edges between them.
    # But we need diameter 2, so we need enough cross-edges.
    
    # Actually, let me try a completely different construction.
    
    # Consider a graph where vertices are partitioned into groups, and we have specific connectivity.
    
    # Let me try: 4 groups of 4 vertices. Within each group, complete graph.
    # Between groups, add edges in a specific pattern.
    
    # Actually, this is getting complex. Let me just try to verify if K_{8,8} is optimal
    # or if we can find something better with a search.
    
    # For now, let's stick with K_{8,8} but write it more cleanly.
    
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([int(u), int(v)])
    
    # Check if we have time to try something better
    if time.time() - start_time < 30:
        # Try to find a better solution
        better = try_better(n)
        if better is not None and len(better) > len(edges):
            edges = better
    
    return edges

def try_better(n):
    """Try to find a graph with more edges than K_{8,8} that satisfies the constraints."""
    # This is a placeholder for a more sophisticated search
    # For now, return None to indicate no better solution found
    
    # Let's try a specific construction:
    # Take the complete graph K_16 and remove edges to make every edge essential.
    
    # Actually, let me think about this differently.
    
    # A graph where every edge is essential for diameter 2 is called a "diameter-2 edge-critical" graph.
    
    # Let me try: take K_{8,8} and add some edges within the partitions.
    # If we add an edge (u1, u2) where u1, u2 are both in the left partition,
    # then we need to check if all edges are still essential.
    
    # Actually, adding edges might make some edges non-essential.
    
    # Let me just return None for now.
    return None
