import itertools

def construct():
    n = 16
    # Partition vertices into two sets of equal size
    # This creates a complete bipartite graph K_{8,8}
    # which is known to be the densest graph on 16 vertices 
    # where every edge is critical for maintaining diameter 2.
    
    left = list(range(n // 2))
    right = list(range(n // 2, n))
    
    # Generate all edges between left and right partitions
    edges = [[u, v] for u, v in itertools.product(left, right)]
    
    # Ensure types are strictly Python int and list
    # (itertools.product yields tuples of ints, list comprehension makes lists)
    # But let's be explicit to satisfy "Types are checked exactly"
    final_edges = []
    for e in edges:
        # e is [u, v]
        # Ensure u and v are Python ints
        u = int(e[0])
        v = int(e[1])
        final_edges.append([u, v])
        
    return final_edges
