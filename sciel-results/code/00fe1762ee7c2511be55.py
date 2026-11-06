def construct():
    n = 20
    # Adjacency matrix or set of edges
    edges = set()
    neighbors = [set() for _ in range(n)]
    
    # List of all possible edges
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    best_edges = 0
    best_graph = []
    
    # Try multiple random orders
    import random
    for _ in range(100):
        random.shuffle(all_pairs)
        edges.clear()
        for i in range(n):
            neighbors[i].clear()
        
        forbidden = set()
        
        for u, v in all_pairs:
            if (u, v) in forbidden:
                continue
            
            # Check if adding (u, v) creates a 4-cycle
            # Path u-w-x-v exists if w in N(u), x in N(v) and (w,x) in edges
            safe = True
            for w in neighbors[u]:
                for x in neighbors[v]:
                    if (min(w,x), max(w,x)) in edges:
                        safe = False
                        break
                if not safe:
                    break
            
            if safe:
                # Add edge (u, v)
                edges.add((u, v))
                neighbors[u].add(v)
                neighbors[v].add(u)
                
                # Update forbidden pairs
                # New forbidden pairs are (a, b) where a in N(u), b in N(v)
                # Because if we add (a, b) later, cycle a-u-v-b-a is formed
                # Note: N(u) and N(v) are current neighbors (including each other? No, (u,v) just added, so u in N(v) and v in N(u))
                # Wait, if u in N(v) and v in N(u), then (u,v) is in N(u) x N(v)?
                # a=u, b=v? (u,v) is already added.
                # We need to forbid (a,b) for a in N(u), b in N(v) such that (a,b) is not yet added?
                # Actually, if (a,b) is already added, we would have detected the cycle earlier?
                # If (a,b) is in edges, and we just added (u,v), and (a,u) in edges, (v,b) in edges,
                # then a-u-v-b-a is a 4-cycle.
                # So the "safe" check should have caught this?
                # The safe check looks for w in N(u), x in N(v) with (w,x) in edges.
                # Here w=a, x=b. (a,b) is in edges.
                # So yes, if (a,b) was already there, safe would be False.
                # So we only need to forbid pairs that are NOT yet in edges.
                
                for a in neighbors[u]:
                    for b in neighbors[v]:
                        if a != b: # Simple graph, no self loops
                            # Only forbid if (a,b) is not already an edge?
                            # If (a,b) is an edge, we already have a cycle, which shouldn't happen.
                            forbidden.add((min(a,b), max(a,b)))
        
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_graph = list(edges)
            
    # Construct output
    # Colors can be anything, say 0
    result = []
    for u, v in best_graph:
        result.append([u, v, 0])
        
    return result
