def _reindex_edges(edges):
    if not edges:
        return 0, []
    
    vertices = set()
    for e in edges:
        vertices.update(e)
    
    # Map old vertex indices to new 0..k-1
    sorted_verts = sorted(vertices)
    mapping = {v: i for i, v in enumerate(sorted_verts)}
    
    new_edges = []
    for e in edges:
        new_e = [mapping[v] for v in e]
        # Ensure distinctness is preserved (mapping is bijective on the set)
        new_edges.append(new_e)
        
    return len(sorted_verts), new_edges

def construct():
    # ... existing search logic ...
    
    if best_edges is None:
        # Fallback: Complete hypergraph on 9 vertices
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
    else:
        # Reindex to ensure vertices are 0..n-1
        n, best_edges = _reindex_edges(best_edges)
        
    return {'n': n, 'edges': best_edges}
