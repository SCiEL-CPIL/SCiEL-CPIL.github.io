def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Build two copies of PG(2,4) polarity graph
    n1, edges1 = _build_polarity_graph_q4()
    n2, edges2 = _build_polarity_graph_q4()
    
    n = n1 + n2
    edges = edges1[:]
    for u, v in edges2:
        edges.append([u + n1, v + n1])
    
    # Verify
    if _verify_graph(n, edges):
        return {'n': n, 'edges': edges}
    
    # If alpha >= 11, try to add edges between copies
    # ... (complex)
    
    # Fallback
    n0, edges0 = _build_polarity_graph_q5()
    if _verify_graph(n0, edges0):
        return {'n': n0, 'edges': edges0}
    
    # Final fallback
    edges_fallback = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges_fallback += [[a,b], [a,c], [b,c]]
    return {'n': 30, 'edges': edges_fallback}
