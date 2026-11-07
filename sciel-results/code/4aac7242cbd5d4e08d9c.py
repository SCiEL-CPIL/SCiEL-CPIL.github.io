def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    
    # Phase 1: Try to find a 3-coloring with a limited time budget
    # 2 colors are impossible for a graph with cycles.
    sub_deadline_3 = start_time + 25
    result = _csp_solve_fast(3, n, edges, adj, sub_deadline_3)
    if result is not None and _validate(result, edges):
        return [int(c) for c in result]
    
    # Phase 2: Try to find a 4-coloring
    # Use remaining time
    result = _csp_solve_fast(4, n, edges, adj, deadline)
    if result is not None and _validate(result, edges):
        return [int(c) for c in result]
        
    # Phase 3: Fallback to 5 colors if 4 fails
    result = _csp_solve_fast(5, n, edges, adj, deadline)
    if result is not None and _validate(result, edges):
        return [int(c) for c in result]
        
    # Fallback: proper coloring with n colors
    return [int(i) for i in range(n)]
