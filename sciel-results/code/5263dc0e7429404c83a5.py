    missing_edges = list(all_edges - edges)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for e in missing_edges:
            if e in edges:
                continue
            next_edges = current_edges + [e]
            if is_valid(next_edges):
                current_edges = next_edges
                edges.add(e)
                improved = True
