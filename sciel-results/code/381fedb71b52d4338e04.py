def minimize_edges(n, edges):
    current_edges = set(tuple(e) for e in edges)
    improved = True
    while improved:
        improved = False
        # Shuffle for randomness
        edge_list = list(current_edges)
        random.shuffle(edge_list)
        
        for e in edge_list:
            if e not in current_edges:
                continue
            candidate = current_edges - {e}
            checker = HypergraphChecker(n, list(candidate))
            if not checker.is_2_colorable():
                current_edges.remove(e)
                improved = True
                # Don't break, keep checking other edges in this pass?
                # Removing e changes the graph. Other edges might now be removable.
                # So we should continue the loop.
    return [list(e) for e in current_edges]
