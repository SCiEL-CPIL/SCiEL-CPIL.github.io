<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Start with complete 5-uniform on 9 vertices: 126 edges
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edges)
    
    def try_remove_edges(edges, n, time_limit):
        """Greedy edge removal: remove edge if hypergraph remains non-2-colorable."""
        edges = [list(e) for e in edges]
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(edges)):
                if time.time() > time_limit:
                    break
                remaining = edges[:i] + edges[i+1:]
                if not is_2_colorable(remaining, n):
                    edges = remaining
                    changed = True
                    break
        return edges
    
    # Phase 1: Reduce the complete 9-vertex hypergraph
    if time.time() < deadline - 40:
        reduced = try_remove_edges(best_edges, best_n, time_limit=20.0)
        if len(reduced) < best_count:
            best_count = len(reduced)
            best_edges = reduced
    
    # Phase 2: Try other n values with random initialization + reduction
    for n in [10, 11, 12]:
        if time.time() > deadline - 20:
            break
        
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        # Start with a dense random subset
        subset = [list(e) for e in all_edges[:min(len(all_edges), 80)]]
        
        if not is_2_colorable(subset, n):
            if time.time() < deadline - 15:
                reduced = try_remove_edges(subset, n, time_limit=10.0)
                if len(reduced) < best_count:
                    best_count = len(reduced)
                    best_n = n
                    best_edges = reduced
    
    # Phase 3: More random restarts on n=9
    if time.time() < deadline - 10:
        for trial in range(10):
            if time.time() > deadline - 10:
                break
            
            all_9 = list(combinations(range(9), 5))
            random.shuffle(all_9)
            
            current = []
            for e in all_9:
                if len(current) >= best_count:
                    break
                current.append(list(e))
                if not is_2_colorable(current, 9):
                    break
            
            if len(current) < best_count and not is_2_colorable(current, 9):
                reduced = try_remove_edges(current, 9, time_limit=5.0)
                if len(reduced) < best_count:
                    best_count = len(reduced)
                    best_n = 9
                    best_edges = reduced
    
    return {'n': best_n, 'edges': best_edges}
=======
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Start with complete 5-uniform on 9 vertices: 126 edges
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edges)
    
    def try_remove_edges(edges, n, time_limit):
        """Greedy edge removal: remove edge if hypergraph remains non-2-colorable."""
        edges = [list(e) for e in edges]
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(edges)):
                if time.time() > time_limit:
                    break
                remaining = edges[:i] + edges[i+1:]
                if not is_2_colorable(remaining, n):
                    edges = remaining
                    changed = True
                    break
        return edges
    
    # Phase 1: Reduce the complete 9-vertex hypergraph
    if time.time() < deadline - 40:
        reduced = try_remove_edges(best_edges, best_n, time_limit=20.0)
        if len(reduced) < best_count:
            best_count = len(reduced)
            best_edges = reduced
    
    # Phase 2: SAT-style greedy edge selection
    # Build a candidate pool and greedily add edges that make the system unsatisfiable
    def greedy_build(n, time_limit, max_edges=None):
        """Greedily select edges to make the hypergraph non-2-colorable."""
        if max_edges is None:
            max_edges = best_count - 1
        
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        current = []
        for e in all_edges:
            if time.time() > time_limit:
                break
            if len(current) >= max_edges:
                break
            
            candidate = current + [list(e)]
            if not is_2_colorable(candidate, n):
                current = candidate
        
        return current
    
    # Phase 2a: Try greedy build on n=9
    if time.time() < deadline - 30:
        built = greedy_build(9, time.time() + 15.0)
        if len(built) < best_count:
            reduced = try_remove_edges(built, 9, time_limit=10.0)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_n = 9
                best_edges = reduced
    
    # Phase 2b: Try greedy build on n=10, 11
    for n in [10, 11]:
        if time.time() > deadline - 20:
            break
        
        built = greedy_build(n, time.time() + 10.0)
        if len(built) < best_count:
            reduced = try_remove_edges(built, n, time_limit=8.0)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_n = n
                best_edges = reduced
    
    # Phase 3: More random restarts on n=9 with incremental construction
    if time.time() < deadline - 10:
        for trial in range(15):
            if time.time() > deadline - 10:
                break
            
            all_9 = list(combinations(range(9), 5))
            random.shuffle(all_9)
            
            current = []
            for e in all_9:
                if len(current) >= best_count:
                    break
                current.append(list(e))
                if not is_2_colorable(current, 9):
                    break
            
            if len(current) < best_count and not is_2_colorable(current, 9):
                reduced = try_remove_edges(current, 9, time_limit=5.0)
                if len(reduced) < best_count:
                    best_count = len(reduced)
                    best_n = 9
                    best_edges = reduced
    
    return {'n': best_n, 'edges': best_edges}
>>>>>>> REPLACE
