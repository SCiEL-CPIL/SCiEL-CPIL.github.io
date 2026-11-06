import random
import time

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Helper: check if adding (u, v, c) creates a non-rainbow 4-cycle
    def is_valid_addition(u, v, c, graph):
        # A 4-cycle containing edge (u,v) is u-v-w-x-u
        # where w is neighbor of v, x is neighbor of u, and (w,x) is an edge.
        
        # Get neighbors of v and their edge colors
        nbrs_v = {}
        for (a, b), col in graph.items():
            if a == v:
                nbrs_v[b] = col
            elif b == v:
                nbrs_v[a] = col
        
        # Get neighbors of u and their edge colors
        nbrs_u = {}
        for (a, b), col in graph.items():
            if a == u:
                nbrs_u[b] = col
            elif b == u:
                nbrs_u[a] = col
        
        for w, col_vw in nbrs_v.items():
            if w == u:
                continue
            for x, col_xu in nbrs_u.items():
                if x == v or x == w:
                    continue
                key_wx = (min(w, x), max(w, x))
                if key_wx in graph:
                    col_wx = graph[key_wx]
                    cols = [c, col_vw, col_wx, col_xu]
                    if len(set(cols)) < 4:
                        return False
        return True

    def find_valid_color(u, v, graph):
        for c in range(num_colors):
            if is_valid_addition(u, v, c, graph):
                return c
        return -1

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    best_edge_count = 0
    best_graph = {}

    # Multiple random restarts with greedy construction
    max_restarts = 20
    restart = 0
    while time.time() < deadline - 10 and restart < max_restarts:
        restart += 1
        graph = {}
        edges_list = list(all_edges)
        random.shuffle(edges_list)
        
        for u, v in edges_list:
            if time.time() > deadline - 5:
                break
            c = find_valid_color(u, v, graph)
            if c != -1:
                graph[(u, v)] = c
        
        if len(graph) > best_edge_count:
            best_edge_count = len(graph)
            best_graph = dict(graph)
    
    # Local search: try to add missing edges
    current_graph = dict(best_graph)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        missing_edges = [(u, v) for (u, v) in all_edges if (u, v) not in current_graph]
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if time.time() > deadline - 5:
                break
            e_key = (u, v)
            valid_color = -1
            for c in range(num_colors):
                current_graph[e_key] = c
                if is_valid_addition(u, v, c, {k: val for k, val in current_graph.items() if k != e_key}):
                    valid_color = c
                    break
                else:
                    del current_graph[e_key]
            
            if valid_color != -1:
                current_graph[e_key] = valid_color
                improved = True
                break  # Restart the scan since graph changed
    
    if len(current_graph) > best_edge_count:
        best_graph = current_graph

    result = []
    for (u, v), c in best_graph.items():
        result.append([u, v, c])
    
    return result
