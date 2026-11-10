import random
import time
import sys

def construct():
    """
    Decomposes three fixed connected graphs into edge-disjoint simple paths.
    Minimizes the total number of paths.
    """
    
    def instance_graph(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges, n, max_time=10.0):
        """
        Decomposes a graph given by its edge list into a list of simple paths.
        Uses a heuristic greedy approach with random restarts.
        """
        if not edges:
            return []
        
        best_paths = None
        best_count = float('inf')
        
        start_time = time.time()
        
        # Precompute adjacency list for the full graph
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        edge_set = set(edges)
        
        # Helper to pick a random edge biased towards high degree endpoints
        def pick_start_edge(remaining_edges, avail_deg):
            if not remaining_edges:
                return None
            # Bias towards edges with high degree endpoints
            # Sample a few candidates and pick the best
            candidates = random.sample(list(remaining_edges), min(10, len(remaining_edges)))
            best_e = None
            best_score = -1
            for e in candidates:
                u, v = e
                score = avail_deg[u] + avail_deg[v]
                # Add some randomness
                score += random.random() * 2
                if score > best_score:
                    best_score = score
                    best_e = e
            return best_e

        while time.time() - start_time < max_time:
            remaining = set(edge_set)
            paths = []
            
            # Track available degree for heuristic
            avail_deg = {i: len(adj[i]) for i in range(n)}
            # Actually, we need dynamic degree. 
            # It's expensive to update a dict for every edge removal in a tight loop.
            # Let's stick to a simpler heuristic: 
            # When extending, prefer neighbors with high *static* degree?
            # Or just use the size of the neighbor's adjacency list in the *remaining* graph?
            # To avoid O(E) updates, we can just use static degree as a proxy, 
            # or maintain a counter.
            
            # Let's maintain a counter for simplicity and correctness of heuristic.
            rem_deg = {i: len(adj[i]) for i in range(n)}
            # This is O(N) to init.
            
            while remaining:
                if not remaining:
                    break
                
                start_edge = pick_start_edge(remaining, rem_deg)
                if start_edge is None:
                    break
                    
                u, v = start_edge
                
                path = [u, v]
                remaining.discard(start_edge)
                rem_deg[u] -= 1
                rem_deg[v] -= 1
                
                # Extend the path greedily
                # Alternate ends or pick randomly?
                # Let's pick the end with more options (higher rem_deg)
                extend_left = rem_deg[u] > rem_deg[v]
                if rem_deg[u] == rem_deg[v]:
                    extend_left = random.random() < 0.5

                while True:
                    extended = False
                    
                    if not extend_left:
                        # Extend right (v)
                        # Get neighbors of v that have available edge to v
                        # We need to find w such that (v, w) in remaining
                        # We can iterate adj[v] and check
                        candidates = []
                        for w in adj[v]:
                            if (v, w) in remaining or (w, v) in remaining:
                                # Check if edge is in remaining
                                edge = (v, w) if (v, w) in remaining else (w, v)
                                if edge in remaining:
                                    if w not in path:
                                        # Heuristic: prefer w with high remaining degree
                                        candidates.append((w, rem_deg[w]))
                        
                        if candidates:
                            # Pick best candidate
                            # Weighted random or max?
                            # Max with noise
                            best_w = None
                            best_score = -1
                            for w, deg in candidates:
                                score = deg + random.random() * 3
                                if score > best_score:
                                    best_score = score
                                    best_w = w
                            
                            w = best_w
                            edge = (v, w) if (v, w) in remaining else (w, v)
                            path.append(w)
                            remaining.discard(edge)
                            rem_deg[v] -= 1
                            rem_deg[w] -= 1
                            v = w
                            extended = True
                            # Update extend direction preference
                            extend_left = rem_deg[u] > rem_deg[v]
                            if rem_deg[u] == rem_deg[v]:
                                extend_left = random.random() < 0.5
                            continue
                        else:
                            extend_left = True
                    else:
                        # Extend left (u)
                        candidates = []
                        for x in adj[u]:
                            if (u, x) in remaining or (x, u) in remaining:
                                edge = (u, x) if (u, x) in remaining else (x, u)
                                if edge in remaining:
                                    if x not in path:
                                        candidates.append((x, rem_deg[x]))
                        
                        if candidates:
                            best_x = None
                            best_score = -1
                            for x, deg in candidates:
                                score = deg + random.random() * 3
                                if score > best_score:
                                    best_score = score
                                    best_x = x
                            
                            x = best_x
                            edge = (u, x) if (u, x) in remaining else (x, u)
                            path.insert(0, x)
                            remaining.discard(edge)
                            rem_deg[u] -= 1
                            rem_deg[x] -= 1
                            u = x
                            extended = True
                            extend_left = rem_deg[u] > rem_deg[v]
                            if rem_deg[u] == rem_deg[v]:
                                extend_left = random.random() < 0.5
                            continue
                        else:
                            extend_left = False
                    
                    if not extended:
                        break

                paths.append(path)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]

        if best_paths is None:
            return [[u, v] for u, v in edges]
            
        return best_paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    # Allocate time budget
    # Total 150s. Leave 10s margin. 140s / 3 graphs = 46s per graph.
    # Use 40s per graph to be safe.
    
    for n, p, seed in specs:
        edges = instance_graph(n, p, seed)
        paths = decompose_graph(edges, n, max_time=40.0)
        results.append(paths)
        
    return results

# EVOLVE-BLOCK-START
def construct():
    """
    Decomposes three fixed connected graphs into edge-disjoint simple paths.
    Minimizes the total number of paths.
    """
    
    def instance_graph(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges, n, max_time=10.0):
        """
        Decomposes a graph given by its edge list into a list of simple paths.
        Uses a heuristic greedy approach with random restarts.
        """
        if not edges:
            return []
        
        best_paths = None
        best_count = float('inf')
        
        start_time = time.time()
        
        # Precompute adjacency list for the full graph
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        edge_set = set(edges)
        
        def pick_start_edge(remaining_edges, rem_deg):
            if not remaining_edges:
                return None
            # Bias towards edges with high degree endpoints
            candidates = random.sample(list(remaining_edges), min(10, len(remaining_edges)))
            best_e = None
            best_score = -1
            for e in candidates:
                u, v = e
                score = rem_deg[u] + rem_deg[v]
                score += random.random() * 2
                if score > best_score:
                    best_score = score
                    best_e = e
            return best_e

        while time.time() - start_time < max_time:
            remaining = set(edge_set)
            paths = []
            
            rem_deg = {i: len(adj[i]) for i in range(n)}
            
            while remaining:
                if not remaining:
                    break
                
                start_edge = pick_start_edge(remaining, rem_deg)
                if start_edge is None:
                    break
                    
                u, v = start_edge
                
                path = [u, v]
                remaining.discard(start_edge)
                rem_deg[u] -= 1
                rem_deg[v] -= 1
                
                extend_left = rem_deg[u] > rem_deg[v]
                if rem_deg[u] == rem_deg[v]:
                    extend_left = random.random() < 0.5

                while True:
                    extended = False
                    
                    if not extend_left:
                        candidates = []
                        for w in adj[v]:
                            edge = (v, w) if (v, w) in remaining else (w, v)
                            if edge in remaining:
                                if w not in path:
                                    candidates.append((w, rem_deg[w]))
                        
                        if candidates:
                            best_w = None
                            best_score = -1
                            for w, deg in candidates:
                                score = deg + random.random() * 3
                                if score > best_score:
                                    best_score = score
                                    best_w = w
                            
                            w = best_w
                            edge = (v, w) if (v, w) in remaining else (w, v)
                            path.append(w)
                            remaining.discard(edge)
                            rem_deg[v] -= 1
                            rem_deg[w] -= 1
                            v = w
                            extended = True
                            extend_left = rem_deg[u] > rem_deg[v]
                            if rem_deg[u] == rem_deg[v]:
                                extend_left = random.random() < 0.5
                            continue
                        else:
                            extend_left = True
                    else:
                        candidates = []
                        for x in adj[u]:
                            edge = (u, x) if (u, x) in remaining else (x, u)
                            if edge in remaining:
                                if x not in path:
                                    candidates.append((x, rem_deg[x]))
                        
                        if candidates:
                            best_x = None
                            best_score = -1
                            for x, deg in candidates:
                                score = deg + random.random() * 3
                                if score > best_score:
                                    best_score = score
                                    best_x = x
                            
                            x = best_x
                            edge = (u, x) if (u, x) in remaining else (x, u)
                            path.insert(0, x)
                            remaining.discard(edge)
                            rem_deg[u] -= 1
                            rem_deg[x] -= 1
                            u = x
                            extended = True
                            extend_left = rem_deg[u] > rem_deg[v]
                            if rem_deg[u] == rem_deg[v]:
                                extend_left = random.random() < 0.5
                            continue
                        else:
                            extend_left = False
                    
                    if not extended:
                        break

                paths.append(path)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]

        if best_paths is None:
            return [[u, v] for u, v in edges]
            
        return best_paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for n, p, seed in specs:
        edges = instance_graph(n, p, seed)
        paths = decompose_graph(edges, n, max_time=40.0)
        results.append(paths)
        
    return results
# EVOLVE-BLOCK-END
