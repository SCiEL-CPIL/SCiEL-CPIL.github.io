import random
import time

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
        Uses a heuristic that prioritizes starting paths at odd-degree vertices
        to minimize the total number of paths.
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
        
        # Calculate initial degrees
        degrees = [0] * n
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
            
        while time.time() - start_time < max_time:
            remaining = set(edge_set)
            paths = []
            
            # Track remaining degrees to identify "odd" vertices dynamically
            rem_deg = degrees[:]
            
            while remaining:
                # Identify vertices with odd remaining degree
                # These are the natural candidates for path endpoints
                odd_vertices = [i for i in range(n) if rem_deg[i] % 2 == 1 and rem_deg[i] > 0]
                
                if odd_vertices:
                    # Pick a random odd vertex to start
                    u = random.choice(odd_vertices)
                    # Pick a random unused edge incident to u
                    neighbors = [v for v in adj[u] if (u, v) in remaining or (v, u) in remaining]
                    if not neighbors:
                        # Should not happen if rem_deg > 0
                        continue
                    v = random.choice(neighbors)
                    edge = (u, v) if (u, v) in remaining else (v, u)
                    path = [u, v]
                    remaining.discard(edge)
                    rem_deg[u] -= 1
                    rem_deg[v] -= 1
                else:
                    # All degrees are even. Pick any remaining edge
                    # This will form a cycle, which we can treat as a path?
                    # No, a cycle is not a simple path in the strict sense (start != end).
                    # But if we break it, it becomes a path.
                    # However, if all degrees are even, we can form a single Eulerian circuit.
                    # Breaking it into paths: we can just pick an edge, traverse the cycle,
                    # and treat the whole cycle as a path? No, path must be simple (no repeated vertices).
                    # If the graph is Eulerian, it might not have a Hamiltonian cycle.
                    # So we might need multiple paths even if all degrees are even?
                    # Actually, if all degrees are even, we can decompose into cycles.
                    # Each cycle can be a path if we break it at one edge?
                    # No, a cycle is a closed path. A simple path is open.
                    # We can take a cycle and remove one edge to make it a path.
                    # But we must cover all edges.
                    # So if we have a cycle, we can make it a path by "opening" it.
                    # This uses 1 path for the cycle.
                    
                    # Pick a random edge
                    start_edge = random.choice(list(remaining))
                    u, v = start_edge
                    path = [u, v]
                    remaining.discard(start_edge)
                    rem_deg[u] -= 1
                    rem_deg[v] -= 1

                # Extend the path greedily
                # We extend from both ends if possible
                # To ensure simplicity, we maintain a set of vertices in the path
                path_set = set(path)
                left_end = path[0]
                right_end = path[-1]
                
                while True:
                    extended = False
                    
                    # Try extending right
                    right_neighbors = [w for w in adj[right_end] 
                                       if (right_end, w) in remaining or (w, right_end) in remaining
                                       and w not in path_set]
                    # Prioritize neighbors with odd remaining degree?
                    # If we extend to an odd vertex, it becomes even (internal), which is good.
                    # If we extend to an even vertex, it becomes odd (endpoint), which is bad if we want to minimize endpoints.
                    # So prefer odd neighbors.
                    odd_right = [w for w in right_neighbors if rem_deg[w] % 2 == 1]
                    if odd_right:
                        w = random.choice(odd_right)
                    elif right_neighbors:
                        w = random.choice(right_neighbors)
                    else:
                        w = None
                    
                    if w is not None:
                        edge = (right_end, w) if (right_end, w) in remaining else (w, right_end)
                        path.append(w)
                        path_set.add(w)
                        remaining.discard(edge)
                        rem_deg[right_end] -= 1
                        rem_deg[w] -= 1
                        right_end = w
                        extended = True
                    
                    # Try extending left
                    left_neighbors = [x for x in adj[left_end] 
                                      if (left_end, x) in remaining or (x, left_end) in remaining
                                      and x not in path_set]
                    odd_left = [x for x in left_neighbors if rem_deg[x] % 2 == 1]
                    if odd_left:
                        x = random.choice(odd_left)
                    elif left_neighbors:
                        x = random.choice(left_neighbors)
                    else:
                        x = None

                    if x is not None:
                        edge = (left_end, x) if (left_end, x) in remaining else (x, left_end)
                        path.insert(0, x)
                        path_set.add(x)
                        remaining.discard(edge)
                        rem_deg[left_end] -= 1
                        rem_deg[x] -= 1
                        left_end = x
                        extended = True

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
        paths = decompose_graph(edges, n, max_time=10.0)
        results.append(paths)
        
    return results
