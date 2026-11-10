import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120  # seconds
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def solve_path_cover(edges):
        if not edges:
            return []
        
        # Map edges to indices
        edge_indices = {e: i for i, e in enumerate(edges)}
        m = len(edges)
        
        # Build Line Graph Adjacency
        # Two edges are adjacent if they share a vertex
        # L_adj[i] is a set of j such that edges[i] and edges[j] share a vertex
        L_adj = [set() for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    L_adj[i].add(j)
                    L_adj[j].add(i)
        
        # Collect all edges in Line Graph for ILP
        l_edges = []
        for i in range(m):
            for j in L_adj[i]:
                if i < j:
                    l_edges.append((i, j))
        
        if not l_edges:
            # No connections, each edge is a path
            return [[list(e)] for e in edges]
        
        num_l_edges = len(l_edges)
        
        # ILP Formulation
        # Maximize sum(x)
        # c = -1 for all variables (minimize -sum)
        c = np.full(num_l_edges, -1.0)
        
        # Constraints: For each node i in L, sum of x_ij <= 1
        # A_ub, b_ub
        A = np.zeros((m, num_l_edges))
        b = np.ones(m)
        for k, (i, j) in enumerate(l_edges):
            A[i, k] = 1
            A[j, k] = 1
        
        constraints = [LinearConstraint(A, 0, 1)]
        bounds = Bounds(0, 1)
        
        try:
            # Use milp if available
            res = milp(c=c, constraints=constraints, integrality=np.ones(num_l_edges), bounds=bounds)
            
            if res.status == 0: # Optimal
                x = res.x
                # Extract matching
                matching = set()
                for k, val in enumerate(x):
                    if val > 0.5:
                        i, j = l_edges[k]
                        matching.add((i, j))
                        
                # Reconstruct paths
                # Build adjacency of the matching graph
                match_adj = {i: [] for i in range(m)}
                for i, j in matching:
                    match_adj[i].append(j)
                    match_adj[j].append(i)
                
                visited = [False] * m
                paths = []
                
                for i in range(m):
                    if not visited[i]:
                        # Start a new component
                        path_edges = []
                        current = i
                        visited[current] = True
                        path_edges.append(current)
                        
                        # Traverse
                        while True:
                            nexts = match_adj[current]
                            next_unvisited = None
                            for n in nexts:
                                if not visited[n]:
                                    next_unvisited = n
                                    break
                            
                            if next_unvisited is None:
                                break
                            
                            current = next_unvisited
                            visited[current] = True
                            path_edges.append(current)
                        
                        # Now path_edges is a list of edge indices forming a path or cycle in L
                        # We need to map this to vertices in G
                        # The edges are connected. We need to order them.
                        # Since it's a path in L, the edges are linked.
                        # Let's reconstruct the vertex sequence.
                        
                        if len(path_edges) == 1:
                            e = edges[path_edges[0]]
                            paths.append(list(e))
                        else:
                            # Determine the order of edges
                            # We have a chain of edges.
                            # Let's pick the first edge e0.
                            # The next edge e1 shares a vertex with e0.
                            # We can build the path by extending.
                            
                            # Get the first edge
                            e0_idx = path_edges[0]
                            u0, v0 = edges[e0_idx]
                            
                            # We need to figure out the direction.
                            # Let's try to build the path from one end to the other.
                            # Find an "end" of the path in L.
                            # An end is an edge index in path_edges that has only 1 neighbor in the matching.
                            ends = []
                            for idx in path_edges:
                                # Count neighbors in path_edges
                                neigh_count = 0
                                for n in match_adj[idx]:
                                    if n in path_edges: # This check is O(N^2) but N is small
                                        neigh_count += 1
                                if neigh_count == 1:
                                    ends.append(idx)
                            
                            if len(ends) == 2:
                                start_idx = ends[0]
                                end_idx = ends[1]
                            else:
                                # It's a cycle
                                start_idx = path_edges[0]
                                end_idx = None
                            
                            # Build path
                            current_edge = start_idx
                            path_vertices = []
                            u, v = edges[current_edge]
                            
                            # Determine start direction
                            # Look at the next edge
                            next_edge = None
                            for n in match_adj[current_edge]:
                                if n in path_edges and n != start_idx: # Rough check
                                    # Ensure n is the one we want
                                    # We need to find the neighbor in the sequence
                                    pass
                            
                            # Simpler reconstruction:
                            # We have a set of edges.
                            # Let's just collect all vertices and sort them? No.
                            # Let's trace.
                            
                            # Create a local graph of the edges in this path
                            # Nodes are vertices of G
                            # Edges are the edges in path_edges
                            
                            # Find the endpoints of the path in G
                            # A vertex is an endpoint if it has degree 1 in the subgraph formed by path_edges
                            # Unless it's a cycle, then all have degree 2.
                            
                            deg = {}
                            for idx in path_edges:
                                u, v = edges[idx]
                                deg[u] = deg.get(u, 0) + 1
                                deg[v] = deg.get(v, 0) + 1
                            
                            if len(ends) == 2:
                                # Path
                                # Find the start vertex
                                start_v = None
                                for v in deg:
                                    if deg[v] == 1:
                                        start_v = v
                                        break
                                if start_v is None:
                                    start_v = u0 # Fallback
                                
                                path_vertices.append(start_v)
                                prev_v = start_v
                                current_v = start_v
                                used_edges = set()
                                
                                while True:
                                    # Find next edge
                                    next_v = None
                                    for idx in path_edges:
                                        if idx in used_edges:
                                            continue
                                        eu, ev = edges[idx]
                                        if eu == current_v:
                                            next_v = ev
                                            used_edges.add(idx)
                                            break
                                        elif ev == current_v:
                                            next_v = eu
                                            used_edges.add(idx)
                                            break
                                    
                                    if next_v is None:
                                        break
                                    
                                    current_v = next_v
                                    path_vertices.append(current_v)
                                    
                                    if end_idx is not None:
                                        # Check if we reached the end
                                        # The end vertex is the other endpoint
                                        end_v = None
                                        for v in deg:
                                            if deg[v] == 1 and v != start_v:
                                                end_v = v
                                                break
                                        if current_v == end_v:
                                            break
                            else:
                                # Cycle
                                # Start at any vertex
                                start_v = u0
                                path_vertices.append(start_v)
                                current_v = start_v
                                used_edges = set()
                                
                                for _ in range(len(path_edges)):
                                    next_v = None
                                    for idx in path_edges:
                                        if idx in used_edges:
                                            continue
                                        eu, ev = edges[idx]
                                        if eu == current_v:
                                            next_v = ev
                                            used_edges.add(idx)
                                            break
                                        elif ev == current_v:
                                            next_v = eu
                                            used_edges.add(idx)
                                            break
                                    
                                    if next_v is None:
                                        break
                                    
                                    current_v = next_v
                                    path_vertices.append(current_v)
                                    # Stop if we return to start
                                    if current_v == start_v and len(used_edges) == len(path_edges):
                                        # Remove the last vertex to make it a simple path?
                                        # A cycle v1..vk v1 is not a simple path.
                                        # We output v1..vk
                                        path_vertices.pop()
                                        break
                            
                            paths.append(path_vertices)
                return paths
            else:
                # Fallback to greedy if ILP fails
                return greedy_matching(edges, L_adj)
        except Exception:
            return greedy_matching(edges, L_adj)
    
    def greedy_matching(edges, L_adj):
        m = len(edges)
        if m == 0: return []
        
        # Greedy matching in Line Graph
        # Randomize order of edges in L
        l_edges = []
        for i in range(m):
            for j in L_adj[i]:
                if i < j:
                    l_edges.append((i, j))
        
        random.shuffle(l_edges)
        
        matched = [False] * m
        matching = []
        
        for i, j in l_edges:
            if not matched[i] and not matched[j]:
                matching.append((i, j))
                matched[i] = True
                matched[j] = True
        
        # Reconstruct paths from matching (same logic as above)
        # We can reuse the reconstruction logic by calling a helper or duplicating
        # For simplicity, let's just return the greedy result
        # To avoid code duplication, we can wrap the reconstruction in a function
        # But given time, let's just implement a simple path finder for the matching
        
        match_adj = {i: [] for i in range(m)}
        for i, j in matching:
            match_adj[i].append(j)
            match_adj[j].append(i)
        
        visited = [False] * m
        paths = []
        
        for i in range(m):
            if not visited[i]:
                path_edges = []
                current = i
                visited[current] = True
                path_edges.append(current)
                
                while True:
                    next_unvisited = None
                    for n in match_adj[current]:
                        if not visited[n]:
                            next_unvisited = n
                            break
                    if next_unvisited is None:
                        break
                    current = next_unvisited
                    visited[current] = True
                    path_edges.append(current)
                
                # Reconstruct vertices
                # Same logic as before
                if len(path_edges) == 1:
                    paths.append(list(edges[path_edges[0]]))
                else:
                    deg = {}
                    for idx in path_edges:
                        u, v = edges[idx]
                        deg[u] = deg.get(u, 0) + 1
                        deg[v] = deg.get(v, 0) + 1
                    
                    ends = [v for v in deg if deg[v] == 1]
                    
                    if len(ends) == 2:
                        start_v = ends[0]
                        end_v = ends[1]
                        path_vertices = [start_v]
                        current_v = start_v
                        used_edges = set()
                        
                        while current_v != end_v:
                            next_v = None
                            for idx in path_edges:
                                if idx in used_edges: continue
                                eu, ev = edges[idx]
                                if eu == current_v:
                                    next_v = ev
                                    used_edges.add(idx)
                                    break
                                elif ev == current_v:
                                    next_v = eu
                                    used_edges.add(idx)
                                    break
                            if next_v is None: break
                            current_v = next_v
                            path_vertices.append(current_v)
                        paths.append(path_vertices)
                    else:
                        # Cycle
                        start_v = list(deg.keys())[0]
                        path_vertices = [start_v]
                        current_v = start_v
                        used_edges = set()
                        for _ in range(len(path_edges)):
                            next_v = None
                            for idx in path_edges:
                                if idx in used_edges: continue
                                eu, ev = edges[idx]
                                if eu == current_v:
                                    next_v = ev
                                    used_edges.add(idx)
                                    break
                                elif ev == current_v:
                                    next_v = eu
                                    used_edges.add(idx)
                                    break
                            if next_v is None: break
                            current_v = next_v
                            path_vertices.append(current_v)
                            if current_v == start_v and len(used_edges) == len(path_edges):
                                path_vertices.pop()
                                break
                        paths.append(path_vertices)
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline - 10:
            # Emergency fallback: individual edges
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        paths = solve_path_cover(edges)
        results.append(paths)
        
    return results
