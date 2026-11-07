import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    deadline = time.time() + 55.0
    n = P['n']
    
    edge_set = set()
    adj = {i: set() for i in range(n)}
    for u, v in P['edges']:
        ek = (min(u,v), max(u,v))
        edge_set.add(ek)
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute edge index for quick lookup
    edge_to_idx = {}
    edge_list = []
    for i, (u, v) in enumerate(P['edges']):
        ek = (min(u,v), max(u,v))
        edge_to_idx[ek] = i
        edge_list.append(ek)
    
    def edge_key(u, v):
        return (min(u,v), max(u,v))
    
    def find_hamiltonian_cycle(available_edge_indices, start=0, time_limit=5.0):
        """Find a Hamiltonian cycle using only edges whose indices are in available_edge_indices."""
        t0 = time.time()
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        # Dynamic degree tracking
        deg = [0] * n
        for idx in available_edge_indices:
            u, v = edge_list[idx]
            deg[u] += 1
            deg[v] += 1
        
        # Precompute adjacency as sorted lists for faster iteration
        adj_list = {i: sorted(adj[i]) for i in range(n)}
        
        # Build availability check: for each vertex, list of available neighbors
        avail_neigh = {i: [] for i in range(n)}
        for idx in available_edge_indices:
            u, v = edge_list[idx]
            avail_neigh[u].append(v)
            avail_neigh[v].append(u)
        
        # Sort available neighbors by degree (most constrained first)
        for i in range(n):
            avail_neigh[i].sort(key=lambda x: deg[x])
        
        nodes_visited = 1
        nodes_remaining = n - 1
        
        def backtrack():
            nonlocal nodes_visited, nodes_remaining
            if time.time() - t0 > time_limit:
                return "timeout"
            
            if nodes_visited == n:
                last = path[-1]
                ek = edge_key(last, start)
                if ek in available_edge_indices:
                    return list(path)
                return None
            
            last = path[-1]
            # Get candidates from avail_neigh, filter visited
            cands = [nb for nb in avail_neigh[last] if not visited[nb]]
            
            if not cands:
                return None
            
            # Forced move: if only one candidate, must take it
            if len(cands) == 1:
                nb = cands[0]
                ek = edge_key(last, nb)
                if ek not in available_edge_indices:
                    return None
                
                # Check if nb will have degree >= 1 after removing this edge
                if deg[nb] - 1 < 1:
                    return None
                
                # Check all unvisited vertices still have deg >= 2 (except nb which needs >=1)
                for v in range(n):
                    if visited[v] or v == nb:
                        continue
                    if deg[v] < 2:
                        return None
                if deg[nb] - 1 < 1:
                    return None
                
                # Remove edge
                available_edge_indices.discard(ek)
                deg[last] -= 1
                deg[nb] -= 1
                
                path.append(nb)
                visited[nb] = True
                nodes_visited += 1
                nodes_remaining -= 1
                
                result = backtrack()
                
                # Undo
                path.pop()
                visited[nb] = False
                nodes_visited -= 1
                nodes_remaining += 1
                deg[last] += 1
                deg[nb] += 1
                available_edge_indices.add(ek)
                
                if result is not None and result != "timeout":
                    return result
                return None
            
            # Multiple candidates: try in order of most constrained
            # Sort by degree of neighbor (lower degree = more constrained)
            cands.sort(key=lambda x: deg[x])
            
            for nb in cands:
                ek = edge_key(last, nb)
                if ek not in available_edge_indices:
                    continue
                
                # Pruning: nb needs at least 1 other available edge after using (last, nb)
                if deg[nb] - 1 < 1:
                    continue
                
                # Check all unvisited vertices (except nb) still have deg >= 2
                valid = True
                for v in range(n):
                    if visited[v] or v == nb:
                        continue
                    if deg[v] < 2:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Remove edge
                available_edge_indices.discard(ek)
                deg[last] -= 1
                deg[nb] -= 1
                
                path.append(nb)
                visited[nb] = True
                nodes_visited += 1
                nodes_remaining -= 1
                
                result = backtrack()
                
                # Undo
                path.pop()
                visited[nb] = False
                nodes_visited -= 1
                nodes_remaining += 1
                deg[last] += 1
                deg[nb] += 1
                available_edge_indices.add(ek)
                
                if result is not None and result != "timeout":
                    return result
                
            return None
        
        result = backtrack()
        if result == "timeout":
            return None
        return result

    cycles = []
    available = set(range(len(edge_list)))  # Use edge indices
    
    max_cycles = 3
    for _ in range(max_cycles):
        if time.time() > deadline:
            break
        
        found = None
        
        # Try different starting vertices, but limit attempts
        for start in range(n):
            if time.time() > deadline:
                break
            remaining_time = deadline - time.time()
            if remaining_time < 0.5:
                break
            time_limit = min(remaining_time * 0.35, 10.0)
            # Make a copy since we modify available during search
            avail_copy = set(available)
            found = find_hamiltonian_cycle(avail_copy, start=start, time_limit=time_limit)
            if found is not None:
                break
        
        if found is None:
            break
            
        # Remove used edges from available
        for i in range(len(found)):
            u = found[i]
            v = found[(i+1) % len(found)]
            ek = edge_key(u, v)
            if ek in edge_to_idx:
                available.discard(edge_to_idx[ek])
        cycles.append(found)
        
        # Check if remaining graph can still support a Hamiltonian cycle
        min_deg = min(sum(1 for idx in available if edge_list[idx][0]==i or edge_list[idx][1]==i) for i in range(n))
        if min_deg < 2:
            break
        # Quick connectivity check
        if available:
            visited_conn = {0}
            stack = [0]
            while stack:
                node = stack.pop()
                for nb in adj[node]:
                    if nb not in visited_conn:
                        ek = edge_key(node, nb)
                        if ek in edge_to_idx and edge_to_idx[ek] in available:
                            visited_conn.add(nb)
                            stack.append(nb)
            if len(visited_conn) < n:
                break

    if len(cycles) == 3:
        result = [list(c) for c in cycles]
        # Verify all constraints before returning
        used_edges = set()
        for cyc in result:
            if len(set(cyc)) != n:
                return [list(range(n))]
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i+1) % len(cyc)]
                ek = edge_key(u, v)
                if ek in used_edges:
                    return [list(range(n))]
                used_edges.add(ek)
        return result
    elif cycles:
        return [list(c) for c in cycles]
    else:
        random.seed(42)
        for _ in range(50):
            avail_copy = set(range(len(edge_list)))
            found = find_hamiltonian_cycle(avail_copy, start=random.randint(0, n-1), time_limit=2.0)
            if found:
                return [found]
        return [list(range(n))]
# EVOLVE-BLOCK-END
