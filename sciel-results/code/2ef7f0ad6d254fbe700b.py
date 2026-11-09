"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    n = P['n']
    all_edges = set()
    for e in P['edges']:
        all_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Build adjacency
    adj = {v: set() for v in range(n)}
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def find_hamiltonian_cycle(available_edges, start_time):
        """Find a Hamiltonian cycle using backtracking with pruning."""
        if time.time() - start_time > deadline - 5:
            return None
        
        # Check if graph is connected
        if not all_edges:
            return None
        
        # Find a start vertex with minimum degree for better pruning
        degrees = {v: len([e for e in available_edges if v in e]) for v in range(n)}
        # Actually compute degree from available edges
        deg = {v: 0 for v in range(n)}
        for u, v in available_edges:
            deg[u] += 1
            deg[v] += 1
        
        # All vertices must have degree >= 2 in available edges
        for v in range(n):
            if deg[v] < 2:
                return None
        
        # Start from vertex with minimum degree (>=2)
        start = min(range(n), key=lambda v: deg[v])
        
        # Build adjacency from available edges
        adj_avail = {v: set() for v in range(n)}
        for u, v in available_edges:
            adj_avail[u].add(v)
            adj_avail[v].add(u)
        
        # Backtracking
        path = [start]
        visited = {start}
        
        def backtrack():
            if time.time() - start_time > deadline - 5:
                return False
            
            if len(path) == n:
                # Check if last vertex connects to start
                last = path[-1]
                if start in adj_avail[last]:
                    return True
                return False
            
            current = path[-1]
            # Order neighbors by degree (Warnsdorff's rule)
            neighbors = sorted(adj_avail[current], key=lambda x: len(adj_avail[x]))
            
            for nb in neighbors:
                if nb in visited:
                    continue
                # Pruning: if nb is not start, check if it has enough unvisited neighbors
                if nb != start:
                    unvisited_neighbors = [x for x in adj_avail[nb] if x not in visited and x != current]
                    # If nb would be a dead end (only one way in, need at least one way out unless it's the last)
                    if len(path) < n - 1 and len(unvisited_neighbors) == 0:
                        continue
                    # If nb is not the last vertex to visit, it needs at least 2 connections total
                    # (one from current, one to next), so it needs at least 1 unvisited neighbor besides current
                    # Actually this is tricky. Let's use a simpler check.
                
                # Additional pruning: check connectivity of remaining unvisited vertices
                if len(path) < n - 1:
                    remaining = set(range(n)) - visited - {nb}
                    if remaining:
                        # Check if remaining + nb is connected (considering only available edges among them and to path)
                        # This is expensive, so do it only for larger remaining sets
                        if len(remaining) > 3:
                            # Simple BFS/DFS to check connectivity
                            from collections import deque
                            q = deque([nb])
                            seen = {nb}
                            while q:
                                cur = q.popleft()
                                for nb2 in adj_avail[cur]:
                                    if nb2 in seen:
                                        continue
                                    if nb2 in visited and nb2 != current:
                                        continue
                                    if nb2 in remaining or nb2 == nb:
                                        seen.add(nb2)
                                        q.append(nb2)
                            if len(seen) < len(remaining) + 1:
                                continue
                
                path.append(nb)
                visited.add(nb)
                
                if backtrack():
                    return True
                
                path.pop()
                visited.remove(nb)
            
            return False
        
        if backtrack():
            cycle = path[:]
            # The cycle is path[0] -> path[1] -> ... -> path[n-1] -> path[0]
            return cycle
        return None
    
    # Greedy approach: repeatedly find Hamiltonian cycles and remove their edges
    cycles = []
    available = set(all_edges)
    
    while time.time() - start_time < deadline - 10:
        if len(available) < n:  # Need at least n edges for a Hamiltonian cycle
            break
        
        cycle = find_hamiltonian_cycle(available, start_time)
        if cycle is None:
            break
        
        # Remove edges of this cycle from available
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge = (min(u, v), max(u, v))
            if edge in available:
                available.remove(edge)
        
        cycles.append(cycle)
    
    # Ensure we return at least 1 cycle (fallback)
    if not cycles:
        # Try a simpler approach: just find any Hamiltonian cycle
        cycle = find_hamiltonian_cycle(all_edges, start_time)
        if cycle:
            cycles.append(cycle)
        else:
            # Last resort: return a trivial valid cycle if possible
            # Find any 3-cycle
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        for w in adj[v]:
                            if w > v and w in adj[u]:
                                cycles.append([u, v, w])
                                break
                        if cycles:
                            break
                if cycles:
                    break
            if not cycles:
                # Should not happen with this graph
                cycles.append([0, 1, 2])
    
    # Convert to list of lists of ints
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
