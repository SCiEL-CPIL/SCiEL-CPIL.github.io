import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    def find_hamiltonian_cycle(adj_map, start_node=0, deadline=None):
        if deadline is None:
            deadline = start_time + 50.0
            
        n_local = len(adj_map)
        visited = [False] * n_local
        visited[start_node] = True
        path = [start_node]
        
        def backtrack():
            if time.time() > deadline:
                return False
                
            current = path[-1]
            
            if len(path) == n_local:
                if start_node in adj_map[current]:
                    return True
                return False
            
            # Get unvisited neighbors
            unvisited_neighbors = [v for v in adj_map[current] if not visited[v]]
            
            if not unvisited_neighbors:
                return False
            
            # Warnsdorff's rule: sort by number of unvisited neighbors
            def priority(v):
                deg = 0
                for w in adj_map[v]:
                    if not visited[w] and w != current:
                        deg += 1
                return deg
                
            unvisited_neighbors.sort(key=priority)
            
            for next_node in unvisited_neighbors:
                # Pruning: last vertex must connect to start
                if len(path) + 1 == n_local:
                    if start_node not in adj_map[next_node]:
                        continue
                        
                visited[next_node] = True
                path.append(next_node)
                
                if backtrack():
                    return True
                
                path.pop()
                visited[next_node] = False
                
            return False
        
        if backtrack():
            return path[:]
        return None

    best_cycles = []
    random.seed(42)
    
    # Try multiple random restarts to ensure we find 3 disjoint cycles
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        current_adj = [set(nei) for nei in adj]
        cycles = []
        success = True
        
        for i in range(3):
            # Random start node
            start_node = random.randint(0, n - 1)
            
            cycle = find_hamiltonian_cycle(current_adj, start_node=start_node, deadline=deadline)
            
            if cycle is None:
                # Try a few other starts
                found = False
                for s in random.sample(range(n), min(5, n)):
                    if time.time() > deadline:
                        break
                    cycle = find_hamiltonian_cycle(current_adj, start_node=s, deadline=deadline)
                    if cycle is not None:
                        found = True
                        break
                if not found:
                    success = False
                    break
            
            cycles.append(cycle)
            
            # Remove edges
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                current_adj[u].discard(v)
                current_adj[v].discard(u)
                
            # Check viability
            viable = True
            for u in range(n):
                if len(current_adj[u]) < 2:
                    viable = False
                    break
            if not viable:
                success = False
                break
        
        if success and len(cycles) > len(best_cycles):
            best_cycles = cycles[:]
            if len(best_cycles) == 3:
                break
                
    # Fallback: Deterministic
    if len(best_cycles) < 3:
        current_adj = [set(nei) for nei in adj]
        cycles = []
        for i in range(3):
            cycle = find_hamiltonian_cycle(current_adj, start_node=0, deadline=deadline)
            if cycle is None:
                break
            cycles.append(cycle)
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                current_adj[u].discard(v)
                current_adj[v].discard(u)
        if len(cycles) > len(best_cycles):
            best_cycles = cycles[:]

    # Ensure types are correct
    result = [list(c) for c in best_cycles]
    
    return result
