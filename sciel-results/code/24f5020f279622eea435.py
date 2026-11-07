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
        edge_set.add((min(u,v), max(u,v)))
        adj[u].add(v)
        adj[v].add(u)
    
    def edge_key(u, v):
        return (min(u,v), max(u,v))
    
    def is_connected(available_edges):
        if not available_edges:
            return n == 1
        adj_temp = {i: set() for i in range(n)}
        for u, v in available_edges:
            adj_temp[u].add(v)
            adj_temp[v].add(u)
        visited = set()
        stack = [0]
        visited.add(0)
        while stack:
            node = stack.pop()
            for nb in adj_temp[node]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        return len(visited) == n

    def min_degree(available_edges):
        degrees = {i: 0 for i in range(n)}
        for u, v in available_edges:
            degrees[u] += 1
            degrees[v] += 1
        for i in range(n):
            if degrees[i] < 2:
                return False
        return True

    def find_hamiltonian_cycle(available_edges, start=0, time_limit=5.0):
        t0 = time.time()
        path = [start]
        visited = {start}
        
        degs = {i: 0 for i in range(n)}
        for u, v in available_edges:
            degs[u] += 1
            degs[v] += 1
        
        # Precompute adjacency restricted to available edges
        avail_adj = {i: set() for i in range(n)}
        for u, v in available_edges:
            avail_adj[u].add(v)
            avail_adj[v].add(u)
            
        def backtrack():
            if time.time() - t0 > time_limit:
                return "timeout"
            
            if len(path) == n:
                last = path[-1]
                if edge_key(last, start) in available_edges:
                    return list(path)
                return None
            
            last = path[-1]
            candidates = []
            for nb in avail_adj[last]:
                if nb not in visited:
                    candidates.append(nb)
            
            if not candidates:
                return None

            # Warnsdorff heuristic: prefer neighbors with fewer available edges (more constrained)
            # But also consider degree in remaining graph
            def heuristic_score(nb):
                # Number of unvisited neighbors of nb in available graph
                unvisited_neighbors = 0
                for nn in avail_adj[nb]:
                    if nn not in visited:
                        unvisited_neighbors += 1
                return (unvisited_neighbors, degs[nb])
            
            candidates.sort(key=heuristic_score)

            for nb in candidates:
                old_deg_last = degs[last]
                old_deg_nb = degs[nb]
                
                degs[last] -= 1
                degs[nb] -= 1
                ek = edge_key(last, nb)
                available_edges.discard(ek)
                avail_adj[last].discard(nb)
                avail_adj[nb].discard(last)
                
                path.append(nb)
                visited.add(nb)
                
                # Pruning 1: Check unvisited vertices have degree >= 2
                valid = True
                for v in range(n):
                    if v not in visited:
                        if degs[v] < 2:
                            valid = False
                            break
                if not valid:
                    path.pop()
                    visited.remove(nb)
                    available_edges.add(ek)
                    avail_adj[last].add(nb)
                    avail_adj[nb].add(last)
                    degs[last] = old_deg_last
                    degs[nb] = old_deg_nb
                    continue
                
                # Pruning 2: Check that the last vertex in path (if not at n-1) 
                # still has a way to reach start (connectedness of remaining + path)
                # Light check: ensure nb can reach at least one unvisited vertex or start
                if len(path) < n:
                    has_exit = False
                    for nn in avail_adj[nb]:
                        if nn not in visited:
                            has_exit = True
                            break
                    if not has_exit:
                        path.pop()
                        visited.remove(nb)
                        available_edges.add(ek)
                        avail_adj[last].add(nb)
                        avail_adj[nb].add(last)
                        degs[last] = old_deg_last
                        degs[nb] = old_deg_nb
                        continue
                
                result = backtrack()
                if result is not None and result != "timeout":
                    return result
                
                # Backtrack
                path.pop()
                visited.remove(nb)
                available_edges.add(ek)
                avail_adj[last].add(nb)
                avail_adj[nb].add(last)
                degs[last] = old_deg_last
                degs[nb] = old_deg_nb
                
            return None
        
        result = backtrack()
        if result == "timeout":
            return None
        return result

    cycles = []
    available = set(edge_set)
    
    max_cycles = 3
    for cycle_idx in range(max_cycles):
        if time.time() > deadline:
            break
        
        found = None
        # Try multiple start vertices with increasing time limits
        starts = [0] + list(range(1, n))
        for start in starts:
            if time.time() > deadline:
                break
            tl = 8.0 if cycle_idx == 0 else 12.0
            found = find_hamiltonian_cycle(available, start=start, time_limit=tl)
            if found is not None:
                break
        
        if found is None:
            break
            
        # Remove used edges
        for i in range(len(found)):
            u = found[i]
            v = found[(i+1) % len(found)]
            available.discard(edge_key(u, v))
        cycles.append(found)
        
        if not min_degree(available) or not is_connected(available):
            break

    if cycles:
        return [list(c) for c in cycles]
    else:
        random.seed(42)
        for _ in range(50):
            found = find_hamiltonian_cycle(set(edge_set), start=random.randint(0, n-1), time_limit=2.0)
            if found:
                return [found]
        return [list(range(n))]
# EVOLVE-BLOCK-END
