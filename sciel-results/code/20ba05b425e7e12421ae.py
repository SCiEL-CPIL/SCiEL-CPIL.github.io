import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 138.0  # 12s margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_cycle(adj, n, rng, min_len=3, max_search_nodes=5000):
        """Find a cycle in the graph. Returns list of vertices or None."""
        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        if not candidates:
            return None
        rng.shuffle(candidates)
        
        best_cycle = None
        nodes_explored = 0
        
        for start in candidates:
            if nodes_explored > max_search_nodes:
                break
            if best_cycle and len(best_cycle) >= 8:
                break
            
            # Iterative DFS
            visited = set()
            visited.add(start)
            path = [start]
            # Use a stack of (node, iterator over neighbors)
            neighbors_iter = {start: iter(sorted(adj[start], key=lambda x: -len(adj[x])))}
            stack = [start]
            
            while stack:
                node = stack[-1]
                if node not in neighbors_iter:
                    neighbors_iter[node] = iter(sorted(adj[node], key=lambda x: -len(adj[x])))
                
                found_next = False
                for nb in neighbors_iter[node]:
                    nodes_explored += 1
                    if nodes_explored > max_search_nodes:
                        return best_cycle
                    
                    if nb == start and len(path) >= min_len:
                        cycle = list(path)
                        if not best_cycle or len(cycle) > len(best_cycle):
                            best_cycle = cycle
                        found_next = True
                        break
                    elif nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        stack.append(nb)
                        neighbors_iter[nb] = iter(sorted(adj[nb], key=lambda x: -len(adj[x])))
                        found_next = True
                        break
                
                if not found_next:
                    stack.pop()
                    path.pop()
                    visited.discard(node)
                    del neighbors_iter[node]
            
            if best_cycle and len(best_cycle) >= 6:
                break
        
        return best_cycle
    
    def decompose(edges, n, rng):
        remaining = set(tuple(sorted(e)) for e in edges)
        pieces = []
        
        for _ in range(200):  # max cycles to find
            if time.time() - start_time > DEADLINE:
                break
            if not remaining:
                break
            
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)
            
            cycle = find_cycle(adj, n, rng)
            
            if not cycle:
                break
            
            # Extract and remove cycle edges
            cycle_edges = set()
            valid = True
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i+1) % len(cycle)]
                e = tuple(sorted((u, v)))
                if e not in remaining:
                    valid = False
                    break
                cycle_edges.add(e)
            
            if not valid:
                break
            
            pieces.append(cycle)
            remaining -= cycle_edges
        
        for u, v in remaining:
            pieces.append([u, v])
        
        return pieces
    
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for n, p, seed in specs:
        if time.time() - start_time > DEADLINE:
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(n, p, seed)
        best_pieces = None
        best_count = len(edges)
        
        # Time budget per graph
        graph_deadline = min(start_time + 40.0, DEADLINE)
        
        for attempt in range(50):
            if time.time() > graph_deadline:
                break
            rng = random.Random(seed * 1000 + attempt * 17 + 3)
            pieces = decompose(edges, n, rng)
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = pieces
                # If we've done very well, stop early
                if count <= len(edges) * 0.3:
                    break
        
        if best_pieces is None:
            best_pieces = [list(e) for e in edges]
        
        results.append(best_pieces)
    
    return results
