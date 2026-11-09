import time
from random import Random

def construct():
    start_time = time.time()
    deadline = 145.0

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        n = max(max(e) for e in edges) + 1
        adj = [set() for _ in range(n)]
        deg = [0] * n
        
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        paths = []
        remaining_edges = len(edges)
        
        while remaining_edges > 0:
            if time.time() > deadline:
                break
            
            # Find a vertex with degree > 0
            u = -1
            for i in range(n):
                if deg[i] > 0:
                    u = i
                    break
            
            if u == -1:
                break
            
            # Try to build a long simple path starting from u
            path = [u]
            visited = {u}
            current = u
            
            while deg[current] > 0:
                # Pick a neighbor that leads to the longest potential path
                # Heuristic: prefer neighbors with higher degree, but avoid cycles
                best_next = -1
                best_score = -1
                
                candidates = list(adj[current])
                # Sort by degree descending to prioritize "trunk" of the graph
                candidates.sort(key=lambda v: deg[v], reverse=True)
                
                for v in candidates:
                    if v in visited:
                        continue
                    # Simple lookahead: if we go to v, can we extend further?
                    # For now, just pick the first valid neighbor with high degree
                    if deg[v] > best_score:
                        best_score = deg[v]
                        best_next = v
                
                if best_next == -1:
                    break
                
                # Traverse edge
                adj[current].remove(best_next)
                adj[best_next].remove(current)
                deg[current] -= 1
                deg[best_next] -= 1
                remaining_edges -= 1
                
                path.append(best_next)
                visited.add(best_next)
                current = best_next
            
            # Only add path if it contains at least one edge (2 vertices)
            if len(path) >= 2:
                paths.append(path)
            else:
                # If no path could be formed (shouldn't happen if deg[u]>0), 
                # force a single edge to be consumed to make progress
                # This case implies u had neighbors but all were in visited? 
                # Or the loop broke immediately. 
                # If deg[u] > 0, there is at least one neighbor. 
                # If best_next was -1, it means all neighbors were in visited.
                # But visited only has u. So this shouldn't happen unless adj is inconsistent.
                # To be safe, if we didn't move, we just pick any neighbor and make a 2-node path.
                if deg[u] > 0:
                    v = next(iter(adj[u]))
                    adj[u].remove(v)
                    adj[v].remove(u)
                    deg[u] -= 1
                    deg[v] -= 1
                    remaining_edges -= 1
                    paths.append([u, v])

        return paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for spec in specs:
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
        if time.time() > deadline:
            break

    return results
