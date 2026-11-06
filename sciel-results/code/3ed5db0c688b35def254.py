import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140
    n = 20
    best_edges = [[0, v, 0] for v in range(1, 20)]  # star fallback
    best_count = 19
    
    all_edges = list(combinations(range(n), 2))
    
    def is_valid(adj, colors, u, v, c):
        # Check if adding edge (u,v) with color c creates a non-rainbow 4-cycle
        # 4-cycle containing (u,v): u-a-b-v-u
        # Edges: (u,v), (u,a), (a,b), (b,v)
        
        for a in adj[u]:
            if a == v:
                continue
            ca = colors.get((u, a))
            if ca is None:
                ca = colors.get((a, u))
            if ca is None:
                continue
            
            for b in adj[a]:
                if b == u or b == v or b == a:
                    continue
                if v not in adj[b]:
                    continue
                
                cab = colors.get((a, b))
                if cab is None:
                    cab = colors.get((b, a))
                cbv = colors.get((b, v))
                if cbv is None:
                    cbv = colors.get((v, b))
                
                if cab is None or cbv is None:
                    continue
                
                if len({c, ca, cab, cbv}) != 4:
                    return False
        
        return True

    def count_edges(adj):
        return sum(len(s) for s in adj.values()) // 2

    def local_search(adj, colors, current_edges, max_time):
        """Try edge swaps to improve edge count."""
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try removing each edge and see if we can add more edges
            for idx in range(len(current_edges) - 1, -1, -1):
                if time.time() > max_time:
                    break
                u, v, c = current_edges[idx]
                # Remove edge
                adj[u].discard(v)
                adj[v].discard(u)
                del colors[(u, v)]
                if (v, u) in colors:
                    del colors[(v, u)]
                current_edges.pop(idx)
                
                # Try to add back edges greedily
                added = 0
                for u2, v2 in all_edges:
                    if time.time() > max_time:
                        break
                    if v2 in adj[u2]:
                        continue
                    # Try each color
                    for c2 in range(n):
                        if is_valid(adj, colors, u2, v2, c2):
                            adj[u2].add(v2)
                            adj[v2].add(u2)
                            colors[(u2, v2)] = c2
                            colors[(v2, u2)] = c2
                            current_edges.append([u2, v2, c2])
                            added += 1
                            break
                
                if added <= 1:
                    # No improvement, restore original state
                    adj[u].add(v)
                    adj[v].add(u)
                    colors[(u, v)] = c
                    colors[(v, u)] = c
                    current_edges.append([u, v, c])
                else:
                    improved = True
        
        return current_edges

    for iteration in range(500):
        if time.time() > deadline:
            break
        
        random.shuffle(all_edges)
        adj = {i: set() for i in range(n)}
        colors = {}
        current_edges = []
        
        for u, v in all_edges:
            # Try to find a valid color
            found = False
            color_order = list(range(n))
            random.shuffle(color_order)
            
            for c in color_order:
                if is_valid(adj, colors, u, v, c):
                    adj[u].add(v)
                    adj[v].add(u)
                    colors[(u, v)] = c
                    colors[(v, u)] = c
                    current_edges.append([u, v, c])
                    found = True
                    break
        
        # Local search to escape local optima
        if len(current_edges) > best_count:
            max_local_time = min(deadline - 5, time.time() + 5)
            current_edges = local_search(adj, colors, current_edges, max_local_time)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
        
        # Early termination if we've found a very dense graph
        if best_count >= 190:
            break

    return best_edges
