import time
import random

def construct():
    n = 16
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Build initial Clebsch graph using bitsets for adjacency
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            xor = u ^ v
            dist = bin(xor).count('1')
            if dist == 1 or dist == 4:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    def count_edges():
        total = 0
        for u in range(n):
            total += bin(adj[u]).count('1')
        return total // 2
    
    def is_triangle_free():
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    if adj[u] & adj[v]:
                        return False
        return True
    
    def is_3_colorable():
        # Backtracking 3-coloring
        colors = [-1] * n
        def backtrack(idx):
            if idx == n:
                return True
            for c in range(3):
                ok = True
                for v in range(idx):
                    if (adj[idx] >> v) & 1 and colors[v] == c:
                        ok = False
                        break
                if ok:
                    colors[idx] = c
                    if backtrack(idx + 1):
                        return True
                    colors[idx] = -1
            return False
        return backtrack(0)
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    # Greedy phase: try to add every possible missing edge
    # that maintains triangle-freeness and 4-chromaticity
    for u in range(n):
        for v in range(u + 1, n):
            if time.time() > deadline:
                break
            if (adj[u] >> v) & 1:
                continue  # already present
            # Check triangle-freeness: u and v must have no common neighbor
            if adj[u] & adj[v]:
                continue
            # Tentatively add
            add_edge(u, v)
            if not is_3_colorable():
                # Keep it
                pass
            else:
                # Remove it
                remove_edge(u, v)
    
    # Local search: try edge swaps (remove one, add another)
    best_edges = count_edges()
    best_adj = adj[:]
    
    while time.time() < deadline - 5:
        improved = False
        # Try removing each edge and adding a different one
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline - 5:
                    break
                if not ((adj[u] >> v) & 1):
                    continue  # not an edge, skip
                # Try removing this edge
                remove_edge(u, v)
                # Try adding each missing edge
                for a in range(n):
                    for b in range(a + 1, n):
                        if time.time() > deadline - 5:
                            break
                        if (adj[a] >> b) & 1:
                            continue  # already present
                        # Check triangle-freeness
                        if adj[a] & adj[b]:
                            continue
                        add_edge(a, b)
                        if not is_3_colorable():
                            # Good swap
                            if count_edges() > best_edges:
                                best_edges = count_edges()
                                best_adj = adj[:]
                                improved = True
                        else:
                            remove_edge(a, b)
                # Restore removed edge
                add_edge(u, v)
                if improved:
                    break
            if improved:
                break
    
    # Use best found
    if best_edges > count_edges():
        adj = best_adj
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
