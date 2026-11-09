import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n = 16
    
    # Helper: check if graph is triangle-free
    def is_triangle_free(adj):
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w == u:
                        continue
                    if w in adj[u]:
                        # Check if (u,v,w) forms a triangle: u-v, v-w, w-u all exist
                        if adj[u][w]:
                            return False
        return True
    
    # Better: use bitmasks for adjacency
    def is_triangle_free_mask(adj_mask):
        for u in range(n):
            for v in range(u+1, n):
                if adj_mask[u] & (1 << v):
                    # Check common neighbors
                    common = adj_mask[u] & adj_mask[v] & ~(1 << u) & ~(1 << v)
                    if common:
                        return False
        return True
    
    def is_3_colorable(adj_mask):
        # Backtracking 3-coloring
        color = [-1] * n
        # Try to color vertices in order of decreasing degree
        deg = [bin(adj_mask[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -deg[x])
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Get colors used by neighbors
            used = set()
            for u in range(n):
                if adj_mask[v] & (1 << u) and color[u] >= 0:
                    used.add(color[u])
            for c in range(3):
                if c not in used:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    # Start with Grötzsch graph on vertices 0-10, extend to 16
    # Grötzsch: 5-cycle + Mycielski
    # Vertices 0-4: cycle, 5-9: Mycielski copies, 10: apex
    
    # Build adjacency masks
    adj_mask = [0] * n
    
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Grötzsch graph edges
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, v in cycle:
        add_edge(u, v)
    for u, w in cycle:
        add_edge(u, 5 + w)
        add_edge(w, 5 + u)
    for u in range(5):
        add_edge(5 + u, 10)
    
    # Vertices 11-15 are isolated for now
    # Now try to add edges involving vertices 11-15 and between all vertices
    
    # Get all possible edges
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not (adj_mask[i] & (1 << j)):
                all_edges.append((i, j))
    
    random.shuffle(all_edges)
    
    best_edge_count = sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
    best_adj = list(adj_mask)
    
    # Local search: try to add edges one by one
    # For each edge, check if adding it keeps graph triangle-free and non-3-colorable
    def can_add_edge(adj_mask, u, v):
        # Check triangle-free: no common neighbor of u and v
        common = adj_mask[u] & adj_mask[v] & ~(1 << u) & ~(1 << v)
        if common:
            return False
        # Try adding the edge and check 3-colorability
        new_adj = list(adj_mask)
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        if is_3_colorable(new_adj):
            return False
        return True
    
    # Greedy: try to add edges
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i, j in all_edges:
            if time.time() > deadline - 5:
                break
            if adj_mask[i] & (1 << j):
                continue
            if can_add_edge(adj_mask, i, j):
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                changed = True
                current_edges = sum(bin(adj_mask[k]).count('1') for k in range(n)) // 2
                if current_edges > best_edge_count:
                    best_edge_count = current_edges
                    best_adj = list(adj_mask)
    
    # Also try removing edges and re-adding to escape local optima
    # Simulated annealing style
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        
        # Randomly perturb: remove a random edge, then try to add back
        current_edges = sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
        if current_edges >= best_edge_count:
            # Try to improve
            pass
        
        # Remove a random existing edge
        existing = []
        for i in range(n):
            for j in range(i+1, n):
                if adj_mask[i] & (1 << j):
                    existing.append((i, j))
        
        if not existing:
            continue
        
        ui, vi = random.choice(existing)
        adj_mask[ui] &= ~(1 << vi)
        adj_mask[vi] &= ~(1 << ui)
        
        # Now try to greedily add edges again
        for i, j in all_edges:
            if time.time() > deadline - 10:
                break
            if adj_mask[i] & (1 << j):
                continue
            if can_add_edge(adj_mask, i, j):
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
        
        current_edges = sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
        if current_edges > best_edge_count:
            best_edge_count = current_edges
            best_adj = list(adj_mask)
    
    # Return best found
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    
    return result
