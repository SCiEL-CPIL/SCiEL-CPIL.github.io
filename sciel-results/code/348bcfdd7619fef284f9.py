import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145
    
    n = 16
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def has_common_neighbor(u, v):
        return (adj[u] & adj[v]) != 0
    
    def edge_count():
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2
    
    def get_edges():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges
    
    def is_3_colorable():
        order = sorted(range(n), key=lambda x: -bin(adj[x]).count('1'))
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = 0
            av = adj[v]
            for c in range(3):
                # Check if color c is available
                # Find all neighbors with color c
                conflict = 0
                for nb in range(n):
                    if (av >> nb) & 1 and colors[nb] == c:
                        conflict = 1
                        break
                if not conflict:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)
    
    def build_grotzsch():
        nonlocal adj
        adj = [0] * n
        for i in range(5):
            add_edge(i, (i + 1) % 5)
        for i in range(5):
            for nb in range(5):
                if adj[i] & (1 << nb):
                    add_edge(5 + i, nb)
        for i in range(5):
            add_edge(10, 5 + i)
    
    def find_independent_set(vertices_mask):
        """Find a large independent set from vertices in mask."""
        remaining = vertices_mask
        independent = 0
        while remaining:
            best_v = -1
            best_deg = 999
            for v in range(n):
                if (remaining >> v) & 1:
                    deg = bin(adj[v] & remaining).count('1')
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
            if best_v == -1:
                break
            independent |= (1 << best_v)
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
        return independent
    
    def add_vertex(new_v):
        """Add vertex new_v connected to a large independent set."""
        current_mask = (1 << new_v) - 1
        independent = find_independent_set(current_mask)
        adj[new_v] = independent
        for v in range(new_v):
            if independent & (1 << v):
                adj[v] |= (1 << new_v)
    
    def local_search():
        """Try to add more edges."""
        changed = True
        while changed:
            changed = False
            if time.time() > DEADLINE:
                return
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if not has_common_neighbor(u, v):
                            add_edge(u, v)
                            if not is_3_colorable():
                                changed = True
                            else:
                                remove_edge(u, v)
    
    best_edges = 0
    best_result = None
    
    # Try multiple random orderings for adding vertices
    for trial in range(20):
        if time.time() > DEADLINE:
            break
        build_grotzsch()
        
        # Add vertices 11-15 in random order
        verts = list(range(11, 16))
        random.shuffle(verts)
        for v in verts:
            if time.time() > DEADLINE:
                break
            add_vertex(v)
        
        # Local search
        local_search()
        
        # Check result
        if not is_3_colorable():
            ec = edge_count()
            if ec > best_edges:
                best_edges = ec
                best_result = get_edges()
        
        if time.time() > DEADLINE:
            break
    
    if best_result is None:
        # Fallback: Grötzsch + isolated
        build_grotzsch()
        best_result = get_edges()
    
    return best_result
