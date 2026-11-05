import time
import random
import sys

def construct():
    n = 20
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_edges = 0
    best_adj = None
    best_edge_list = []
    
    def check_validity(adj):
        # Verify no K2,3
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    common = 0
                    for w in range(n):
                        if w != u and w != v and adj[u][w] and adj[v][w]:
                            common += 1
                            if common > 2:
                                return False
        return True

    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2

    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    def can_add_edge(adj, u, v):
        # Check if adding (u,v) creates K2,3
        # 1. u and v themselves
        common_uv = 0
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                common_uv += 1
                if common_uv > 2:
                    return False
        
        # 2. u and some w where v becomes a new common neighbor
        # This happens if v is in N(w) and |N(u) \cap N(w)| == 2
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                # v is neighbor of w
                # Count common neighbors of u and w currently
                common_uw = 0
                for x in range(n):
                    if x != u and x != w and adj[u][x] and adj[w][x]:
                        common_uw += 1
                        if common_uw > 2:
                            return False
                        if common_uw == 2:
                            # If we add (u,v), and v is in N(w), 
                            # is v in N(u)? Yes, we are adding it.
                            # Is v a common neighbor of u and w?
                            # Yes, if v is in N(w) and v in N(u).
                            # So common count becomes 3.
                            return False
                            
        # 3. v and some w where u becomes a new common neighbor
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u][w]:
                # u is neighbor of w
                common_vw = 0
                for x in range(n):
                    if x != v and x != w and adj[v][x] and adj[w][x]:
                        common_vw += 1
                        if common_vw > 2:
                            return False
                        if common_vw == 2:
                            return False
                            
        return True

    def try_add_all(adj):
        modified = True
        while modified:
            modified = False
            for u in range(n):
                for v in range(u+1, n):
                    if not adj[u][v]:
                        if can_add_edge(adj, u, v):
                            adj[u][v] = 1
                            adj[v][u] = 1
                            modified = True

    def local_search(adj, iterations=500):
        # Greedy fill
        try_add_all(adj)
        
        # Hill climbing: try to swap edges
        # Simple version: pick a random edge, remove it, try to fill again
        # Or: try to add any edge, if fails, remove a random edge and retry
        for _ in range(iterations):
            if time.time() > deadline:
                break
            # Try to add a random missing edge
            added = False
            for _ in range(20):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                if not adj[u][v]:
                    if can_add_edge(adj, u, v):
                        adj[u][v] = 1
                        adj[v][u] = 1
                        added = True
                        break
            
            if not added:
                # Remove a random existing edge to allow new ones
                existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
                if existing_edges:
                    u, v = random.choice(existing_edges)
                    adj[u][v] = 0
                    adj[v][u] = 0
                    try_add_all(adj)
                    
    # Initialize with multiple random starts
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization with varying density
        adj = [[0]*n for _ in range(n)]
        density = 0.1 + 0.3 * (r % 5) / 4  # Vary density from 0.1 to 0.4
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < density:
                    adj[u][v] = 1
                    adj[v][u] = 1
        
        local_search(adj)
        
        # Final systematic fill
        try_add_all(adj)
        
        # Verify
        if check_validity(adj):
            e_count = count_edges(adj)
            if e_count > best_edges:
                best_edges = e_count
                best_adj = [row[:] for row in adj]
                best_edge_list = get_edge_list(adj)
                
    # If no graph was found (should not happen), return seed
    if best_adj is None:
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
        
    return best_edge_list
