import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    n = 20
    num_colors = 20
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    m = len(all_edges)
    
    best_count = 0
    best_edges = []
    
    def is_valid_4cycle(adj, u, a, b, v):
        """Check 4-cycle u-a-b-v: edges (u,a),(a,b),(b,v),(v,u) all exist and rainbow."""
        c1 = adj[u][a]
        c2 = adj[a][b]
        c3 = adj[b][v]
        c4 = adj[v][u]
        if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
            return False
        return len({c1, c2, c3, c4}) == 4
    
    def can_add_edge(adj, neighbors, u, v):
        """Check if edge (u,v) can be added with some color. Returns (feasible, forbidden_colors)."""
        forbidden = set()
        feasible = True
        
        for a in neighbors[u]:
            if a == v:
                continue
            ca = adj[u][a]
            for b in neighbors[v]:
                if b == u or b == a:
                    continue
                if adj[a][b] != -1:
                    cb = adj[a][b]
                    cv = adj[b][v]
                    # 4-cycle: u-a-b-v, edges: (u,a)=ca, (a,b)=cb, (b,v)=cv, (v,u)=new
                    # For rainbow: ca, cb, cv, new must all be distinct
                    # If any two of ca, cb, cv are equal, infeasible
                    if ca == cb or ca == cv or cb == cv:
                        return False, set()
                    forbidden.add(ca)
                    forbidden.add(cb)
                    forbidden.add(cv)
        
        return True, forbidden
    
    def greedy_build(seed, init_edges=None):
        rng = random.Random(seed)
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        if init_edges:
            for u, v, c in init_edges:
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
            existing = set((min(u,v), max(u,v)) for u, v, c in edges)
            remaining = [e for e in all_edges if e not in existing]
        else:
            remaining = all_edges[:]
        
        rng.shuffle(remaining)
        
        for (u, v) in remaining:
            feasible, forbidden = can_add_edge(adj, neighbors, u, v)
            if feasible:
                if forbidden:
                    candidates = [c for c in range(num_colors) if c not in forbidden]
                    if candidates:
                        c = rng.choice(candidates)
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
                else:
                    c = rng.randrange(num_colors)
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])
        
        return edges
    
    def validate_edges(edges):
        """Validate that all 4-cycles are rainbow. Returns True if valid."""
        adj = [[-1] * n for _ in range(n)]
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
        
        # Check all 4-cycles
        for u in range(n):
            for a in range(n):
                if a == u or adj[u][a] == -1:
                    continue
                for b in range(n):
                    if b == u or b == a or adj[a][b] == -1:
                        continue
                    for v in range(n):
                        if v == u or v == a or v == b or adj[b][v] == -1 or adj[v][u] == -1:
                            continue
                        c1 = adj[u][a]
                        c2 = adj[a][b]
                        c3 = adj[b][v]
                        c4 = adj[v][u]
                        if len({c1, c2, c3, c4}) < 4:
                            return False
        return True
    
    def greedy_build_bipartite_seed(seed):
        """Greedy build with bipartite initial structure for better density."""
        rng = random.Random(seed)
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        # Start with a bipartite graph between {0..9} and {10..19}
        # Use a structured coloring to avoid conflicts
        for u in range(10):
            for v in range(10, 20):
                c = (u + v) % num_colors
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
        
        existing = set((min(u,v), max(u,v)) for u, v, c in edges)
        remaining = [e for e in all_edges if e not in existing]
        rng.shuffle(remaining)
        
        for (u, v) in remaining:
            feasible, forbidden = can_add_edge(adj, neighbors, u, v)
            if feasible:
                if forbidden:
                    candidates = [c for c in range(num_colors) if c not in forbidden]
                    if candidates:
                        c = rng.choice(candidates)
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
                else:
                    c = rng.randrange(num_colors)
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])
        
        return edges
    
    # Phase 1: Greedy random restarts (plain)
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        edges = greedy_build(i)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    # Phase 1b: Greedy with bipartite seed
    for i in range(50):
        if time.time() > deadline:
            break
        edges = greedy_build_bipartite_seed(i)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    # Phase 2: Local search — remove k edges, refill greedily
    current_best = best_edges[:]
    
    while time.time() < deadline:
        k = random.randint(1, 6)
        if len(current_best) <= k:
            break
        
        to_remove = random.sample(current_best, k)
        rem_set = set((min(u,v), max(u,v)) for u, v, c in to_remove)
        base = [e for e in current_best if (min(e[0], e[1]), max(e[0], e[1])) not in rem_set]
        
        seed = random.randint(0, 10**9)
        new_edges = greedy_build(seed, base)
        
        if len(new_edges) > best_count:
            best_count = len(new_edges)
            best_edges = new_edges
            current_best = new_edges[:]
    
    # Phase 3: Final validation and cleanup
    if best_edges:
        if not validate_edges(best_edges):
            # Fallback: find a valid subset
            adj = [[-1] * n for _ in range(n)]
            valid_edges = []
            for u, v, c in best_edges:
                adj[u][v] = c
                adj[v][u] = c
                # Check if adding this edge creates any non-rainbow 4-cycle
                ok = True
                for a in range(n):
                    if a == u or a == v or adj[u][a] == -1:
                        continue
                    for b in range(n):
                        if b == u or b == v or b == a or adj[a][b] == -1 or adj[b][v] == -1:
                            continue
                        c1 = adj[u][a]
                        c2 = adj[a][b]
                        c3 = adj[b][v]
                        c4 = c
                        if len({c1, c2, c3, c4}) < 4:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    valid_edges.append([u, v, c])
                else:
                    adj[u][v] = -1
                    adj[v][u] = -1
            if valid_edges:
                best_edges = valid_edges
                best_count = len(best_edges)
            else:
                # Last resort: star graph
                best_edges = [[0, v, v % 20] for v in range(1, 20)]
                best_count = len(best_edges)
    else:
        best_edges = [[0, v, v % 20] for v in range(1, 20)]
        best_count = len(best_edges)
    
    return best_edges
