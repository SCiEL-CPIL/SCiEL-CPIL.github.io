import time
import itertools
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = [tuple(e) for e in P['edges']]
    base_set = set()
    for e in base_edges:
        base_set.add(e)
        base_set.add((e[1], e[0]))
    
    # Build adjacency for base
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    # All possible edges (not in base)
    all_possible = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in base_set:
                all_possible.append((u, v))
    
    def check_solution(added_edges):
        """Check if added_edges form a valid solution: triangle-free, connected, diameter <= 4"""
        if len(added_edges) == 0:
            return False
        # Build full adjacency
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check triangle-free
        for u in range(n):
            neighbors = list(adj[u])
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        return False
        
        # Check connected and diameter <= 4 via BFS from each node
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return False
                        q.append(v)
            if any(d == -1 for d in dist):
                return False
        return True
    
    def check_fast(added_edges_set):
        """Faster check using adjacency sets"""
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Triangle-free check
        for u in range(n):
            neighbors = adj[u]
            for x in neighbors:
                for y in neighbors:
                    if x < y and y in adj[x]:
                        return False
        
        # Diameter check
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return False
                        q.append(v)
            if any(d == -1 for d in dist):
                return False
        return True
    
    deadline = time.time() + 80  # 80s safety margin
    
    best_solution = None
    best_count = float('inf')
    
    def try_solution(added_edges):
        nonlocal best_solution, best_count
        if len(added_edges) >= best_count:
            return
        if check_solution(added_edges):
            best_solution = list(added_edges)
            best_count = len(added_edges)
    
    # Strategy 1: Hub-based initialization
    # Try different hub configurations
    
    # Single hub strategies
    for hub in range(n):
        if time.time() > deadline:
            break
        # Connect hub to all vertices except adjacent ones (to avoid triangles with path edges)
        # Actually, connecting hub to v is fine unless there's a triangle
        # Triangle forms if hub connected to v and v-1, and v-1 connected to hub (no) 
        # Triangle: hub-v, v-(v+1), (v+1)-hub => triangle if hub connected to both v and v+1
        # So hub can connect to any set of vertices as long as no two consecutive are both connected? No.
        # Triangle: (hub, v, w) where v-w is an edge. So if hub connects to v and w, and v-w is in base, triangle.
        # Base edges are (i, i+1). So hub cannot connect to both i and i+1 for any i.
        # So hub's connections must be an independent set in the path = no two consecutive.
        
        # Max independent set on path 0..23: 12 vertices (e.g., 0,2,4,...,22)
        # But we need diameter <= 4, so we need enough coverage.
        
        # Let's try: hub connects to an independent set, plus maybe some additional structure
        # For now, try all maximal independent sets is too many. Try specific patterns.
        
        # Pattern: connect hub to every other vertex starting from 0
        conn1 = [i for i in range(n) if i != hub and i % 2 == 0]
        # Remove if consecutive to hub's position in the set... actually just check triangle-free
        added = [(hub, v) for v in conn1 if v != hub]
        try_solution(added)
        
        conn2 = [i for i in range(n) if i != hub and i % 2 == 1]
        added = [(hub, v) for v in conn2 if v != hub]
        try_solution(added)
    
    # Strategy 2: Two hubs
    for h1 in range(n):
        if time.time() > deadline:
            break
        for h2 in range(h1+1, n):
            if time.time() > deadline:
                break
            # Connect h1 to some, h2 to others, and h1-h2
            # h1 and h2 should not be adjacent (triangle with path)
            if abs(h1 - h2) == 1:
                continue
            # h1 connects to even indices, h2 to odd indices (roughly)
            # More carefully: each hub connects to an independent set
            added = []
            # h1-h2 edge
            if (h1, h2) not in base_set:
                added.append((h1, h2))
            # h1 connects to vertices where no triangle
            for v in range(n):
                if v == h1 or v == h2:
                    continue
                if (h1, v) in base_set:
                    continue
                # Check if adding (h1,v) creates triangle
                # Triangle if there exists w such that (h1,w) and (v,w) are edges
                # For now, just add and check later
                added.append((h1, v))
            # This is getting complex. Let's use a simpler approach.
    
    # Strategy 3: Greedy construction
    # Start with no added edges, greedily add edges that most reduce diameter
    if time.time() < deadline:
        current_adj = [set() for _ in range(n)]
        for u, v in base_edges:
            current_adj[u].add(v)
            current_adj[v].add(u)
        
        def get_diameter(adj):
            max_d = 0
            for src in range(n):
                dist = [-1] * n
                dist[src] = 0
                q = deque([src])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            max_d = max(max_d, dist[v])
                            q.append(v)
                if any(d == -1 for d in dist):
                    return float('inf')
            return max_d
        
        def is_triangle_free(adj):
            for u in range(n):
                neighbors = list(adj[u])
                for i in range(len(neighbors)):
                    for j in range(i+1, len(neighbors)):
                        if neighbors[j] in adj[neighbors[i]]:
                            return False
            return True
        
        current_edges = []
        current_diam = get_diameter(current_adj)
        
        # Greedily add edges to reduce diameter
        while current_diam > 4 and time.time() < deadline:
            best_edge = None
            best_new_diam = current_diam
            for u, v in all_possible:
                if (u, v) in [(e[0], e[1]) for e in current_edges] or (v, u) in [(e[0], e[1]) for e in current_edges]:
                    continue
                # Check if adding (u,v) creates a triangle
                # Triangle if exists w: (u,w) and (v,w) in current_adj
                creates_tri = False
                for w in current_adj[u]:
                    if w in current_adj[v]:
                        creates_tri = True
                        break
                if creates_tri:
                    continue
                # Try adding
                current_adj[u].add(v)
                current_adj[v].add(u)
                new_diam = get_diameter(current_adj)
                current_adj[u].remove(v)
                current_adj[v].remove(u)
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = (u, v)
            if best_edge is None:
                break
            u, v = best_edge
            current_adj[u].add(v)
            current_adj[v].add(u)
            current_edges.append((u, v))
            current_diam = best_new_diam
        
        if current_diam <= 4 and is_triangle_free(current_adj):
            try_solution(current_edges)
    
    # Strategy 4: Local search from seed solution
    if time.time() < deadline:
        # Start with seed solution
        seed_edges = [(u, v) for u in range(n) for v in range(u+1, n) if (u-v) % 2 == 1 and (u, v) not in base_set]
        if check_solution(seed_edges):
            try_solution(seed_edges)
        
        # Local search: try removing edges
        current = list(seed_edges)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                candidate = current[:i] + current[i+1:]
                if check_solution(candidate):
                    current = candidate
                    improved = True
                    break
        if check_solution(current):
            try_solution(current)
    
    # Strategy 5: More aggressive local search with random restarts
    import random
    random.seed(42)
    
    if time.time() < deadline:
        # Try various hub-based initializations
        for hub in range(n):
            if time.time() > deadline:
                break
            # Independent set around hub
            # Connect hub to non-consecutive vertices
            conn = []
            for v in range(n):
                if v == hub:
                    continue
                if (hub, v) in base_set or (v, hub) in base_set:
                    continue
                # Check no triangle: for all w in adj[hub] and adj[v], they shouldn't be adjacent
                # Simplified: just add and check
                conn.append(v)
            
            # Remove consecutive connections to avoid triangles
            conn.sort()
            filtered = []
            for v in conn:
                if not filtered or abs(v - filtered[-1]) > 1:
                    filtered.append(v)
            
            added = [(hub, v) for v in filtered]
            if check_solution(added):
                try_solution(added)
                # Local search
                current = list(added)
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    for i in range(len(current)):
                        candidate = current[:i] + current[i+1:]
                        if check_solution(candidate):
                            current = candidate
                            improved = True
                            break
                if check_solution(current):
                    try_solution(current)
    
    # Strategy 6: Systematic search over small edge sets
    if time.time() < deadline:
        # Try all combinations of k edges for small k
        # This is expensive, so limit
        for k in range(5, min(15, best_count if best_count < float('inf') else 15)):
            if time.time() > deadline:
                break
            # Sample random combinations
            for _ in range(1000):
                if time.time() > deadline:
                    break
                sample = random.sample(all_possible, k)
                if check_solution(sample):
                    try_solution(sample)
    
    # Strategy 7: Two-hub construction
    if time.time() < deadline:
        for h1 in range(0, n, 2):  # even hubs
            if time.time() > deadline:
                break
            for h2 in range(h1+2, n, 2):  # even hubs, not adjacent
                if time.time() > deadline:
                    break
                if abs(h1 - h2) <= 1:
                    continue
                added = []
                # h1-h2 edge
                added.append((h1, h2))
                # h1 connects to vertices in [0, h1+4] range, non-consecutive
                for v in range(0, min(h1+5, n)):
                    if v == h1 or v == h2:
                        continue
                    if (h1, v) in base_set or (v, h1) in base_set:
                        continue
                    # Check triangle with existing
                    tri = False
                    for w in [x for x in range(n) if (h1, x) in [(e[0],e[1]) for e in added] or (x, h1) in [(e[0],e[1]) for e in added]]:
                        if (v, w) in base_set or (w, v) in base_set:
                            tri = True
                            break
                    if not tri:
                        added.append((h1, v))
                # h2 connects to vertices in [h2-4, n-1] range
                for v in range(max(h2-5, 0), n):
                    if v == h1 or v == h2:
                        continue
                    if (h2, v) in base_set or (v, h2) in base_set:
                        continue
                    tri = False
                    for w in [x for x in range(n) if (h2, x) in [(e[0],e[1]) for e in added] or (x, h2) in [(e[0],e[1]) for e in added]]:
                        if (v, w) in base_set or (w, v) in base_set:
                            tri = True
                            break
                    if not tri:
                        added.append((h2, v))
                
                # Remove duplicates
                seen = set()
                unique_added = []
                for e in added:
                    key = (min(e[0], e[1]), max(e[0], e[1]))
                    if key not in seen and key not in base_set:
                        seen.add(key)
                        unique_added.append(e)
                
                if check_solution(unique_added):
                    try_solution(unique_added)
                    # Local search
                    current = list(unique_added)
                    improved = True
                    while improved and time.time() < deadline:
                        improved = False
                        for i in range(len(current)):
                            candidate = current[:i] + current[i+1:]
                            if check_solution(candidate):
                                current = candidate
                                improved = True
                                break
                    if check_solution(current):
                        try_solution(current)
    
    if best_solution is not None:
        return [[int(u), int(v)] for u, v in best_solution]
    
    # Fallback: return seed
    return [[u, v] for u in range(n) for v in range(u+1, n) if (u-v) % 2 == 1 and (u, v) not in base_set]

# EVOLVE-BLOCK-END
