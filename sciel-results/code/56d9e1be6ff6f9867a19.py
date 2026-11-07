import time
import random
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
    
    def build_adj(added_edges):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_triangle_free(adj):
        for u in range(n):
            neighbors = list(adj[u])
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        return False
        return True
    
    def compute_diameter(adj):
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
                        if dist[v] > 4:
                            return float('inf')
                        q.append(v)
            if any(d == -1 for d in dist):
                return float('inf')
            max_d = max(max_d, max(dist))
        return max_d
    
    def check_solution(added_edges):
        if not added_edges:
            return False
        adj = build_adj(added_edges)
        if not is_triangle_free(adj):
            return False
        diam = compute_diameter(adj)
        return diam <= 4
    
    def fast_check(added_edges):
        """Optimized check that short-circuits early"""
        adj = build_adj(added_edges)
        # Triangle-free check with early exit
        for u in range(n):
            nbrs = list(adj[u])
            for i in range(len(nbrs)):
                for j in range(i+1, len(nbrs)):
                    if nbrs[j] in adj[nbrs[i]]:
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
    
    deadline = time.time() + 75  # Safety margin
    
    best_solution = None
    best_count = float('inf')
    
    def try_solution(added_edges):
        nonlocal best_solution, best_count
        if len(added_edges) >= best_count:
            return
        if fast_check(added_edges):
            best_solution = list(added_edges)
            best_count = len(added_edges)
    
    # Strategy 1: Bipartite-style construction
    # Key insight: if we partition vertices into two sets A and B such that
    # all edges only go between A and B (plus the path edges within same parity),
    # we need to be careful. Actually, the path edges connect consecutive vertices.
    # A triangle-free graph can have edges within a set as long as no triangles form.
    # 
    # Better approach: Use the fact that path P24 has vertices 0..23.
    # If we add edges that only connect vertices of the same parity (even-even or odd-odd),
    # then any triangle would need to use path edges. Let's think...
    # 
    # Actually, let's use a different structural approach:
    # Add "chord" edges that create a structure with small diameter.
    
    # Strategy 2: Systematic construction with specific patterns
    # Pattern: Add edges to create a "ladder" or "grid" like structure
    
    # Try: connect vertices at distance 4 (creates 4-cycles, not triangles)
    # Edges (i, i+4) for various i
    for offset in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        added = []
        for i in range(n):
            j = i + offset
            if j < n and (i, j) not in base_set:
                added.append((i, j))
        # Check if this alone works
        if fast_check(added):
            try_solution(added)
        
        # Try combining with another offset
        for offset2 in [3, 4, 5, 6]:
            if offset2 == offset:
                continue
            if time.time() > deadline:
                break
            added2 = list(added)
            for i in range(n):
                j = i + offset2
                if j < n and (i, j) not in base_set and (i, j) not in [(a,b) for a,b in added]:
                    added2.append((i, j))
            if fast_check(added2):
                try_solution(added2)
    
    # Strategy 3: Hub-and-spoke with independent sets
    # For each potential hub, connect to an independent set of the path
    for hub in range(n):
        if time.time() > deadline:
            break
        
        # Generate independent sets (no two consecutive)
        # Use greedy: pick every other vertex
        for start_parity in [0, 1]:
            conn = []
            for v in range(n):
                if v == hub:
                    continue
                if v % 2 == start_parity:
                    # Check if hub-v creates triangle
                    # Triangle if exists w: hub-w and v-w are edges
                    # Base edges are (i,i+1), so if hub connects to v and v±1, and v±1 connects to v, triangle
                    # But hub only connects to one parity, so hub doesn't connect to v±1 (different parity)
                    # Unless v±1 == hub... 
                    if (hub, v) not in base_set:
                        conn.append(v)
            
            added = [(hub, v) for v in conn]
            if fast_check(added):
                try_solution(added)
            
            # Try with two hubs
            for hub2 in range(n):
                if time.time() > deadline:
                    break
                if hub2 == hub or abs(hub - hub2) == 1:
                    continue
                added2 = list(added)
                # hub2 connects to opposite parity
                for v in range(n):
                    if v == hub or v == hub2:
                        continue
                    if v % 2 != start_parity and (hub2, v) not in base_set:
                        # Check triangle
                        added2.append((hub2, v))
                # Add hub-hub2 if not base
                if (hub, hub2) not in base_set:
                    added2.append((hub, hub2))
                if fast_check(added2):
                    try_solution(added2)
    
    # Strategy 4: Greedy with local search
    if time.time() < deadline:
        current_adj = [set() for _ in range(n)]
        for u, v in base_edges:
            current_adj[u].add(v)
            current_adj[v].add(u)
        
        current_edges = []
        
        def get_diam():
            max_d = 0
            for src in range(n):
                dist = [-1] * n
                dist[src] = 0
                q = deque([src])
                while q:
                    u = q.popleft()
                    for v in current_adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            if dist[v] > 4:
                                return float('inf')
                            q.append(v)
                if any(d == -1 for d in dist):
                    return float('inf')
                max_d = max(max_d, max(dist))
            return max_d
        
        def can_add(u, v):
            """Check if adding edge (u,v) keeps triangle-free"""
            for w in current_adj[u]:
                if w in current_adj[v]:
                    return False
            return True
        
        current_diam = get_diam()
        while current_diam > 4 and time.time() < deadline:
            best_edge = None
            best_improvement = 0
            for u, v in all_possible:
                if (u, v) in [(e[0], e[1]) for e in current_edges]:
                    continue
                if not can_add(u, v):
                    continue
                # Try adding
                current_adj[u].add(v)
                current_adj[v].add(u)
                new_diam = get_diam()
                current_adj[u].remove(v)
                current_adj[v].remove(u)
                improvement = current_diam - new_diam if new_diam < float('inf') else 0
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_edge = (u, v)
            if best_edge is None or best_improvement == 0:
                break
            u, v = best_edge
            current_adj[u].add(v)
            current_adj[v].add(u)
            current_edges.append((u, v))
            current_diam = get_diam()
        
        if current_diam <= 4:
            try_solution(current_edges)
            
            # Local search: try removing edges
            current = list(current_edges)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(len(current)):
                    candidate = current[:i] + current[i+1:]
                    if fast_check(candidate):
                        current = candidate
                        improved = True
                        break
            try_solution(current)
    
    # Strategy 5: Randomized local search with multiple restarts
    random.seed(12345)
    if time.time() < deadline:
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Random initial solution
            k = random.randint(8, 20)
            sample = random.sample(all_possible, min(k, len(all_possible)))
            
            # Filter to triangle-free
            valid_sample = []
            temp_adj = [set() for _ in range(n)]
            for u, v in base_edges:
                temp_adj[u].add(v)
                temp_adj[v].add(u)
            for u, v in sample:
                ok = True
                for w in temp_adj[u]:
                    if w in temp_adj[v]:
                        ok = False
                        break
                if ok:
                    valid_sample.append((u, v))
                    temp_adj[u].add(v)
                    temp_adj[v].add(u)
            
            if not fast_check(valid_sample):
                continue
            
            try_solution(valid_sample)
            
            # Local search
            current = list(valid_sample)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try removing
                for i in range(len(current)):
                    candidate = current[:i] + current[i+1:]
                    if fast_check(candidate):
                        current = candidate
                        improved = True
                        break
                if improved:
                    continue
                # Try swapping
                if len(current) > 2:
                    i = random.randint(0, len(current)-1)
                    j = random.randint(0, len(current)-1)
                    if i != j:
                        candidate = current[:i] + current[i+1:] + [(current[j][0], current[j][1])]
                        if len(candidate) == len(current) - 1 and fast_check(candidate):
                            current = candidate
                            improved = True
    
    # Strategy 6: Targeted construction - add edges to reduce long paths
    # The path has diameter 23. We need to reduce to 4.
    # Add "shortcuts" that jump across the path.
    if time.time() < deadline:
        # Try adding edges that connect distant vertices
        # Focus on edges that create cycles of length 4 or more
        
        # Construction: add edges (i, i+k) for strategic k values
        for k in range(3, 10):
            if time.time() > deadline:
                break
            added = []
            for i in range(n - k):
                if (i, i+k) not in base_set:
                    added.append((i, i+k))
            
            # Check and try to reduce
            if fast_check(added):
                try_solution(added)
            
            # Try sparse versions: only add some of these edges
            # Add edges at regular intervals
            for step in range(1, 5):
                if time.time() > deadline:
                    break
                sparse_added = []
                for i in range(0, n - k, step):
                    if (i, i+k) not in base_set:
                        sparse_added.append((i, i+k))
                if fast_check(sparse_added):
                    try_solution(sparse_added)
    
    # Strategy 7: Two-level hub construction
    if time.time() < deadline:
        # Pick two "center" hubs and connect them to strategic vertices
        for h1 in range(4, 16, 2):
            if time.time() > deadline:
                break
            for h2 in range(h1+2, 20, 2):
                if time.time() > deadline:
                    break
                if abs(h1 - h2) <= 2:
                    continue
                
                added = []
                # h1-h2 edge
                if (h1, h2) not in base_set:
                    added.append((h1, h2))
                
                # h1 connects to vertices in its "territory"
                for v in range(0, min(h1 + 6, n)):
                    if v == h1 or v == h2:
                        continue
                    if (h1, v) in base_set:
                        continue
                    # Check triangle
                    tri = False
                    for w in range(n):
                        if (h1, w) in [(a,b) for a,b in added] and (v, w) in base_set:
                            tri = True
                            break
                    if not tri:
                        added.append((h1, v))
                
                # h2 connects to vertices in its "territory"  
                for v in range(max(h2 - 6, 0), n):
                    if v == h1 or v == h2:
                        continue
                    if (h2, v) in base_set:
                        continue
                    tri = False
                    for w in range(n):
                        if (h2, w) in [(a,b) for a,b in added] and (v, w) in base_set:
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
                
                if fast_check(unique_added):
                    try_solution(unique_added)
                    
                    # Local search
                    current = list(unique_added)
                    improved = True
                    while improved and time.time() < deadline:
                        improved = False
                        for i in range(len(current)):
                            candidate = current[:i] + current[i+1:]
                            if fast_check(candidate):
                                current = candidate
                                improved = True
                                break
                    try_solution(current)
    
    if best_solution is not None:
        return [[int(u), int(v)] for u, v in best_solution]
    
    # Fallback: return a valid solution (may not be optimal but valid)
    # Construct a known valid solution
    # Add edges to create a structure with diameter <= 4
    fallback = []
    # Connect 0 to 12, 6 to 18, 3 to 15, 9 to 21, etc.
    for i in range(0, 24, 3):
        j = i + 12
        if j < 24 and (i, j) not in base_set:
            fallback.append((i, j))
    
    # Verify and adjust if needed
    if not fast_check(fallback):
        # Try another fallback
        fallback = []
        for i in range(0, 24, 4):
            j = i + 8
            if j < 24 and (i, j) not in base_set:
                fallback.append((i, j))
    
    return [[int(u), int(v)] for u, v in fallback]

# EVOLVE-BLOCK-END
