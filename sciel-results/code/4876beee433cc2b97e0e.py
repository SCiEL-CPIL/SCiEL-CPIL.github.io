"""Construct a k-regular graph of girth at least g, minimizing vertex count."""
import random
import math

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g on as few vertices as possible.
    For k=5, g=9, we use a heuristic approach since the optimal (1116) is hard to construct.
    
    Strategy:
    1. Try known small constructions for specific (k,g) pairs
    2. For general case, use random k-regular graph generation with girth checking
    3. Verify girth using BFS/non-backtracking walks
    """
    # Special case: known small cages
    # For k=5, g=9: the Hoffman-Singleton graph is 7-regular girth 5, not what we want
    # The (5,9)-cage is not known in closed form; smallest known is 1116 vertices
    
    # Try to construct using a random approach for smaller graphs first
    # Moore bound for (k,g): 
    # For odd girth: n >= 2 * sum_{i=0}^{(g-3)/2} (k-1)^i
    # For even girth: n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
    
    def moore_bound(k, g):
        if g % 2 == 1:
            # odd girth
            half = (g - 1) // 2
            s = 0
            for i in range(half):
                s += (k - 1) ** i
            return 2 * s + 1  # Actually for odd girth, it's a bit different
        else:
            half = g // 2
            s = 0
            for i in range(half):
                s += (k - 1) ** i
            return 2 * s
    
    # For k=5, g=9:
    # odd: half = 4, s = 1 + 4 + 16 + 64 = 85, n >= 171? That doesn't match 428 lower bound
    # Let me recalculate: for odd girth g, Moore bound is 1 + k*sum_{i=0}^{(g-3)/2}(k-1)^i
    # = 1 + 5*(1 + 4 + 16 + 64) = 1 + 5*85 = 426. So n >= 426, and best known lower bound is 428.
    
    def moore_bound_odd(k, g):
        if g % 2 != 1:
            return 0
        half = (g - 3) // 2
        s = 0
        for i in range(half + 1):
            s += (k - 1) ** i
        return 1 + k * s
    
    def moore_bound_even(k, g):
        if g % 2 != 0:
            return 0
        half = g // 2
        s = 0
        for i in range(half):
            s += (k - 1) ** i
        return 2 * s
    
    lb = moore_bound_odd(k, g) if g % 2 == 1 else moore_bound_even(k, g)
    
    # For k=5, g=9: lb = 426
    # Known smallest: 1116
    
    # Try random construction starting from a reasonable n
    # We'll try n values and use random regular graph generation
    
    def generate_random_k_regular(n, k, seed=42):
        """Generate a random k-regular graph on n vertices."""
        if n * k % 2 != 0:
            return None
        rng = random.Random(seed)
        
        # Configuration model
        half_edges = []
        for v in range(n):
            for _ in range(k):
                half_edges.append(v)
        rng.shuffle(half_edges)
        
        edges = set()
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                # Self-loop, swap with another
                j = (i + 2) % len(half_edges)
                if j == i:
                    j = (i + 4) % len(half_edges)
                if half_edges[j] == v and half_edges[i + 1] != half_edges[j]:
                    # Try to fix
                    pass
                # Just retry
                return generate_random_k_regular(n, k, seed + 1)
            if (v, u) in edges:
                return generate_random_k_regular(n, k, seed + 1)
            edges.add((u, v))
        
        # Check if it's actually k-regular (should be by construction)
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != k for d in deg):
            return None
        
        return list(edges)
    
    def verify_girth(edges, n, g):
        """Verify that the graph has girth >= g using BFS from each vertex."""
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each vertex, do BFS to find shortest cycle
        # A cycle through v has length = dist(u, v) + 1 where u is a neighbor of v
        # that is not the parent of v in BFS, and dist is the BFS distance from v to u
        
        # Better: for each edge (u,v), the girth is at least the shortest cycle containing that edge
        # = 2 * dist(u, v) + 1 where dist is shortest path not using edge (u,v)
        # This is expensive. Instead, use BFS from each vertex.
        
        # For vertex s, BFS gives dist[s][v]. A cycle of length L exists iff
        # there is an edge (u,v) where dist[s][u] + dist[s][v] + 1 >= L and they're in same tree
        # Actually simpler: for each s, find min over edges (u,v) not incident to s's BFS tree
        # of dist[s][u] + dist[s][v] + 1
        
        # Most efficient: for each s, BFS, then for each vertex v != s, 
        # for each neighbor w of v, if parent[v] != w (in BFS tree from s),
        # then we have a cycle of length dist[s][v] + dist[s][w] + 1
        # Take min over all such cycles and all s
        
        min_cycle = n + 1  # infinity
        
        for s in range(n):
            # BFS from s
            parent = [-1] * n
            dist = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
            
            # For each vertex v (not s), for each neighbor w of v
            # if parent[v] != w, cycle length = dist[v] + dist[w] + 1
            # But we need to ensure this is a valid cycle (not just two paths to same node)
            # The cycle goes: s -> ... -> w -> v -> ... -> s
            # Length = dist[s][w] + 1 + dist[s][v]
            # But this is only a simple cycle if the paths from s to w and s to v 
            # intersect only at s. In BFS, if we use the BFS tree, the path from s to any 
            # node is unique, so the cycle s->...->w->v->...->s is simple iff 
            # the LCA of w and v in the BFS tree is s.
            # This is getting complicated. Let's use a simpler but correct method.
            
            # Alternative: for each s, the shortest cycle through s can be found by
            # removing s and finding shortest path between any two neighbors of s.
            # But that's also expensive.
            
            # Let's use: for each s, for each pair of neighbors (a, b) of s,
            # find shortest path from a to b not using s. Cycle length = path_len + 2.
            # This is O(n * k^2 * (n+k)) which might be too slow for large n.
            
            # Simpler heuristic check: just verify no cycle shorter than g by
            # doing BFS from each vertex and checking back edges
            # Actually for verification, let's do a proper girth computation
            
            # For each vertex v, for each neighbor w of v where parent[v] != w:
            # if the path from s to v and s to w share no vertices except s,
            # then we have a cycle of length dist[v] + dist[w] + 1
            # To check shared vertices, we can use a different approach
            
            # Let's just do: for each s, compute BFS. Then for each edge (u,v) in the graph,
            # if u and v are both reachable and the BFS tree paths from s to u and s to v
            # don't share any vertex except possibly s, then cycle length = dist[u] + dist[v] + 1
            # This is still hard to check efficiently.
            
            # Practical approach: for small n (< 2000), we can afford to check
            # For each s, for each neighbor a of s, for each neighbor b of s, a != b,
            # do BFS from a avoiding s, find dist to b. Min over all gives shortest cycle through s.
            
            pass  # Skip this expensive verification for now, trust construction
        
        return True  # Placeholder - we'll verify differently
    
    def compute_girth(edges, n):
        """Compute the girth (shortest cycle length) of the graph."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        girth = n + 1
        
        for s in range(n):
            # Find shortest cycle through s
            # For each pair of neighbors (a,b) of s, shortest path a->b avoiding s
            neighbors_of_s = adj[s]
            if len(neighbors_of_s) < 2:
                continue
            
            # Multi-source BFS from all neighbors of s, avoiding s
            dist = [n + 1] * n
            parent = [-1] * n
            dist[s] = -1  # forbidden
            from collections import deque
            queue = deque()
            for a in neighbors_of_s:
                dist[a] = 0
                queue.append(a)
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v == s:
                        continue
                    if dist[v] == n + 1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
            
            # For each pair of neighbors (a, b) of s, find shortest path a->b
            # In the BFS tree from all neighbors, the shortest path between a and b
            # is dist_lca related. Actually, since we did multi-source BFS,
            # the distance between a and b in the graph avoiding s is:
            # We need to find the actual shortest path, not just BFS from all sources.
            
            # Better: for each neighbor a of s, do BFS from a avoiding s, 
            # record dist_a[b] for each other neighbor b of s.
            # Then min over a < b of dist_a[b] + 2 gives shortest cycle through s.
            
            for i, a in enumerate(neighbors_of_s):
                # BFS from a avoiding s
                dist_a = [-1] * n
                dist_a[a] = 0
                qa = [a]
                qia = 0
                while qia < len(qa):
                    u = qa[qia]
                    qia += 1
                    for v in adj[u]:
                        if v == s:
                            continue
                        if dist_a[v] == -1:
                            dist_a[v] = dist_a[u] + 1
                            qa.append(v)
                
                for j in range(i + 1, len(neighbors_of_s)):
                    b = neighbors_of_s[j]
                    if dist_a[b] != -1:
                        cycle_len = dist_a[b] + 2
                        if cycle_len < girth:
                            girth = cycle_len
        
        return girth if girth <= n else 2 * n  # No cycle found (shouldn't happen for connected)
    
    # For k=5, g=9, try to find a valid graph
    # Start with n close to known bounds and try random graphs
    
    if k == 5 and g == 9:
        # Try n values from a reasonable range
        # Known smallest: 1116, lower bound: 428
        # Let's try to find something valid, even if not optimal
        # Start with smaller n and increase
        
        candidates = []
        # Try some specific n values
        base_ns = [500, 600, 700, 800, 900, 1000, 1100, 1116, 1200]
        
        # Also try n based on Moore bound with some factor
        mb = moore_bound_odd(5, 9)  # 426
        for factor in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]:
            n_try = int(mb * factor)
            if n_try not in candidates:
                candidates.append(n_try)
        
        candidates = sorted(set(candidates))
        
        for n_try in candidates:
            if n_try < lb or n_try * 5 % 2 != 0:
                continue
            
            # Try multiple random seeds
            for seed in range(20):
                edges = generate_random_k_regular(n_try, 5, seed)
                if edges is None:
                    continue
                
                # Quick check: compute girth
                girth = compute_girth(edges, n_try)
                if girth >= 9:
                    return n_try, edges
        
        # If we couldn't find one, try larger n
        for n_try in range(1200, 2000, 50):
            if n_try * 5 % 2 != 0:
                continue
            for seed in range(10):
                edges = generate_random_k_regular(n_try, 5, seed)
                if edges is None:
                    continue
                girth = compute_girth(edges, n_try)
                if girth >= 9:
                    return n_try, edges
        
        # Last resort: return a valid but possibly larger graph
        n_try = 2000
        while True:
            edges = generate_random_k_regular(n_try, 5, 42)
            if edges is not None:
                girth = compute_girth(edges, n_try)
                if girth >= 9:
                    return n_try, edges
            n_try += 100
    
    else:
        # General case: try random k-regular graphs
        mb = moore_bound_odd(k, g) if g % 2 == 1 else moore_bound_even(k, g)
        
        for factor in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
            n_try = int(mb * factor)
            if n_try < lb or n_try * k % 2 != 0:
                continue
            for seed in range(10):
                edges = generate_random_k_regular(n_try, k, seed)
                if edges is None:
                    continue
                girth = compute_girth(edges, n_try)
                if girth >= g:
                    return n_try, edges
        
        # Increase n
        n_try = int(mb * 5)
        while True:
            if n_try * k % 2 != 0:
                n_try += 1
                continue
            for seed in range(5):
                edges = generate_random_k_regular(n_try, k, seed)
                if edges is None:
                    continue
                girth = compute_girth(edges, n_try)
                if girth >= g:
                    return n_try, edges
            n_try += 50

# EVOLVE-BLOCK-END
