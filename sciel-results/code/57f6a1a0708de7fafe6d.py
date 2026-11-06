"""Improved C6-free graph construction via local search with edge swaps."""
import time
import random
import sys

# Deadline: 150s wall-clock; use 135s as safety margin
DEADLINE = 135.0

def construct():
    start_time = time.time()
    random.seed(42)
    
    n = 40
    
    # Start with 8 disjoint K5s (104 edges)
    def make_k5_graph():
        edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    edges.append([5 * k + a, 5 * k + b])
        return edges
    
    # Build adjacency as a set of frozensets for fast lookup
    def edges_to_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges
    
    # Check if adding edge (u,v) creates a C6
    # A C6 containing edge (u,v) needs a path of length 4 from u to v in the graph without using that edge
    # i.e., we need to find if there's a path u - a - b - c - d - v where all intermediate vertices are distinct
    # and not equal to u or v
    def would_create_c6(adj, u, v):
        # BFS from u up to depth 4, tracking paths
        # We need a path of exactly 4 edges from u to v (not using the edge u-v directly since it's not in graph yet)
        # Path: u -> a -> b -> c -> d -> v, so we need dist from u to v = 4 in the current graph
        # But we also need the path to be simple (no repeated vertices)
        
        # BFS from u, tracking shortest path distance
        # If shortest path from u to v is exactly 4, then adding edge creates a C6
        # But we need to be careful: the path must not use edge (u,v) which isn't there yet
        
        # Actually, we need ANY simple path of length 4 from u to v
        # Shortest path of length 4 implies existence of such a path
        
        # BFS from u
        from collections import deque
        dist = [-1] * n
        parent = [-1] * n
        dist[u] = 0
        queue = deque([u])
        while queue:
            node = queue.popleft()
            if dist[node] >= 4:
                continue
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
        
        # Check if v is at distance exactly 4
        return dist[v] == 4
    
    # More efficient: check if there exists a path of length 4 from u to v
    # Using BFS with depth limit
    def has_path_length_4(adj, u, v):
        # BFS from u up to depth 4
        from collections import deque
        # State: (node, depth, visited_mask) - but 40 vertices is too many for bitmask
        # Instead, BFS from u to depth 2, BFS from v to depth 2, check overlap at depth 2
        # Path u-a-b-c-d-v: a is neighbor of u, b is neighbor of a, c is neighbor of b, d is neighbor of c, v is neighbor of d
        # So: dist from u to b is 2, dist from v to c is 2, and b-c is an edge
        # Wait, that's not quite right either.
        
        # Let me think again. Path of length 4 from u to v: u - x1 - x2 - x3 - v
        # x1 adjacent to u, x2 adjacent to x1, x3 adjacent to x2, v adjacent to x3
        # All vertices distinct.
        
        # Approach: For each pair (x1, x3) where x1 ~ u and x3 ~ v and x1 != x3,
        # check if there's a path of length 2 from x1 to x3 (i.e., x1 ~ x2 ~ x3 for some x2)
        # with x2 != u, x2 != v, x2 != x1, x2 != x3
        
        # Actually simpler: BFS from u with depth limit 4, but we need simple paths.
        # For n=40, we can do a limited DFS.
        
        # Let's use a different approach:
        # Find all vertices at distance 2 from u (via intermediate vertex)
        # Find all vertices at distance 2 from v (via intermediate vertex)
        # If there's an edge between a vertex in dist2(u) and a vertex in dist2(v),
        # and the path is simple, then we have a C6.
        
        # dist2_u: vertices reachable from u in exactly 2 steps
        # dist2_v: vertices reachable from v in exactly 2 steps
        # If there's an edge (a,b) where a in dist2_u, b in dist2_v, and the paths are disjoint/simple...
        
        # This is getting complex. Let me just do a proper BFS/DFS.
        
        # DFS from u, depth up to 4, looking for v
        # With pruning: don't revisit vertices on current path
        def dfs(node, depth, visited):
            if depth == 4:
                return node == v
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if dfs(neighbor, depth + 1, visited):
                        return True
                    visited.remove(neighbor)
            return False
        
        visited = {u}
        return dfs(u, 0, visited)
    
    # Even better: use the fact that we only need to check if adding (u,v) creates ANY C6
    # which means there exists a path of length 4 from u to v in the current graph
    # Let's optimize by checking dist2 from both ends
    
    def would_create_c6_fast(adj, u, v):
        # Get vertices at distance 1 from u (neighbors of u, excluding v)
        # Get vertices at distance 1 from v (neighbors of v, excluding u)
        # Then check if there's a path of length 2 between any neighbor of u and any neighbor of v
        # That would give path: u - a - b - c - v where a~u, c~v, a~b~c
        # Wait, that's path length 4: u-a-b-c-v. Yes!
        
        # neighbors of u (excluding v)
        nu = [x for x in adj[u] if x != v]
        # neighbors of v (excluding u)
        nv = [x for x in adj[v] if x != u]
        
        # For each a in nu, c in nv, check if there's a vertex b such that a~b and b~c
        # and b != u, b != v, b != a, b != c
        # Also need the full path to be simple, but if a~b~c and a!=c, and we ensure b is distinct,
        # the path u-a-b-c-v is simple as long as a,b,c are all distinct and not u,v.
        
        for a in nu:
            for c in nv:
                if a == c:
                    continue
                # Check if there's a common neighbor of a and c (other than u, v, a, c)
                common = adj[a] & adj[c]
                # Remove u, v, a, c from common
                common.discard(u)
                common.discard(v)
                common.discard(a)
                common.discard(c)
                if common:
                    return True
        return False
    
    # Initialize
    edges = make_k5_graph()
    adj = edges_to_adj(edges)
    best_edges = list(edges)
    best_count = len(edges)
    
    # Local search: try adding random non-edges, if it doesn't create C6, add it
    # Then try removing edges that are in many C6s... actually we just want to add edges
    # Also try edge swaps: remove one edge, add another
    
    def try_add_edges(adj, edges, time_limit):
        nonlocal best_edges, best_count
        # Try all non-edges in random order
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        existing = set()
        for u, v in edges:
            existing.add(frozenset((u, v)))
        
        non_edges = [(i, j) for i, j in all_pairs if frozenset((i, j)) not in existing]
        random.shuffle(non_edges)
        
        added = 0
        for u, v in non_edges:
            if time.time() > time_limit:
                break
            if not would_create_c6_fast(adj, u, v):
                # Safe to add
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                added += 1
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = list(edges)
        
        return edges, adj
    
    def try_edge_swaps(adj, edges, time_limit):
        nonlocal best_edges, best_count
        # Try swapping: remove edge (a,b), add edge (c,d)
        # This is expensive, so do a limited version
        edge_list = list(edges)
        random.shuffle(edge_list)
        
        swaps = 0
        for i, (a, b) in enumerate(edge_list):
            if time.time() > time_limit:
                break
            # Remove edge (a,b)
            adj[a].discard(b)
            adj[b].discard(a)
            edges.remove([a, b])
            
            # Try to add a random non-edge
            candidates = []
            for _ in range(20):
                c = random.randint(0, n - 1)
                d = random.randint(0, n - 1)
                if c == d or c > d:
                    continue
                if [c, d] in edges or [d, c] in edges:
                    continue
                if (c, d) == (a, b) or (c, d) == (b, a):
                    continue
                if not would_create_c6_fast(adj, c, d):
                    candidates.append((c, d))
                    break
            
            if candidates:
                c, d = candidates[0]
                adj[c].add(d)
                adj[d].add(c)
                edges.append([c, d])
                swaps += 1
            else:
                # Restore original edge
                adj[a].add(b)
                adj[b].add(a)
                edges.append([a, b])
            
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = list(edges)
        
        return edges, adj
    
    # Phase 1: Add edges greedily
    edges, adj = try_add_edges(adj, edges, start_time + 30)
    
    # Phase 2: More rounds of adding + swapping
    for round_num in range(20):
        if time.time() > start_time + DEADLINE - 10:
            break
        
        # Try adding more edges
        edges, adj = try_add_edges(adj, edges, start_time + DEADLINE - 10)
        
        # Try edge swaps
        edges, adj = try_edge_swaps(adj, edges, start_time + DEADLINE - 10)
    
    # Phase 3: Final polish - try all possible additions
    if time.time() < start_time + DEADLINE - 5:
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        existing = set()
        for u, v in edges:
            existing.add(frozenset((u, v)))
        non_edges = [(i, j) for i, j in all_pairs if frozenset((i, j)) not in existing]
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > start_time + DEADLINE - 5:
                break
            if not would_create_c6_fast(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = list(edges)
    
    # Return the best found
    if len(best_edges) > len(edges):
        return best_edges
    return edges
