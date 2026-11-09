import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 135  # safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def has_triangle(n, edges_set):
        """Check if graph has any triangle using adjacency sets."""
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            return (u, v, w)
        return None
    
    def find_triangle(n, adj):
        """Find a triangle using adjacency sets. Returns (u,v,w) or None."""
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    common = adj[u] & adj[v]
                    for w in common:
                        if w > v:
                            return (u, v, w)
        return None
    
    def independence_number(n, adj):
        """Compute exact independence number using branch and bound."""
        # Convert to bitmask adjacency
        adj_mask = [0] * n
        for u in range(n):
            for v in adj[u]:
                adj_mask[u] |= (1 << v)
        
        best = [0]
        full_mask = (1 << n) - 1
        
        def bnb(remaining, current_size):
            if remaining == 0:
                best[0] = max(best[0], current_size)
                return
            if current_size + bin(remaining).count('1') <= best[0]:
                return
            # Pick a vertex
            lb = remaining & (-remaining)
            v = lb.bit_length() - 1
            # Branch: include v
            new_remaining = remaining & ~(1 << v) & ~adj_mask[v]
            bnb(new_remaining, current_size + 1)
            # Branch: exclude v
            bnb(remaining & ~(1 << v), current_size)
        
        bnb(full_mask, 0)
        return best[0]
    
    def independence_number_greedy(n, adj):
        """Quick greedy upper bound on independence number."""
        # Greedy: repeatedly pick vertex with smallest degree in remaining graph
        remaining = set(range(n))
        count = 0
        while remaining:
            # Find vertex with minimum degree in induced subgraph
            min_deg = float('inf')
            min_v = None
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and u in adj[v])
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            count += 1
            remaining.discard(min_v)
            remaining -= adj[min_v]
        return count
    
    def random_graph_with_no_triangles(n, target_edge_density=0.3):
        """Generate a random graph and remove edges to eliminate triangles."""
        adj = [set() for _ in range(n)]
        # Start with random edges
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < target_edge_density:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Iteratively remove edges to destroy triangles
        max_iters = 5000
        for _ in range(max_iters):
            tri = find_triangle(n, adj)
            if tri is None:
                break
            u, v, w = tri
            # Remove one of the three edges (random choice)
            edges_to_try = [(u,v), (v,w), (u,w)]
            random.shuffle(edges_to_try)
            for a, b in edges_to_try:
                if b in adj[a]:
                    adj[a].discard(b)
                    adj[b].discard(a)
                    break
        
        return adj
    
    def count_edges(adj, n):
        return sum(len(adj[i]) for i in range(n)) // 2
    
    def try_target_n(n, max_restarts=50):
        """Try to find valid graph on n vertices."""
        for restart in range(max_restarts):
            if elapsed() > DEADLINE:
                return None
            
            adj = random_graph_with_no_triangles(n, target_edge_density=0.35)
            
            # Check triangle-free
            if find_triangle(n, adj) is not None:
                continue
            
            # Quick check: independence number
            alpha = independence_number(n, adj)
            
            if alpha < 10:
                # Valid! Convert to edge list
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
        
        return None
    
    def try_local_search(n, adj):
        """Local search: try to reduce independence number by adding edges (if no triangle created)."""
        if elapsed() > DEADLINE:
            return None
        
        alpha = independence_number(n, adj)
        if alpha < 10:
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        
        # Try to add edges that don't create triangles
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            # Try adding random non-edges
            candidates = []
            for i in range(n):
                for j in range(i+1, n):
                    if j not in adj[i]:
                        candidates.append((i, j))
            random.shuffle(candidates)
            
            for u, v in candidates[:100]:
                if elapsed() > DEADLINE:
                    break
                # Check if adding (u,v) creates a triangle
                creates_tri = False
                for w in adj[u] & adj[v]:
                    creates_tri = True
                    break
                if not creates_tri:
                    adj[u].add(v)
                    adj[v].add(u)
                    new_alpha = independence_number(n, adj)
                    if new_alpha < alpha:
                        alpha = new_alpha
                        improved = True
                        if alpha < 10:
                            edges = []
                            for i in range(n):
                                for j in adj[i]:
                                    if i < j:
                                        edges.append([i, j])
                            return {'n': n, 'edges': edges}
                    else:
                        # Revert
                        adj[u].discard(v)
                        adj[v].discard(u)
        
        return None
    
    # Try n=40 first, then 39, then 38, etc.
    for n in [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if elapsed() > DEADLINE - 5:
            break
        
        result = try_target_n(n, max_restarts=30)
        if result is not None:
            return result
        
        # Try local search approach
        if elapsed() < DEADLINE - 5:
            adj = random_graph_with_no_triangles(n, target_edge_density=0.4)
            if find_triangle(n, adj) is None:
                result = try_local_search(n, adj)
                if result is not None:
                    return result
    
    # Fallback: Andrásfai graph
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
