"""Construct a 3-regular graph with girth >= g on as few vertices as possible."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 140  # 10s safety margin
    
    def generate_random_3reg(n):
        """Generate a random 3-regular graph on n vertices using configuration model."""
        if n % 2 != 0 or n < 4:
            return None
        
        # Create stubs: each vertex has 3 stubs
        stubs = []
        for v in range(n):
            stubs.extend([v, v, v])
        
        random.shuffle(stubs)
        
        # Pair up stubs to create edges
        edges = set()
        adj = [[] for _ in range(n)]
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                continue  # Skip self-loops
            if (min(u,v), max(u,v)) in edges:
                continue  # Skip multi-edges
            edges.add((min(u,v), max(u,v)))
            adj[u].append(v)
            adj[v].append(u)
        
        # Check if all vertices have degree 3
        for v in range(n):
            if len(adj[v]) != 3:
                return None
        
        return adj, list(edges)
    
    def check_girth(n, adj, g):
        """Check if graph has girth >= g using BFS from each vertex."""
        r = g // 2 - 1
        if r < 0:
            return True
        
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            counts = [0] * (r + 1)
            counts[0] = 1
            
            while queue:
                u = queue.popleft()
                if dist[u] >= r:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] <= r:
                            counts[dist[v]] += 1
                            queue.append(v)
            
            # Check if counts match the tree structure
            expected = [1] + [k * (k - 1) ** i for i in range(r)]
            if counts != expected:
                return False
        
        return True
    
    # Start with small n and increase
    # Moore bound for 3-regular girth 16: 1 + 3 + 6 + 12 + 24 + 48 = 95 (lower bound)
    # But we need to find a valid graph, so start from a reasonable size
    min_n = 2 * (k * (k-1) ** (g//2 - 1))  # Bipartite lower bound
    
    # Try different sizes, starting from a reasonable estimate
    # For girth 16, 3-regular, known constructions give ~936 vertices
    # But let's try smaller first
    base_n = max(min_n, 100)
    
    best_n = None
    best_edges = None
    
    # Try sizes: base_n, base_n + 2, base_n + 4, ... (must be even for 3-regular)
    # Also try some heuristic sizes
    sizes_to_try = []
    for size in range(base_n, base_n + 5000, 2):
        sizes_to_try.append(size)
    
    # Also try some specific sizes that might work
    for size in [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536]:
        if size >= base_n and size % 2 == 0:
            sizes_to_try.append(size)
    
    # Remove duplicates and sort
    sizes_to_try = sorted(set(sizes_to_try))
    
    for n in sizes_to_try:
        if time.time() > deadline:
            break
        
        # Try multiple random graphs of this size
        max_attempts = 200 if n < 500 else 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            result = generate_random_3reg(n)
            if result is None:
                continue
            
            adj, edges = result
            if len(edges) != n * k // 2:
                continue
            
            if check_girth(n, adj, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                    # Found a valid graph, return it
                    return n, edges
    
    # If no valid graph found, return the smallest valid one we could find
    # or return an error-like result
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: return a simple valid graph if possible
    # For now, return None or raise an error
    raise ValueError("Could not construct a valid graph")
# EVOLVE-BLOCK-END
