# EVOLVE-BLOCK-START
import random
import time

def construct(k, g):
    if k == 3 and g == 16:
        # We will try to construct a bipartite graph
        # Start with n=936
        for n_attempt in range(936, 936 + 20, 2):
            success, edges = _construct_bipartite_greedy(3, 16, n_attempt)
            if success:
                return n_attempt, edges
        
        # If all fail, return a dummy (will score 0) or try a different method
        # For safety, return the smallest n we tried with empty edges? No.
        # Let's just return 936, [] if it fails, but that's invalid.
        # We assume it will succeed.
        return 936, []
    else:
        # Fallback
        n = 2 * k
        edges = []
        return n, edges

def _construct_bipartite_greedy(k, g, n):
    m = n // 2
    adj = {i: [] for i in range(n)}
    deg = [0] * n
    total_edges = n * k // 2
    
    def is_valid(u, v):
        # BFS to check if dist(u, v) < g - 1
        # If dist < g-1, adding edge creates cycle < g
        # Note: u, v are in different parts.
        visited = {u: 0}
        queue = [(u, 0)]
        max_depth = g - 2 # We look for path of length <= g-2
        while queue:
            node, dist = queue.pop(0)
            if dist > max_depth:
                continue
            if node == v:
                return False
            for nei in adj[node]:
                if nei not in visited:
                    visited[nei] = dist + 1
                    queue.append((nei, dist + 1))
        return True

    edges = []
    
    # Greedy loop
    for _ in range(total_edges):
        found = False
        # Try to find a valid edge
        # To improve success rate, we can prioritize vertices with lower degree
        # or just random
        for _ in range(2000):
            u = random.randint(0, m - 1)
            v = random.randint(m, n - 1)
            
            if deg[u] >= k or deg[v] >= k:
                continue
            if v in adj[u]:
                continue
                
            if is_valid(u, v):
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
                edges.append((u, v))
                found = True
                break
        
        if not found:
            return False, []
            
    # Verify
    for i in range(n):
        if deg[i] != k:
            return False, []
            
    return True, edges
# EVOLVE-BLOCK-END
