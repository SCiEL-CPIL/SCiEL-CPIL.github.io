"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Precompute adjacency for base
    adj_base = [set() for _ in range(n)]
    for u, v in base_edges:
        adj_base[u].add(v)
        adj_base[v].add(u)
    
    # Only even-odd edges can be added (triangle-free constraint with path)
    # Actually, we need to verify: adding edge (u,v) where u,v same parity creates triangle?
    # If u and v have same parity, then |u-v| is even. In the path, there's a path of even length between them.
    # For triangle: need w adjacent to both u and v. In path, u's neighbors are u-1,u+1 (opposite parity).
    # If u,v same parity, their path neighbors have opposite parity to u,v, so no common neighbor in path.
    # But if we add edges, it could create triangles. Let's be safe: only consider even-odd pairs.
    
    even_verts = [i for i in range(n) if i % 2 == 0]
    odd_verts = [i for i in range(n) if i % 2 == 1]
    
    # Candidate edges: all even-odd pairs not in base
    candidates = []
    for u in even_verts:
        for v in odd_verts:
            if (u, v) not in base_edges and (v, u) not in base_edges:
                candidates.append((u, v))
    
    def check_diameter(added_edges):
        """Check if graph with base + added_edges has diameter <= 4 and is triangle-free."""
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check triangle-free
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        return False  # triangle u-v-w
        
        # Check connected and diameter
        # BFS from each vertex
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            if -1 in dist:
                return False  # not connected
            if max(dist) > 4:
                return False
        return True
    
    # Try to find minimum number of edges
    # Start with small k and increase
    deadline = time.time() + 75  # 75 second deadline, 15s margin
    
    # First, try a greedy approach to get a good upper bound
    # Then try to improve
    
    # Greedy: repeatedly add the edge that most reduces the diameter
    def compute_diameter(adj):
        n_local = len(adj)
        max_d = 0
        for start in range(n_local):
            dist = [-1] * n_local
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            if -1 in dist:
                return float('inf')
            max_d = max(max_d, max(dist))
        return max_d
    
    # Greedy construction
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    greedy_added = []
    max_k = 20  # upper bound on edges to try
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        # Try all candidates
        best_edge = None
        best_diam = float('inf')
        for (u, v) in candidates:
            if (u, v) in greedy_added or (v, u) in greedy_added:
                continue
            # Check if adding (u,v) creates a triangle
            creates_tri = False
            for w in adj[u]:
                if w in adj[v]:
                    creates_tri = True
                    break
            if creates_tri:
                continue
            # Add temporarily
            adj[u].add(v)
            adj[v].add(u)
            d = compute_diameter(adj)
            # Remove
            adj[u].remove(v)
            adj[v].remove(u)
            if d < best_diam:
                best_diam = d
                best_edge = (u, v)
        
        if best_edge is None:
            break
        greedy_added.append(best_edge)
        adj[best_edge[0]].add(best_edge[1])
        adj[best_edge[1]].add(best_edge[0])
        if best_diam <= 4:
            break
    
    # Verify greedy solution
    if len(greedy_added) > 0 and check_diameter(greedy_added):
        # Try to reduce: remove edges one by one and see if still valid
        improved = list(greedy_added)
        changed = True
        while changed:
            changed = False
            for i in range(len(improved)):
                if time.time() > deadline:
                    break
                trial = improved[:i] + improved[i+1:]
                if check_diameter(trial):
                    improved = trial
                    changed = True
                    break
        if time.time() > deadline:
            return [[int(u), int(v)] for u, v in improved]
        result = [[int(u), int(v)] for u, v in improved]
        return result
    
    # If greedy didn't work, try exhaustive search for small k
    for k in range(1, 15):
        if time.time() > deadline:
            break
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                break
            if check_diameter(list(combo)):
                return [[int(u), int(v)] for u, v in combo]
    
    # Fallback: return all candidates (shouldn't happen)
    return [[int(u), int(v)] for u, v in candidates]
# EVOLVE-BLOCK-END
