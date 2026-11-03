import random
from collections import deque
import time

def construct(k, g):
    # Moore Bound
    r = (g - 1) // 2
    M = 1
    for i in range(r):
        M += k * (k - 1) ** i
    
    # Target n
    n = M + 10
    
    def gen_random_regular(n, k):
        half_edges = [i // k for i in range(n * k)]
        random.shuffle(half_edges)
        adj = [[] for _ in range(n)]
        for i in range(n * k // 2):
            u = half_edges[2 * i]
            v = half_edges[2 * i + 1]
            if u == v or v in adj[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def get_girth(adj, n):
        # BFS from each node
        min_girth = float('inf')
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            visited[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if visited[u] >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        # Cycle length
                        # Find LCA
                        # This is O(n) per edge, too slow.
                        # Approximation: If visited[u] + visited[v] + 1 < min_girth, update.
                        # This is an upper bound on the cycle length if trees are disjoint, but they aren't.
                        # Actually, for local search, we just need to know if girth < g.
                        # If we find ANY cycle < g, return that length.
                        # A simple check: 
                        # If we encounter a visited node, the cycle length is at most 
                        # visited[u] + visited[v] + 1.
                        # If this sum < g, we have a short cycle.
                        if visited[u] + visited[v] + 1 < min_girth:
                            min_girth = visited[u] + visited[v] + 1
        return min_girth

    def local_search(adj, n, k, g, time_limit=110):
        start_time = time.time()
        girth = get_girth(adj, n)
        
        while girth < g and time.time() - start_time < time_limit:
            # Find a short cycle
            # Pick a random edge (u, v)
            u = random.randint(0, n-1)
            v = random.choice(adj[u])
            
            # Try to swap (u, v) with (u', v')
            # u' is a neighbor of u (not v)
            # v' is a neighbor of v (not u)
            # New edges: (u, v'), (u', v)
            # Must not create loops or multi-edges.
            
            u_prime = random.choice([x for x in adj[u] if x != v])
            v_prime = random.choice([x for x in adj[v] if x != u])
            
            # Check validity
            if v_prime == u_prime or v_prime == u or u_prime == v:
                continue
            if v_prime in adj[u] or u_prime in adj[v]:
                continue
            if u_prime == v_prime:
                continue
            
            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[u].append(v_prime)
            adj[v].append(u_prime)
            adj[u_prime].remove(v)
            adj[v_prime].remove(u)
            adj[u_prime].append(v)
            adj[v_prime].append(u)
            
            # Recalculate girth (expensive)
            new_girth = get_girth(adj, n)
            if new_girth < girth:
                # Revert
                adj[u].remove(v_prime)
                adj[v].remove(u_prime)
                adj[u].append(v)
                adj[v].append(u)
                adj[u_prime].remove(v)
                adj[v_prime].remove(u)
                adj[u_prime].append(v)
                adj[v_prime].append(u)
            else:
                girth = new_girth
                
        return adj, girth

    # Generate initial graph
    adj = None
    for _ in range(50):
        adj = gen_random_regular(n, k)
        if adj:
            break
            
    if not adj:
        return 0, []
        
    adj, girth = local_search(adj, n, k, g)
    
    if girth < g:
        # Failed
        return 0, []
        
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return n, edges
