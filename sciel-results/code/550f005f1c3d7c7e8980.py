"""Construct a k-regular graph of girth at least g, minimizing vertex count."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def time_left():
        return deadline - time.time()

    # Moore bound
    if g % 2 == 0:
        moore = k + k * (k - 1) ** (g // 2 - 1)
    else:
        if k == 2:
            moore = g
        else:
            h = (g - 1) // 2
            moore = 1 + k * ((k - 1) ** h - 1) // (k - 2) if k > 2 else g

    def generate_random_regular(n, k):
        """Generate a random k-regular graph on n vertices."""
        if n * k % 2 != 0:
            return None
        for _ in range(100): # Retry 100 times
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            edge_set = set()
            valid = True
            for i in range(n * k // 2):
                u, v = stubs[2 * i], stubs[2 * i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                return adj, list(edge_set)
        return None

    def verify_girth(adj, n, min_girth):
        """Verify girth >= min_girth using BFS from each vertex."""
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= (min_girth - 1) // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            if dist[u] + dist[v] + 1 < min_girth:
                                return False
        return True

    def local_search(adj, edges, n, min_girth, time_limit):
        """Improve graph by breaking short cycles."""
        start_ls = time.time()
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        while time.time() - start_ls < time_limit and time_left() > 2:
            if verify_girth(adj, n, min_girth):
                return adj, list(edge_set)
            
            # Try random 2-opt swaps
            improved = False
            for _ in range(100):
                if time.time() - start_ls > time_limit:
                    break
                # Pick two random edges
                if len(edge_set) < 2:
                    break
                e1 = random.choice(list(edge_set))
                e2 = random.choice(list(edge_set))
                if e1 == e2: continue
                a, b = e1
                c, d = e2
                if a in e2 or b in e2:
                    continue
                
                ne1 = (min(a, c), max(a, c))
                ne2 = (min(b, d), max(b, d))
                
                if ne1 in edge_set or ne2 in edge_set:
                    continue
                if a == c or b == d:
                    continue
                
                edge_set.discard(e1)
                edge_set.discard(e2)
                edge_set.add(ne1)
                edge_set.add(ne2)
                
                adj[a].remove(b)
                adj[b].remove(a)
                adj[c].remove(d)
                adj[d].remove(c)
                
                adj[a].append(c)
                adj[c].append(a)
                adj[b].append(d)
                adj[d].append(b)
                
                improved = True
                break
            
            if not improved:
                # If no valid swap found, we might be stuck.
                # Break to avoid infinite loop
                break
        
        return adj, list(edge_set)

    # Try increasing n from Moore bound
    max_n = 1200
    for n in range(max(moore, k+1), max_n + 1):
        if time_left() < 5:
            break
            
        # Try a few random graphs for this n
        found = False
        for attempt in range(10):
            if time_left() < 5:
                break
            res = generate_random_regular(n, k)
            if res is None:
                continue
            adj, edges = res
            adj, edges = local_search(adj, edges, n, g, time_limit=min(5.0, time_left() - 5))
            
            if verify_girth(adj, n, g):
                return n, edges
            found = True

    # Fallback
    n = 1116
    res = generate_random_regular(n, k)
    if res:
        adj, edges = res
        adj, edges = local_search(adj, edges, n, g, time_limit=time_left() - 2)
        if verify_girth(adj, n, g):
            return n, edges

    return moore, []
# EVOLVE-BLOCK-END
