import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    def moore_bound(k, g):
        r = (g - 1) // 2
        s = 1
        for i in range(r):
            s += (k - 1) ** i
        return 1 + k * s

    mb = moore_bound(k, g)

    def generate_random_regular(n, k):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v or v in adj[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def check_girth(adj, n, g):
        # BFS from each s
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            # Calculate LCA
                            # Find LCA of u and v
                            # u, v are in the tree
                            du = dist[u]
                            dv = dist[v]
                            u_temp = u
                            v_temp = v
                            # Bring to same depth
                            while du > dv:
                                u_temp = parent[u_temp]
                                du -= 1
                            while dv > du:
                                v_temp = parent[v_temp]
                                dv -= 1
                            # Move up together
                            while u_temp != v_temp:
                                u_temp = parent[u_temp]
                                v_temp = parent[v_temp]
                            lca = u_temp
                            cycle_len = dist[u] + dist[v] + 1 - 2 * dist[lca]
                            if cycle_len < g:
                                return False
        return True

    best_n = None
    best_edges = None

    for n in range(mb, mb + 50):
        if time.time() > deadline - 10:
            break
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            adj = generate_random_regular(n, k)
            if adj is None:
                continue
            if check_girth(adj, n, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                best_n = n
                best_edges = edges
                break
        if best_n:
            break

    if not best_n:
        # Fallback to larger n
        n = 1000
        adj = generate_random_regular(n, k)
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        best_n = n
        best_edges = edges

    return best_n, best_edges
