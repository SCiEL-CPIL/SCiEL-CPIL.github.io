import random
from collections import deque

def construct(k, g):
    def check_girth(adj, min_girth):
        n = len(adj)
        half = min_girth // 2 - 1
        for start in range(n):
            visited = {}
            queue = deque()
            queue.append((start, -1, 0))
            visited[(start, -1)] = 0
            while queue:
                v, parent, dist = queue.popleft()
                if dist >= half:
                    continue
                for u in adj[v]:
                    if u == parent:
                        continue
                    key = (u, v)
                    if key in visited:
                        cycle_len = dist + 1 + visited[key] + 1
                        if cycle_len < min_girth:
                            return False
                    elif key not in visited:
                        visited[key] = dist + 1
                        queue.append((u, v, dist + 1))
        return True
    
    def gen_bipartite(n, k):
        left_stubs = [i for i in range(n) for _ in range(k)]
        right_stubs = list(range(n)) * k
        random.shuffle(right_stubs)
        adj = [[] for _ in range(2 * n)]
        for i in range(n):
            for j in range(k):
                lv = i
                rv = n + right_stubs[i * k + j]
                adj[lv].append(rv)
                adj[rv].append(lv)
        for v in range(2 * n):
            if len(adj[v]) != len(set(adj[v])):
                return None
        return adj
    
    for n in range(121, 250):
        for seed in range(200):
            random.seed(seed * 1000 + n)
            adj = gen_bipartite(n, k)
            if adj is None:
                continue
            if check_girth(adj, g):
                edges = []
                for v in range(n):
                    for u in adj[v]:
                        edges.append((v, u - n))
                return 2 * n, edges
    return 0, []
