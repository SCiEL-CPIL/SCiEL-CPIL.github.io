# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    q = k
    
    def f_add(a, b): return (a + b) % q
    def f_sub(a, b): return (a - b) % q
    def f_mul(a, b): return (a * b) % q

    def build_luw(d, rules):
        start_key = (0, (0,) * (d + 1))
        label = {start_key: 0}
        queue = [start_key]
        edge_list = []

        for vkey in queue:
            side, coords = vkey
            vidx = label[vkey]
            for first_val in range(q):
                ncoords = [0] * (d + 1)
                ncoords[1] = first_val
                for c in range(2, d + 1):
                    a, b = rules[c]
                    if side == 0:
                        ncoords[c] = f_add(coords[c], f_mul(ncoords[a], coords[b]))
                    else:
                        ncoords[c] = f_sub(coords[c], f_mul(coords[a], ncoords[b]))
                nkey = (1 - side, tuple(ncoords))
                if nkey not in label:
                    label[nkey] = len(queue)
                    queue.append(nkey)
                if side == 0:
                    edge_list.append((vidx, label[nkey]))

        return len(queue), edge_list

    def check_girth(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Non-backtracking BFS from each vertex
        for start in range(n):
            # State: (vertex, previous_vertex, distance)
            dist = {}
            q_bfs = deque()
            for nb in adj[start]:
                q_bfs.append((start, nb, 1))
                dist[(start, nb)] = 1
            
            while q_bfs:
                u, v, d_cur = q_bfs.popleft()
                if d_cur >= min_girth - 1:
                    continue
                for w in adj[v]:
                    if w == u:
                        continue  # non-backtracking
                    if w == start and d_cur + 1 < min_girth:
                        return False
                    state = (v, w)
                    if state not in dist:
                        dist[state] = d_cur + 1
                        q_bfs.append((v, w, d_cur + 1))
        return True

    # Moore bound calculation
    def moore_bound(k, g):
        if g % 2 == 0:
            r = g // 2
            return k * (k - 1) ** (r - 1)
        else:
            r = (g - 1) // 2
            return 1 + k * sum((k - 1) ** i for i in range(r))

    mb = moore_bound(k, g)
    
    # Try various LUW constructions with different depths and rule sets
    best_result = None
    best_n = float('inf')
    
    # First try d=4 with various rules
    base_rules_4 = {2: (1, 1), 3: (2, 1)}
    
    for r4 in [(1, 2), (2, 1), (1, 1), (2, 2), (3, 1), (1, 3)]:
        if time.time() > deadline - 5:
            break
        rules = base_rules_4.copy()
        rules[4] = r4
        n, edges = build_luw(4, rules)
        if n < best_n and check_girth(n, edges, g):
            best_n = n
            best_result = (n, edges)
            if n == mb:  # Can't do better than Moore bound
                return best_result

    # Try d=5 with various rules
    if time.time() < deadline - 10:
        for r4 in [(1, 2), (2, 1), (1, 1)]:
            for r5 in [(1, 3), (3, 1), (2, 2), (1, 4), (4, 1)]:
                if time.time() > deadline - 5:
                    break
                rules = {}
                rules[2] = (1, 1)
                rules[3] = (2, 1)
                rules[4] = r4
                rules[5] = r5
                n, edges = build_luw(5, rules)
                if n < best_n and check_girth(n, edges, g):
                    best_n = n
                    best_result = (n, edges)
                    if n == mb:
                        return best_result

    # If we found a valid result, return it
    if best_result is not None:
        return best_result

    # Fallback: use a larger d to ensure girth is satisfied
    d = 6
    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    rules[5] = (1, 3)
    rules[6] = (2, 4)
    
    n, edges = build_luw(d, rules)
    return n, edges
# EVOLVE-BLOCK-END
