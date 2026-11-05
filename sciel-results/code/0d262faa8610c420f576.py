import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    if k != 5 or g != 9:
        return _luw_construct(k, g)

    n = 1116
    
    max_restarts = 500
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        graph = _try_greedy(n, k, g, seed=restart, deadline=deadline)
        if graph is not None:
            if _verify(graph, k, g):
                return n, graph
                
    return _luw_construct(k, g)

def _try_greedy(n, k, g, seed=0, deadline=0):
    random.seed(seed)
    
    adj = [[] for _ in range(n)]
    edges = []
    
    perm = list(range(n))
    random.shuffle(perm)
    for i in range(n):
        u = i
        v = perm[i]
        if u not in adj[v]:
            adj[u].append(v)
            adj[v].append(u)
            edges.append((u, v))
            
    if not _check_2reg_girth(adj, n, g):
        return None
        
    candidates = [i for i in range(n) if len(adj[i]) < k]
    
    while len(candidates) > 0:
        if time.time() > deadline - 15:
            return None
            
        idx = random.randrange(len(candidates))
        u = candidates[idx]
        
        found = False
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts and not found:
            v_idx = random.randrange(len(candidates))
            v = candidates[v_idx]
            if u == v:
                continue
            if v in adj[u]:
                continue
                
            if _dist_geq(adj, u, v, g - 1):
                adj[u].append(v)
                adj[v].append(u)
                edges.append((u, v))
                
                if len(adj[u]) == k:
                    try:
                        ui = candidates.index(u)
                        candidates[ui] = candidates[-1]
                        candidates.pop()
                    except ValueError:
                        pass
                if len(adj[v]) == k:
                    try:
                        vi = candidates.index(v)
                        candidates[vi] = candidates[-1]
                        candidates.pop()
                    except ValueError:
                        pass
                found = True
            attempts += 1
            
        if not found:
            return None
            
    return edges

def _check_2reg_girth(adj, n, g):
    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            curr = i
            prev = -1
            length = 0
            while True:
                visited[curr] = True
                length += 1
                nexts = [x for x in adj[curr] if x != prev]
                if not nexts:
                    break
                next_node = nexts[0]
                if visited[next_node]:
                    break
                prev, curr = curr, next_node
                
            if length < g:
                return False
    return True

def _dist_geq(adj, u, v, min_dist):
    if u == v:
        return 0 >= min_dist
    if v in adj[u]:
        return 1 >= min_dist
        
    visited = {u: 0}
    queue = deque([u])
    while queue:
        curr = queue.popleft()
        curr_dist = visited[curr]
        if curr_dist >= min_dist - 1:
            return True
            
        for nei in adj[curr]:
            if nei not in visited:
                if nei == v:
                    return (curr_dist + 1) >= min_dist
                visited[nei] = curr_dist + 1
                queue.append(nei)
                
    return True

def _verify(graph, k, g):
    n = max([u for u, v in graph] + [v for u, v in graph]) + 1
    adj = [[] for _ in range(n)]
    for u, v in graph:
        adj[u].append(v)
        adj[v].append(u)
        
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    for u, v in graph:
        adj[u].remove(v)
        adj[v].remove(u)
        
        dist = _shortest_path(adj, u, v)
        
        adj[u].append(v)
        adj[v].append(u)
        
        if dist < g - 1:
            return False
            
    return True

def _shortest_path(adj, u, v):
    if u == v:
        return 0
    if v in adj[u]:
        return 1
    visited = {u: 0}
    queue = deque([u])
    while queue:
        curr = queue.popleft()
        d = visited[curr]
        for nei in adj[curr]:
            if nei not in visited:
                if nei == v:
                    return d + 1
                visited[nei] = d + 1
                queue.append(nei)
    return float('inf')

def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
