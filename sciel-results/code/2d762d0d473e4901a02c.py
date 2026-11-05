import time
import random
import math
from collections import deque

_DEADLINE = 120.0

def _verify(n, edges, k, g):
    """Verify graph is simple, k-regular, connected, girth >= g."""
    if n < 1:
        return False
    if len(edges) != len(set(edges)):
        return False
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # connectivity
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    # girth check
    # BFS from each vertex up to depth g//2
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([(start, 0)])
        while q:
            u, d = q.popleft()
            if d >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = d + 1
                    q.append((v, d + 1))
                else:
                    # Cycle found
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

def _luzw(k, g):
    """Fallback LUW construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
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

def _generate_random_bipartite_regular(n, k, max_attempts=100):
    """
    Generates a random simple k-regular bipartite graph with n vertices (n/2 in each partition).
    Returns (n, edges) or None.
    """
    if n % 2 != 0:
        return None
    
    half = n // 2
    # Partitions: 0..half-1 and half..n-1
    
    for _ in range(max_attempts):
        # Create half-edges for left side (0..half-1)
        left_stubs = []
        for i in range(half):
            for _ in range(k):
                left_stubs.append(i)
        
        # Create half-edges for right side (half..n-1)
        right_stubs = []
        for i in range(half):
            for _ in range(k):
                right_stubs.append(half + i)
        
        random.shuffle(left_stubs)
        random.shuffle(right_stubs)
        
        # Pair them
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(len(left_stubs)):
            u = left_stubs[i]
            v = right_stubs[i]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if not valid:
            continue
            
        edges = []
        for u in range(half):
            for v in adj[u]:
                edges.append((u, v))
                
        return n, edges

    return None

def _improve_girth(n, edges, k, g, time_limit):
    """
    Attempts to increase girth using 2-switches.
    Returns (n, edges) if girth >= g is achieved, else None.
    """
    start_time = time.time()
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def get_girth():
        min_g = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= min_g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            if dist[u] + dist[v] + 1 < min_g:
                                min_g = dist[u] + dist[v] + 1
        return min_g if min_g != float('inf') else n
        
    current_girth = get_girth()
    if current_girth >= g:
        return n, edges
        
    for _ in range(500):
        if time.time() - start_time > time_limit:
            break
            
        edge1 = random.choice(edges)
        edge2 = random.choice(edges)
        if edge1 == edge2:
            continue
            
        u1, v1 = edge1
        u2, v2 = edge2
        
        if u1 == u2 or v1 == v2:
            continue
        if u2 in adj[u1] or v2 in adj[v1]:
            continue
            
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        
        adj[u1].add(u2)
        adj[u2].add(u1)
        adj[v1].add(v2)
        adj[v2].add(v1)
        
        edges.remove(edge1)
        edges.remove(edge2)
        edges.append((u1, u2))
        edges.append((v1, v2))
        
        new_girth = get_girth()
        if new_girth >= g:
            return n, edges
        if new_girth > current_girth:
            current_girth = new_girth
        else:
            adj[u1].remove(u2)
            adj[u2].remove(u1)
            adj[v1].remove(v2)
            adj[v2].remove(v1)
            
            adj[u1].add(v1)
            adj[v1].add(u1)
            adj[u2].add(v2)
            adj[v2].add(u2)
            
            edges.remove((u1, u2))
            edges.remove((v1, v2))
            edges.append(edge1)
            edges.append(edge2)

    return None

def construct(k, g):
    start = time.time()
    
    if k == 4 and g == 10:
        # Try to find a graph with n < 320
        # Moore bound for bipartite (4,10) is 242.
        # Known best is 320.
        # We try n from 244 to 320 (even n only for bipartite).
        
        for n_attempt in range(244, 321, 2):
            if time.time() - start > 100:
                break
                
            found = None
            for attempt in range(50):
                if time.time() - start > 100:
                    break
                result = _generate_random_bipartite_regular(n_attempt, k, max_attempts=1)
                if result:
                    n, edges = result
                    # Quick girth check
                    adj = [set() for _ in range(n)]
                    for u, v in edges:
                        adj[u].add(v)
                        adj[v].add(u)
                    
                    valid = True
                    for s in range(n):
                        dist = [-1] * n
                        dist[s] = 0
                        q = deque([(s, 0)])
                        while q:
                            u, d = q.popleft()
                            if d >= 5:
                                continue
                            for v in adj[u]:
                                if dist[v] == -1:
                                    dist[v] = d + 1
                                    q.append((v, d + 1))
                                else:
                                    if dist[u] + dist[v] + 1 < 10:
                                        valid = False
                                        break
                            if not valid:
                                break
                        if not valid:
                            break
                            
                    if valid:
                        found = (n, edges)
                        break
                        
            if found:
                n, edges = found
                if _verify(n, edges, k, g):
                    return n, edges
                    
        # If not found, try 320 specifically with more effort
        if time.time() - start < 110:
            for attempt in range(100):
                if time.time() - start > 110:
                    break
                result = _generate_random_bipartite_regular(320, k, max_attempts=1)
                if result:
                    n, edges = result
                    improved = _improve_girth(n, edges, k, g, time_limit=5.0)
                    if improved:
                        n, edges = improved
                        if _verify(n, edges, k, g):
                            return n, edges
                    else:
                        if _verify(n, edges, k, g):
                            return n, edges

    # Fallback to LUW
    n, edges = _luzw(k, g)
    return n, edges
