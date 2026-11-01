import time
import random
from collections import deque

def _verify_girth(n, edges, min_girth):
    """Return True if girth >= min_girth. Uses non-backtracking BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # Check no loops or multi-edges
    for u in range(n):
        if len(adj[u]) != 3:
            return False
        if u in adj[u]:
            return False
        if len(set(adj[u])) != 3:
            return False
    # Non-backtracking BFS from each vertex to find shortest cycle
    for start in range(n):
        # BFS state: (current, prev) -> distance
        visited = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        while queue:
            cur, prev, dist = queue.popleft()
            if dist >= min_girth:
                break
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == start:
                    if dist + 1 < min_girth:
                        return False
                    break
                state = (nb, cur)
                if state in visited:
                    if visited[state] < dist + 1:
                        continue
                    # Found a cycle
                    if visited[state] + dist + 1 < min_girth:
                        return False
                    continue
                visited[state] = dist + 1
                queue.append((nb, cur, dist + 1))
    return True

def _generalized_petersen(n, k):
    """Return edges for GP(n,k), 3-regular on 2n vertices. Valid for n>=5, 1<=k<=n/2, k!=n/2 if n even... actually k in [1, n//2]."""
    if n < 5:
        return None
    if k < 1 or k > n // 2:
        return None
    if n % 2 == 0 and k == n // 2:
        return None  # Not simple
    edges = []
    # Outer cycle: 0..n-1
    for i in range(n):
        edges.append((i, (i + 1) % n))
    # Spokes
    for i in range(n):
        edges.append((i, n + i))
    # Inner star
    for i in range(n):
        edges.append((n + i, n + (i + k) % n))
    return edges

def _random_3reg(n):
    """Generate a random 3-regular graph on n vertices (n even). Returns edges or None."""
    if n < 4 or n % 2 != 0:
        return None
    # Configuration model
    stubs = []
    for v in range(n):
        stubs.extend([v] * 3)
    random.shuffle(stubs)
    adj = [[] for _ in range(n)]
    edges = []
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None
        if v in adj[u]:
            return None
        adj[u].append(v)
        adj[v].append(u)
        edges.append((min(u, v), max(u, v)))
    # Check all degrees
    for v in range(n):
        if len(adj[v]) != 3:
            return None
    return edges

def _edge_swap(edges, adj, n):
    """Try a random 2-switch. Returns new edges, new adj or None."""
    if len(edges) < 4:
        return None
    i = random.randrange(len(edges))
    j = random.randrange(len(edges))
    if i == j:
        return None
    u, v = edges[i]
    x, y = edges[j]
    # Try swapping to (u,x) and (v,y) or (u,y) and (v,x)
    for new_e1, new_e2 in [((min(u, x), max(u, x)), (min(v, y), max(v, y))),
                            ((min(u, y), max(u, y)), (min(v, x), max(v, x)))]:
        a, b = new_e1
        c, d = new_e2
        if a == b or c == d:
            continue
        if (a, b) in edges or (b, a) in edges or (c, d) in edges or (d, c) in edges:
            continue
        if b in adj[a] or d in adj[c]:
            continue
        # Apply swap
        new_edges = list(edges)
        new_edges[i] = new_e1
        new_edges[j] = new_e2
        new_adj = [list(lst) for lst in adj]
        new_adj[a].remove(b)
        new_adj[b].remove(a)
        new_adj[c].remove(d)
        new_adj[d].remove(c)
        new_adj[a].append(c)
        new_adj[c].append(a)
        new_adj[b].append(d)
        new_adj[d].append(b)
        return new_edges, new_adj
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    if k != 3:
        # Fallback: use LUW for other k
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
    
    best_n = float('inf')
    best_edges = None
    
    # Phase 1: Search generalized Petersen graphs
    # GP(n,k) has 2n vertices, girth depends on n,k
    for n in range(5, 300):
        if time.time() > deadline:
            break
        for k_val in range(1, n // 2 + 1):
            if n % 2 == 0 and k_val == n // 2:
                continue
            edges = _generalized_petersen(n, k_val)
            if edges is None:
                continue
            total = 2 * n
            if total >= best_n:
                continue
            if _verify_girth(total, edges, g):
                best_n = total
                best_edges = edges
                break  # Move to next n
    
    # Phase 2: Random 3-regular + edge swaps
    # Try various sizes
    for target_n in range(50, 600, 10):
        if target_n % 2 != 0:
            continue
        if target_n >= best_n:
            continue
        if time.time() > deadline:
            break
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            edges = _random_3reg(target_n)
            if edges is None:
                continue
            adj = [[] for _ in range(target_n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Try to improve girth via swaps
            for step in range(500):
                if time.time() > deadline:
                    break
                if _verify_girth(target_n, edges, g):
                    if target_n < best_n:
                        best_n = target_n
                        best_edges = edges
                    break
                result = _edge_swap(edges, adj, target_n)
                if result is None:
                    break
                edges, adj = result
    
    # Fallback to LUW if nothing found
    if best_edges is None:
        q = 3
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
    
    return best_n, best_edges
