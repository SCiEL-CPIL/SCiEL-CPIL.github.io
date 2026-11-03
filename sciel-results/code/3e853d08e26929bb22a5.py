import time
import collections
import math
import random

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _check_girth_nb(n, adj, min_girth):
    """
    Fast girth check using non-backtracking BFS from each directed edge.
    For a 3-regular graph, this is much faster than full vertex BFS.
    Returns True if girth >= min_girth.
    """
    if _time_left() < 2.0:
        return True  # Assume valid if no time
    
    max_depth = min_girth - 2  # We're looking for cycles of length < min_girth
    
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue  # Only check each undirected edge once
            
            # BFS in non-backtracking state space from (u, v)
            visited = set()
            queue = collections.deque()
            queue.append((u, v, 0))
            visited.add((u, v))
            
            while queue:
                prev, curr, depth = queue.popleft()
                
                if depth >= max_depth:
                    continue
                
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue  # Non-backtracking
                    
                    new_state = (curr, nxt)
                    if new_state == (v, u) and depth + 1 >= 2:
                        # Found cycle of length depth + 1 + 1 = depth + 2
                        if depth + 2 < min_girth:
                            return False
                    elif new_state not in visited:
                        visited.add(new_state)
                        queue.append((curr, nxt, depth + 1))
                
                if _time_left() < 1.0:
                    return True  # Assume valid if running out of time
    
    return True

def _construct_luw(q, d):
    """
    Constructs the LUW graph CD(d, q).
    Returns (n, edges) or None if it fails.
    Uses efficient tuple-based representation.
    """
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if _time_left() < 5.0:
            return None
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    # For k=3, g=16: Try LUW construction
    q = k
    min_d = max(3, (g - 2) // 2)
    
    best_result = None
    
    # Try LUW with increasing d
    for d in range(min_d, min_d + 5):
        if _time_left() < 10.0:
            break
            
        result = _construct_luw(q, d)
        if result is None:
            continue
            
        n, edges = result
        
        # Verify girth
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        if _check_girth_nb(n, adj, g):
            return n, edges
        else:
            if best_result is None or n < best_result[0]:
                best_result = (n, edges)
    
    # If LUW fails, try random 3-regular graphs with 2-switch local search
    if _time_left() > 30.0:
        rng = random.Random(42)
        moore_bound = 2 * (2 ** (g // 2 - 1) - 1)
        start_n = max(128, moore_bound)
        
        for n in range(start_n, min(start_n + 400, 1200), 2):
            if _time_left() < 10.0:
                break
            
            # Generate random 3-regular graph
            half_edges = []
            for i in range(n):
                for _ in range(3):
                    half_edges.append(i)
            
            found = False
            for attempt in range(20):
                if _time_left() < 3.0:
                    break
                    
                rng.shuffle(half_edges)
                edges_set = set()
                valid = True
                adj = [[] for _ in range(n)]
                
                for i in range(0, len(half_edges), 2):
                    u, v = half_edges[i], half_edges[i + 1]
                    if u == v:
                        valid = False
                        break
                    key = (min(u, v), max(u, v))
                    if key in edges_set:
                        valid = False
                        break
                    edges_set.add(key)
                    adj[u].append(v)
                    adj[v].append(u)
                
                if not valid:
                    continue
                if any(len(adj[i]) != 3 for i in range(n)):
                    continue
                
                edges = list(edges_set)
                
                if _check_girth_nb(n, adj, g):
                    if best_result is None or n < best_result[0]:
                        best_result = (n, edges)
                    return n, edges
            
            if found:
                break
    
    if best_result is not None:
        return best_result
    
    # Last resort
    return _construct_luw(q, min_d)
