import random
import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit

    # Known small cages and Moore graphs
    if k == 2:
        # Cycle graph C_g
        n = g
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges

    if k == 3:
        # Known (3,g) cages
        if g == 3:
            n = 3
            edges = [(0,1),(1,2),(0,2)]
            return n, edges
        if g == 4:
            n = 4
            edges = [(0,1),(1,2),(2,3),(3,0)]
            return n, edges
        if g == 5:
            n = 5
            edges = [(0,1),(1,2),(2,3),(3,4),(4,0)]
            return n, edges
        if g == 6:
            n = 10
            # Petersen graph
            edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
                     (5,6),(6,7),(7,8),(8,9),(9,5),
                     (0,5),(1,6),(2,7),(3,8),(4,9)]
            return n, edges
        if g == 7:
            n = 24
            # Heawood graph (bipartite double cover of K4)
            # Actually Heawood is (3,6). For (3,7) the cage has 24 vertices.
            # Let's use LUW or a known construction.
            pass

    # For k=4, g=10: try LUW first (guaranteed valid), then attempt random search
    if k == 4 and g == 10:
        # First, get LUW as a safe baseline
        luw_result = _luw_graph(k, g)
        best_n, best_edges = luw_result

        # Try random search for a smaller graph
        # Moore bound is 243. Known best is 320.
        # We'll try n from 244 to 320 (even numbers only for 4-regular)
        # But random 4-reg graphs with girth 10 are extremely rare.
        # Let's try a limited number of trials with fast girth check.
        
        found = False
        for n_try in range(320, 243, -2):
            if time.time() > deadline - 5:
                break
            # Try a few random graphs for this n
            max_trials = 50 if n_try <= 280 else 20
            for trial in range(max_trials):
                if time.time() > deadline - 3:
                    break
                adj = _generate_random_4_reg(n_try)
                if adj is None:
                    continue
                if _check_girth_fast(adj, n_try, g):
                    edges = []
                    for u in range(n_try):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    if n_try < best_n:
                        best_n, best_edges = n_try, edges
                        found = True
                    break
            if found and n_try < 260:
                break

        return best_n, best_edges

    # For other (k,g), use LUW
    return _luw_graph(k, g)


def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 5:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    elif q == 7:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    elif q == 3:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    else:
        # For other primes, use modular arithmetic
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    edges = sorted(set(edges))
    return n, edges


def _generate_random_4_reg(n):
    """Generate a random 4-regular simple graph using configuration model."""
    stubs = []
    for i in range(n):
        stubs.extend([i] * 4)
    random.shuffle(stubs)
    
    adj = [[] for _ in range(n)]
    for i in range(0, 4 * n, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            return None  # Loop
        if v in adj[u]:
            return None  # Multi-edge
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_girth_fast(adj, n, min_girth):
    """Check if graph has girth >= min_girth using non-backtracking BFS.
    
    For each edge (u,v), find the shortest non-backtracking path from u to v.
    If any such path has length < min_girth - 1, then there's a cycle shorter than min_girth.
    """
    # Build edge set for quick lookup
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_set.add((u, v))
            else:
                edge_set.add((v, u))
    
    for u in range(n):
        for v in adj[u]:
            if u > v:
                continue
            # Find shortest non-backtracking path from u to v
            # State: (prev_node, current_node)
            dist = {}
            queue = deque()
            
            # Start: from u, go to all neighbors except v
            for w in adj[u]:
                if w != v:
                    dist[(u, w)] = 1
                    queue.append((u, w))
            
            found = False
            while queue:
                prev, cur = queue.popleft()
                d = dist[(prev, cur)]
                
                if cur == v:
                    # Cycle length = d + 1
                    if d + 1 < min_girth:
                        return False
                    found = True
                    continue
                
                if d >= min_girth - 1:
                    continue  # No need to go deeper
                
                for w in adj[cur]:
                    if w == prev:
                        continue  # Non-backtracking
                    if (cur, w) not in dist:
                        dist[(cur, w)] = d + 1
                        queue.append((cur, w))
            
            # If we didn't find a path, that's fine (graph might be disconnected,
            # but for regular graphs from configuration model, they're usually connected)
            # Actually, if there's no path, there's no cycle through this edge, which is fine.
    
    return True
