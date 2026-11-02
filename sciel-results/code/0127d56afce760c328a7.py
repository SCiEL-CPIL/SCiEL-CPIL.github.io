"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
from collections import deque

def _check_girth(n, edges, min_girth):
    """Check if graph has no cycle shorter than min_girth using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        # BFS tracking parent to avoid backtracking
        visited = {start: (-1, 0)}  # node -> (parent, dist)
        queue = deque([(start, -1, 0)])
        while queue:
            node, parent, dist = queue.popleft()
            if dist >= min_girth // 2:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                if nb in visited:
                    # Found a cycle of length dist + visited[nb][1] + 1
                    cycle_len = dist + visited[nb][1] + 1
                    if cycle_len < min_girth:
                        return False
                else:
                    visited[nb] = (node, dist + 1)
                    queue.append((nb, node, dist + 1))
    return True

def _generate_random_4reg(n, max_tries=100):
    """Generate a random 4-regular simple graph on n vertices using configuration model."""
    for _ in range(max_tries):
        stubs = []
        for v in range(n):
            stubs.extend([v] * 4)
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edges:
                valid = False
                break
            edges.add(key)
        if not valid:
            continue
        edge_list = list(edges)
        deg = [0] * n
        for u, v in edge_list:
            deg[u] += 1
            deg[v] += 1
        if any(d != 4 for d in deg):
            continue
        return edge_list
    return None

def _count_short_cycles(n, edges, max_len):
    """Count cycles of length < max_len. Returns (count, list_of_cycles)."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    cycles = []
    for start in range(n):
        # Find cycles starting at 'start' with length < max_len
        # Simple DFS with depth limit
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) > max_len:
                continue
            for nb in adj[node]:
                if nb == path[0] and len(path) >= 3:
                    # Found a cycle
                    # Normalize cycle to avoid duplicates (optional for counting, needed for swapping)
                    # For now, just record it
                    cycles.append(path)
                elif nb not in path:
                    stack.append((nb, path + [nb]))
    return len(cycles), cycles

def _optimize_girth(n, edges, min_girth, max_iters=5000):
    """Use local search to increase girth."""
    edges_set = set((min(u,v), max(u,v)) for u,v in edges)
    adj = [[] for _ in range(n)]
    for u, v in edges_set:
        adj[u].append(v)
        adj[v].append(u)
    
    def get_short_cycle_count():
        # Efficient check: just return True/False for girth < min_girth
        # We want to minimize the number of violations
        # For speed, we just check if _check_girth is False, and if so, find one cycle to break.
        # A better metric: count of cycles of length < min_girth
        # To be fast, we'll just use _check_girth and if it fails, we do a random swap.
        # This is a "hill climbing" on the binary validity, which is weak.
        # Better: minimize the shortest cycle length.
        # Let's stick to: if valid, stop. If not, find a short cycle and swap an edge in it.
        
        if _check_girth(n, list(edges_set), min_girth):
            return 0
        return 1

    current_edges = list(edges_set)
    current_score = get_short_cycle_count()
    
    if current_score == 0:
        return current_edges

    for i in range(max_iters):
        # Find a short cycle
        # We need a specific cycle to break.
        # Let's find the shortest cycle.
        shortest_cycle = None
        min_len = min_girth
        for start in range(n):
            visited = {start: (-1, 0)}
            queue = deque([(start, -1, 0)])
            while queue:
                node, parent, dist = queue.popleft()
                if dist >= min_len // 2:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if nb in visited:
                        cycle_len = dist + visited[nb][1] + 1
                        if cycle_len < min_len:
                            min_len = cycle_len
                            # Reconstruct path? Hard here.
                            # For now, just pick a random edge in the graph to swap?
                            # No, we need to break the specific cycle.
                            # Let's just pick a random edge from the current graph and swap it with another random edge.
                            # This is random walk on the space of 4-regular graphs.
                            pass
                    else:
                        visited[nb] = (node, dist + 1)
                        queue.append((nb, node, dist + 1))
        
        # Random 2-switch
        # Pick two random edges (u,v) and (x,y) such that u!=x, u!=y, v!=x, v!=y, etc.
        # And (u,x), (v,y) are not existing edges.
        # This maintains 4-regularity.
        
        u, v = random.choice(current_edges)
        x, y = random.choice(current_edges)
        
        # Check distinctness
        if u == v or x == y: continue
        if u in (x, y) or v in (x, y): continue
        
        # Try swap 1: (u,x) and (v,y)
        e1 = (min(u,x), max(u,x))
        e2 = (min(v,y), max(v,y))
        
        if e1 in edges_set or e2 in edges_set or e1 == e2:
            # Try swap 2: (u,y) and (v,x)
            e1 = (min(u,y), max(u,y))
            e2 = (min(v,x), max(v,x))
            if e1 in edges_set or e2 in edges_set or e1 == e2:
                continue
        
        # Perform swap
        edges_set.remove((min(u,v), max(u,v)))
        edges_set.remove((min(x,y), max(x,y)))
        edges_set.add(e1)
        edges_set.add(e2)
        
        # Update adj
        adj[u].remove(v); adj[v].remove(u)
        adj[x].remove(y); adj[y].remove(x)
        adj[u].append(e1[1] if e1[0]==u else e1[0])
        adj[e1[1] if e1[0]==u else e1[0]].append(u)
        adj[v].append(e2[1] if e2[0]==v else e2[0])
        adj[e2[1] if e2[0]==v else e2[0]].append(v)
        
        current_edges = list(edges_set)
        new_score = get_short_cycle_count()
        
        if new_score == 0:
            return current_edges
        
        # If score didn't improve (still 1), we keep the move (random walk) 
        # or we could revert if we had a better metric.
        # For now, random walk is fine to explore the space.

    return current_edges

def construct(k, g):
    if k != 4 or g != 10:
        # Fallback: use L-U-W style construction for other parameters
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
    
    # For k=4, g=10: Use Local Search from near Moore Bound
    random.seed(42)
    
    # Moore bound is 243. Try from 243 to 320.
    # We will use local search to find a valid graph.
    
    for n in range(243, 321):
        # Generate initial random graph
        edges = _generate_random_4reg(n, max_tries=20)
        if edges is None:
            continue
        
        # Optimize
        optimized_edges = _optimize_girth(n, edges, g, max_iters=2000)
        
        if _check_girth(n, optimized_edges, g):
            return n, optimized_edges
            
    # Fallback: L-U-W construction (from seed)
    q = 4
    d = 5
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2), 5: (1, 3)}
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
# EVOLVE-BLOCK-END
