"""Construct 4-regular girth-10 cage using algebraic construction + optimized verification."""
import time
import random
import numpy as np
from collections import deque

_DEADLINE = 115.0  # hard wall-clock limit (evaluator kills at 150s)


def _is_valid_4reg(adj, n):
    """Check adj is a simple 4-regular graph on n vertices."""
    if len(adj) != n:
        return False
    for i in range(n):
        nbrs = adj[i]
        if len(nbrs) != 4:
            return False
        if len(set(nbrs)) != 4:
            return False
        if i in nbrs:
            return False
        for j in nbrs:
            if i not in adj[j]:
                return False
    return True


def _girth_check_fast(adj, n, g):
    """Fast girth check using BFS on directed edges. Returns girth or g if no cycle < g."""
    # Build adjacency as numpy arrays for speed
    adj_arr = np.array(adj, dtype=np.int32)
    min_len = g
    
    # For each vertex, do a multi-source BFS on directed edges
    # State: (current_vertex, previous_vertex)
    # We want to find the shortest non-backtracking cycle
    
    # Use a distance array indexed by (prev, cur) pair
    # Since n can be up to ~320, we use a 2D array
    dist = np.full((n, n), -1, dtype=np.int32)
    
    for start in range(n):
        # Reset distance array for this start
        dist[:] = -1
        queue = deque()
        
        # Initialize: all directed edges from start
        for v in adj[start]:
            dist[start, v] = 1
            queue.append((v, start))
        
        while queue:
            cur, prev = queue.popleft()
            d = dist[prev, cur]
            
            if d >= g:
                break
            
            # Check if we've returned to start
            if cur == start and d >= 3:
                if d < min_len:
                    min_len = d
                break
            
            for w in adj[cur]:
                if w == prev:
                    continue
                if dist[cur, w] == -1:
                    dist[cur, w] = d + 1
                    queue.append((w, cur))
    
    return min_len


def _find_short_cycle_fast(adj, n, g):
    """Find one non-backtracking cycle of length < g. Returns list of vertices or None."""
    adj_arr = np.array(adj, dtype=np.int32)
    
    for start in range(n):
        # BFS from start with parent tracking
        parent = np.full((n, n), -1, dtype=np.int32)  # parent[cur, prev] = the prev of prev
        depth = np.full((n, n), -1, dtype=np.int32)
        queue = deque()
        
        for v in adj[start]:
            parent[v, start] = -1  # root
            depth[v, start] = 1
            queue.append((v, start))
        
        while queue:
            cur, prev = queue.popleft()
            d = depth[prev, cur]
            
            if d >= g:
                break
            
            if cur == start and d >= 3:
                # Reconstruct cycle
                cycle = [cur]
                state_prev = prev
                while state_prev != -1:
                    cycle.append(state_prev)
                    next_prev = parent[state_prev, cur] if state_prev != prev else -1
                    # Actually we need to track the full path
                    # Let me redo with a different approach
                    break
                # Simpler: just return that a short cycle exists
                return list(range(d))  # placeholder
            for w in adj[cur]:
                if w == prev:
                    continue
                if depth[cur, w] == -1:
                    depth[cur, w] = d + 1
                    parent[cur, w] = prev
                    queue.append((w, cur))
    return None


def _girth_check(adj, n, g):
    """Return the girth (shortest cycle length) of the graph, or g if no cycle < g."""
    return _girth_check_fast(adj, n, g)


def _random_4reg(n, rng):
    """Generate a random 4-regular graph on n vertices using configuration model."""
    half_edges = list(range(n * 4))
    rng.shuffle(half_edges)
    adj = [set() for _ in range(n)]
    valid = True
    for i in range(0, n * 4, 2):
        u = half_edges[i] // 4
        v = half_edges[i + 1] // 4
        if u == v:
            valid = False
            break
        if v in adj[u]:
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
    if not valid:
        return None
    return [sorted(adj[i]) for i in range(n)]


def _two_switch(adj, n, edge1, edge2):
    """Apply a 2-switch: remove edges (a,b) and (c,d), add (a,c) and (b,d) or (a,d) and (b,c)."""
    a, b = edge1
    c, d = edge2
    if b not in adj[a] or d not in adj[c]:
        return None
    if a != c and c not in adj[a] and b != d and d not in adj[b]:
        new_adj = [list(x) for x in adj]
        new_adj[a].remove(b)
        new_adj[b].remove(a)
        new_adj[c].remove(d)
        new_adj[d].remove(c)
        new_adj[a].append(c)
        new_adj[c].append(a)
        new_adj[b].append(d)
        new_adj[d].append(b)
        return new_adj
    if a != d and d not in adj[a] and b != c and c not in adj[b]:
        new_adj = [list(x) for x in adj]
        new_adj[a].remove(b)
        new_adj[b].remove(a)
        new_adj[c].remove(d)
        new_adj[d].remove(c)
        new_adj[a].append(d)
        new_adj[d].append(a)
        new_adj[b].append(c)
        new_adj[c].append(b)
        return new_adj
    return None


def construct(k, g):
    """Construct a k-regular graph of girth >= g with as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + _DEADLINE - 5  # safety margin
    
    if k != 4:
        return _luw_fallback(k, g)
    
    # Primary: use L-U-W algebraic construction
    n_luw, edges_luw = _luw_fallback(4, g)
    
    # Verify the L-U-W result
    adj = [[] for _ in range(n_luw)]
    for u, v in edges_luw:
        adj[u].append(v)
        adj[v].append(u)
    
    girth = _girth_check(adj, n_luw, g)
    if girth >= g:
        return n_luw, edges_luw
    
    # If L-U-W doesn't work, try random + local search
    rng = random.Random(42)
    best_n = n_luw
    best_edges = edges_luw
    
    # Try random constructions with local search
    for n in range(244, 330, 2):
        if time.time() > deadline:
            break
        if n >= best_n:
            break
        for attempt in range(3):
            if time.time() > deadline:
                break
            adj = _random_4reg(n, rng)
            if adj is None:
                continue
            girth = _girth_check(adj, n, g)
            if girth >= g:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_n = n
                best_edges = edges
                break
            else:
                adj = _local_search(adj, n, g, rng, deadline)
                if adj is not None:
                    girth = _girth_check(adj, n, g)
                    if girth >= g:
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        best_n = n
                        best_edges = edges
                        break
    
    return best_n, best_edges


def _find_short_cycle(adj, n, g):
    """Find one non-backtracking cycle of length < g. Returns list of vertices or None."""
    for u in range(n):
        for v in adj[u]:
            parent = {}
            depth = {}
            queue = deque()
            queue.append((v, u))
            parent[(v, u)] = None
            depth[(v, u)] = 1
            while queue:
                cur, prev = queue.popleft()
                d = depth[(cur, prev)]
                if d >= g:
                    break
                if cur == u and d >= 3:
                    # Reconstruct cycle
                    cycle = [cur]
                    state = (cur, prev)
                    while parent[state] is not None:
                        p_state = parent[state]
                        cycle.append(p_state[0])
                        state = p_state
                    cycle.reverse()
                    return cycle
                for w in adj[cur]:
                    if w == prev:
                        continue
                    if (w, cur) not in parent:
                        parent[(w, cur)] = (cur, prev)
                        depth[(w, cur)] = d + 1
                        queue.append((w, cur))
    return None


def _local_search(adj, n, g, rng, deadline):
    """Local search: apply 2-switches to break short cycles."""
    adj = [list(x) for x in adj]
    for iteration in range(200):
        if time.time() > deadline:
            break
        girth = _girth_check(adj, n, g)
        if girth >= g:
            return adj
        
        cycle = _find_short_cycle(adj, n, g)
        if cycle is None:
            return adj
        
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        if len(edges) < 2:
            break
        
        cycle_edges = []
        for i in range(len(cycle)):
            a, b = cycle[i], cycle[(i + 1) % len(cycle)]
            if a < b:
                cycle_edges.append((a, b))
            else:
                cycle_edges.append((b, a))
        
        found_improvement = False
        for attempt in range(10):
            if time.time() > deadline:
                break
            e1 = rng.choice(cycle_edges)
            e2 = rng.choice(edges)
            if e1 == e2:
                continue
            new_adj = _two_switch(adj, n, e1, e2)
            if new_adj is None:
                continue
            new_girth = _girth_check(new_adj, n, g)
            if new_girth > girth:
                adj = new_adj
                girth = new_girth
                found_improvement = True
                break
        
        if not found_improvement:
            for attempt in range(3):
                if time.time() > deadline:
                    break
                e1 = rng.choice(edges)
                e2 = rng.choice(edges)
                if e1 == e2:
                    continue
                new_adj = _two_switch(adj, n, e1, e2)
                if new_adj is None:
                    continue
                new_girth = _girth_check(new_adj, n, g)
                if new_girth > girth:
                    adj = new_adj
                    girth = new_girth
                    break
    return adj


def _luw_fallback(k, g):
    """Fallback to Lazebnik-Ustimenko-Woldar construction."""
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
