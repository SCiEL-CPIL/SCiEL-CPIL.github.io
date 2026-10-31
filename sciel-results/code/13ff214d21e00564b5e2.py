"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random

def _has_short_cycle(adj, n, g):
    """Check if graph has any cycle of length < g using non-backtracking BFS from each vertex."""
    # For each start vertex, do non-backtracking walks up to length g-1
    # If we return to start, we found a short cycle
    for start in range(n):
        # non-backtracking: (current, prev) -> next where next != prev
        # Track visited (vertex, prev) pairs to avoid infinite loops, but we limit depth
        visited = set()
        frontier = [(start, -1)]
        depth = 0
        while frontier and depth < g - 1:
            new_frontier = []
            for (v, prev) in frontier:
                if (v, prev) in visited:
                    continue
                visited.add((v, prev))
                for w in adj[v]:
                    if w == prev:
                        continue
                    if w == start and depth + 1 >= 3:  # cycle found (length >= 3)
                        return True
                    new_frontier.append((w, v))
            frontier = new_frontier
            depth += 1
    return False

def _find_short_cycle_length(adj, n, g):
    """Return the length of the shortest cycle found, or g if none < g."""
    best = g
    for start in range(n):
        # BFS non-backtracking
        # State: (vertex, prev_vertex), distance from start
        from collections import deque
        dist = {}
        dq = deque()
        dq.append((start, -1, 0))
        dist[(start, -1)] = 0
        while dq:
            v, prev, d = dq.popleft()
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    if d + 1 < best:
                        best = d + 1
                    continue
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append((w, v, d + 1))
    return best

def _generate_random_3reg(n):
    """Generate a random 3-regular graph on n vertices using configuration model."""
    if n % 2 != 0:
        return None
    # Stubs: each vertex has 3 stubs
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    _random.shuffle(stubs)
    # Pair stubs
    adj = [set() for _ in range(n)]
    edges = []
    for i in range(0, n * 3, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None  # loop
        if v in adj[u]:
            return None  # multi-edge
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
    # Check connectedness
    visited = set()
    stack = [0]
    visited.add(0)
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in visited:
                visited.add(w)
                stack.append(w)
    if len(visited) != n:
        return None
    return adj, edges

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    if k != 3:
        # Fallback to LUW for other k values
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
        else:
            add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,) + (0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    
    # For k=3: try random 3-regular graphs with girth verification
    # Start from Moore bound and go up
    moore_bound = 2 * (2**(g//2 - 1) - 1) + 2  # for k=3, even g
    # Actually Moore bound for k=3, girth g: 2*sum_{i=0}^{g/2-1} 2^i = 2*(2^(g/2)-1) for even g
    # For g=16: 2*(2^8-1) = 510
    if g % 2 == 0:
        n_min = 2 * (2**(g//2) - 1)
    else:
        n_min = 2 * (2**(g//2 - 1) - 1) + 2
    
    # Try sizes starting from a reasonable lower bound
    # Random 3-reg graphs have many short cycles, so we need larger n
    # Let's try n from 512 upward in steps
    candidate_sizes = []
    n_try = n_min
    while n_try < 20000:
        if n_try % 2 == 0:
            candidate_sizes.append(n_try)
        n_try += 64  # step by 64
    
    _random.seed(42)
    
    for n in candidate_sizes:
        if _time.time() > deadline - 10:
            break
        # Try multiple random graphs at this size
        max_attempts = 200
        for attempt in range(max_attempts):
            if _time.time() > deadline - 5:
                break
            result = _generate_random_3reg(n)
            if result is None:
                continue
            adj, edges = result
            # Check girth
            if not _has_short_cycle(adj, n, g):
                return n, edges
    
    # Fallback: LUW construction
    q = 3
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
