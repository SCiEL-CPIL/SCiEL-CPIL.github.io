"""Stronger program: LUW baseline + greedy randomized search for smaller k-regular girth-g graphs."""
import time
import random
from collections import deque, defaultdict

_DEADLINE = 110  # safety margin before 120s wall-clock kill

def _time_left():
    return _DEADLINE - time.time()

def _moore_lower(k, g):
    """Moore bound: minimum possible n for k-regular graph with girth g."""
    if g % 2 == 0:
        r = g // 2
        # n >= k * (k-1)^(r-1) / 2 * ... simplified: for even girth g=2r
        # n >= k * ((k-1)^(r-1) - 1) / (k-2) * 2 / k ... use standard form
        # For girth g (even): n >= k * sum_{i=0}^{r-1} (k-1)^i = k * ((k-1)^r - 1)/(k-2)
        # Actually the Moore bound for k-regular, girth g:
        # if g even (g=2r): n >= k * ((k-1)^(r-1) - 1) / (k-2) * 2 ... let me use the tree bound
        # A tree of depth r-1 from each vertex: number of distinct vertices >= 2 * sum_{i=0}^{r-1}(k-1)^i
        # = 2 * ((k-1)^r - 1)/(k-2) for k>2
        r = g // 2
        if k <= 2:
            return g
        return 2 * ((k - 1) ** r - 1) // (k - 2)
    else:
        # g odd (g=2r+1): center vertex, tree of depth r
        # n >= 1 + k * sum_{i=0}^{r-1}(k-1)^i = 1 + k * ((k-1)^r - 1)/(k-2)
        r = (g - 1) // 2
        if k <= 2:
            return g
        return 1 + k * ((k - 1) ** r - 1) // (k - 2)

def _check_girth(adj, n, g):
    """Check no cycle shorter than g using non-backtracking BFS from each vertex.
    Returns True if girth >= g."""
    if n == 0:
        return True
    # For each start vertex, do non-backtracking BFS up to depth floor((g-1)/2)
    max_depth = (g - 1) // 2
    for start in range(n):
        # BFS state: (node, prev_node) to avoid backtracking
        visited = set()
        queue = deque()
        queue.append((start, -1))
        visited.add((start, -1))
        depth = 0
        found_short = False
        while queue and depth <= max_depth:
            next_queue = deque()
            while queue:
                node, prev = queue.popleft()
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if (nb, node) in visited:
                        # Found a cycle of length depth+1 (or less)
                        if depth + 1 < g:
                            return False
                        continue
                    visited.add((nb, node))
                    next_queue.append((nb, node))
            queue = next_queue
            depth += 1
    return True

def _check_degree(adj, n, k):
    """Check every vertex has degree exactly k."""
    for i in range(n):
        if len(adj[i]) != k:
            return False
    return True

def _check_simple(adj, n):
    """Check no loops, no repeated edges."""
    for i in range(n):
        if i in adj[i]:
            return False
        s = set(adj[i])
        if len(s) != len(adj[i]):
            return False
    return True

def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    if not _check_simple(adj, n):
        return False
    if not _check_degree(adj, n, k):
        return False
    if not _check_girth(adj, n, g):
        return False
    return True

def _lazebnik_construct(k, g):
    """LUW graph construction - fast, always valid for prime k or k=4."""
    q = k
    # d such that girth >= d + 5 (for odd d) or d + 4 (for even d)
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _greedy_search(k, g, time_budget=60.0):
    """Greedy randomized search for a smaller k-regular graph with girth >= g.
    Tries to build graphs from the Moore bound upward."""
    start_time = time.time()
    best_n = None
    best_edges = None
    
    lower = _moore_lower(k, g)
    # Try a range of n values
    max_n_to_try = lower + 200  # limit search space
    
    for target_n in range(lower, min(lower + 200, 500)):
        if time.time() - start_time > time_budget:
            break
        if target_n < k + 1:
            continue
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() - start_time > time_budget:
                break
            
            # Build a random k-regular graph on target_n vertices
            adj = _random_k_regular(target_n, k)
            if adj is None:
                continue
            
            # Verify girth
            if _check_girth(adj, target_n, g):
                edges = []
                for i in range(target_n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                if best_n is None or target_n < best_n:
                    best_n = target_n
                    best_edges = edges
                break  # Found valid for this target_n, move to smaller? No, target_n increases.
            else:
                # Try to repair: remove edges that create short cycles, add others
                adj = _repair_girth(adj, target_n, k, g, time.time() - start_time, time_budget)
                if adj is not None:
                    if _check_girth(adj, target_n, g):
                        edges = []
                        for i in range(target_n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        if best_n is None or target_n < best_n:
                            best_n = target_n
                            best_edges = edges
                        break
    
    if best_n is not None:
        return best_n, best_edges
    return None, None

def _random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices using configuration model + simplification."""
    if n < k + 1 or k * n % 2 != 0:
        return None
    if k >= n:
        return None
    
    random.shuffle
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    # Randomly pair stubs
    adj = [set() for _ in range(n)]
    for _ in range(100):  # try 100 times to get a simple graph
        random.shuffle(stubs)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if valid:
            # Convert to lists
            return [list(s) for s in adj]
    return None

def _repair_girth(adj, n, k, g, elapsed, budget):
    """Try to repair a k-regular graph to have girth >= g by swapping edges."""
    if elapsed > budget:
        return None
    remaining = budget - elapsed
    max_iters = min(5000, int(remaining * 50))
    
    adj = [list(s) for s in adj]  # copy
    
    for _ in range(max_iters):
        if time.time() - (budget - remaining) > budget:
            break
        # Find a short cycle
        cycle = _find_short_cycle(adj, n, g)
        if cycle is None:
            return adj
        # Try to fix by swapping an edge in the cycle
        # Pick a random edge in the cycle and try to replace it
        idx = random.randint(0, len(cycle) - 1)
        u = cycle[idx]
        v = cycle[(idx + 1) % len(cycle)]
        # Remove edge u-v
        adj[u].remove(v)
        adj[v].remove(u)
        # Find new partners: find x in adj[u] and y in adj[v] such that x-y is not an edge
        # and adding u-y and v-x doesn't create a short cycle
        found = False
        for _try in range(50):
            x = random.choice(adj[u]) if adj[u] else None
            y = random.choice(adj[v]) if adj[v] else None
            if x is None or y is None:
                break
            if x == y or y in adj[x] or x == v or y == u:
                continue
            # Tentatively add u-y and v-x
            adj[u].append(y)
            adj[v].append(x)
            adj[y].append(u)
            adj[x].append(v)
            if _check_girth(adj, n, g):
                found = True
                break
            else:
                # Revert
                adj[u].remove(y)
                adj[v].remove(x)
                adj[y].remove(u)
                adj[x].remove(v)
        if not found:
            # Revert the removal
            adj[u].append(v)
            adj[v].append(u)
    
    return adj if _check_girth(adj, n, g) else None

def _find_short_cycle(adj, n, g):
    """Find a cycle of length < g if one exists. Returns list of vertices or None."""
    max_depth = (g - 1) // 2
    for start in range(n):
        # BFS with parent tracking
        parent = {start: None}
        depth = {start: 0}
        queue = deque([(start, -1)])
        while queue:
            node, prev = queue.popleft()
            if depth[node] >= max_depth:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb in parent:
                    # Found a cycle
                    cycle_len = depth[node] + depth[nb] + 1
                    if cycle_len < g:
                        # Reconstruct cycle
                        cycle = [node]
                        cur = node
                        while cur != nb:
                            cur = parent[cur]
                            cycle.append(cur)
                        cur = nb
                        while cur != node:
                            cur = parent[cur]
                            cycle.append(cur)
                        return cycle
                else:
                    parent[nb] = node
                    depth[nb] = depth[node] + 1
                    queue.append((nb, node))
    return None

def construct(k, g):
    """Construct a k-regular graph with girth >= g, minimizing n."""
    start_time = time.time()
    
    # Always get LUW as baseline
    luw_n, luw_edges = _lazebnik_construct(k, g)
    best_n, best_edges = luw_n, luw_edges
    
    # Try greedy search with remaining time
    remaining = _DEADLINE - (time.time() - start_time) - 5  # 5s safety margin
    if remaining > 10:
        greedy_n, greedy_edges = _greedy_search(k, g, time_budget=remaining)
        if greedy_n is not None and greedy_n < best_n:
            best_n, best_edges = greedy_n, greedy_edges
    
    return best_n, best_edges
