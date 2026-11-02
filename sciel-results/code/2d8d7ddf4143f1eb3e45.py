import time
import random
from collections import deque
import heapq

def _luw_construct(k, g):
    """LUW construction - fallback."""
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


def _generate_random_k_regular(k, n):
    """Generate a simple k-regular graph using configuration model."""
    if n * k % 2 != 0:
        return None
    for _ in range(200):
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if valid:
            return adj
    return None


def _nonbacktracking_girth(adj, n, limit):
    """Compute girth using non-backtracking walks (Wielandt). Returns girth or limit+1."""
    # For each directed edge, do BFS in the non-backtracking graph
    # A cycle of length g exists iff there's a non-backtracking walk of length g from an edge back to its reverse
    # More practically: for each edge (u,v), do BFS from (v,u) avoiding backtracking,
    # check if we can reach (u,v) in exactly g steps.
    
    # Actually, simpler: use the standard BFS from each vertex with parent tracking
    # but only track non-backtracking paths.
    
    # For efficiency with limit, we use the approach:
    # For each directed edge e=(u,v), compute shortest non-backtracking walk from e back to reverse(e)
    # This equals girth if such a walk exists.
    
    # Let's use a different approach: BFS from each vertex, tracking (node, prev) pairs
    # dist[(node, prev)] = shortest non-backtracking distance from (start, -1) to (node, prev)
    # A cycle through start of length L exists if dist[(start, neighbor_of_start)] = L-1 for some neighbor
    
    # Actually the cleanest: for each start vertex s, do BFS in the "line graph" (directed edges)
    # starting from all directed edges (s, v) where v is neighbor of s.
    # We want to find if we can reach (v, s) for some neighbor v of s, with path length >= 3.
    
    # For girth >= g, we need NO non-backtracking walk of length < g from (s,v) to (v',s) for any neighbors v,v' of s.
    # Equivalently, no cycle of length < g.
    
    min_girth = limit + 1
    for s in range(n):
        # BFS from (s, -1) in the non-backtracking graph
        # State: (current_node, prev_node)
        # Start: (s, -1) with distance 0
        # We want to reach (s, v) for some neighbor v of s, with distance = cycle_length - 1
        # Actually: a cycle s-v1-v2-...-vk-s has length k+1
        # Non-backtracking walk: (s,-1) -> (v1,s) -> (v2,v1) -> ... -> (s,vk)
        # This has length k, and the cycle has length k+1.
        # So we need to find the shortest non-backtracking walk from (s,-1) to (s, v) for any neighbor v.
        # The cycle length = walk_length + 1.
        
        visited = {}  # (node, prev) -> dist
        queue = deque()
        for v in adj[s]:
            state = (v, s)
            visited[state] = 1
            queue.append(state)
        
        while queue:
            (node, prev), dist = queue.popleft()
            if dist >= limit:
                continue
            if node == s and prev != -1:
                # Found a cycle of length dist + 1
                cycle_len = dist + 1
                if cycle_len < min_girth:
                    min_girth = cycle_len
                # Don't continue expanding from s
                continue
            for nbr in adj[node]:
                if nbr == prev:
                    continue  # non-backtracking
                state = (nbr, node)
                if state not in visited:
                    visited[state] = dist + 1
                    queue.append(state)
        
        if min_girth <= 3:
            break
    
    return min_girth


def _find_short_cycle_nb(adj, n, max_len):
    """Find a cycle of length <= max_len using non-backtracking BFS. Returns list of vertices or None."""
    for s in range(n):
        # BFS from (s, -1), track parent
        parent = {}  # (node, prev) -> (prev_node, prev_prev)
        dist = {}
        queue = deque()
        for v in adj[s]:
            state = (v, s)
            parent[state] = (-1, -1)  # came from start
            dist[state] = 1
            queue.append(state)
        
        while queue:
            (node, prev), d = queue.popleft()
            if d >= max_len:
                continue
            if node == s:
                # Found cycle of length d + 1
                if d + 1 <= max_len:
                    # Reconstruct
                    cycle = [s, node]
                    cur = (node, prev)
                    while parent[cur][0] != -1:
                        cur = parent[cur]
                        cycle.append(cur[0])
                    cycle.reverse()
                    return cycle
            for nbr in adj[node]:
                if nbr == prev:
                    continue
                state = (nbr, node)
                if state not in dist:
                    dist[state] = d + 1
                    parent[state] = (node, prev)
                    queue.append(state)
    return None


def _check_switch_valid(adj, u, v, w, x, g, n):
    """Check if 2-switch removing (u,v),(w,x) and adding (u,w),(v,x) keeps girth >= g.
    We need to verify no new short cycles are created.
    New cycles must use at least one of the new edges (u,w) or (v,x).
    """
    # Temporarily remove edges
    adj[u].discard(v)
    adj[v].discard(u)
    adj[w].discard(x)
    adj[x].discard(w)
    
    # Check if adding (u,w) creates a short cycle:
    # Need path from u to w of length < g-1 in current graph
    # Check if adding (v,x) creates a short cycle:
    # Need path from v to x of length < g-1 in current graph
    
    def has_path(src, dst, max_dist):
        if src == dst:
            return True
        visited = {src: 0}
        q = deque([(src, 0)])
        while q:
            node, d = q.popleft()
            if d >= max_dist:
                continue
            for nbr in adj[node]:
                if nbr == dst:
                    return True
                if nbr not in visited:
                    visited[nbr] = d + 1
                    q.append((nbr, d + 1))
        return False
    
    # A cycle using (u,w) has length = 1 + dist(u,w) in G-{(u,v),(w,x)}
    # We need this to be >= g, so dist(u,w) >= g-1
    if has_path(u, w, g - 2):
        # Short path exists, would create cycle of length < g
        adj[u].add(v)
        adj[v].add(u)
        adj[w].add(x)
        adj[x].add(w)
        return False
    
    if has_path(v, x, g - 2):
        adj[u].add(v)
        adj[v].add(u)
        adj[w].add(x)
        adj[x].add(w)
        return False
    
    # Also need to check cycles using both new edges
    # Cycle: u-w-...-x-v-...-u or u-v-...-x-w-...-u etc.
    # This is more complex. For now, just check the simple cases.
    
    # Restore
    adj[u].add(w)
    adj[w].add(u)
    adj[v].add(x)
    adj[x].add(v)
    return True


def _switching_construct(k, g, n, time_limit):
    """Construct k-regular graph of girth >= g using 2-switches."""
    start_time = time.time()
    
    adj = _generate_random_k_regular(k, n)
    if adj is None:
        return None
    
    # Check initial girth
    girth = _nonbacktracking_girth(adj, n, g)
    if girth >= g:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    # Iteratively fix short cycles
    max_iter = 50000
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        cycle = _find_short_cycle_nb(adj, n, g - 1)
        if cycle is None:
            break
        
        # Try to break this cycle with a 2-switch
        cl = len(cycle)
        if cl < 4:
            # Triangle: pick an edge on the triangle and switch it
            u = cycle[0]
            v = cycle[1]
            # Find another edge (w,x) to switch with
            found = False
            for w in range(n):
                if w in (u, v):
                    continue
                for x in adj[w]:
                    if x in (u, v, w) or x == v:
                        continue
                    if x in adj[u] or w in adj[v]:
                        continue
                    # Check validity
                    if _check_switch_valid(adj, u, v, w, x, g, n):
                        found = True
                        break
                if found:
                    break
            if not found:
                # Try random switch
                for _ in range(10):
                    u2 = random.randrange(n)
                    if not adj[u2]:
                        continue
                    v2 = random.choice(list(adj[u2]))
                    w2 = random.randrange(n)
                    if w2 in (u2, v2) or not adj[w2]:
                        continue
                    x2 = random.choice(list(adj[w2]))
                    if x2 in (u2, v2, w2) or x2 == v2:
                        continue
                    if x2 in adj[u2] or w2 in adj[v2]:
                        continue
                    if _check_switch_valid(adj, u2, v2, w2, x2, g, n):
                        found = True
                        break
            if not found:
                continue
        else:
            # Pick two non-adjacent edges on the cycle
            found = False
            for _ in range(20):
                i1 = random.randrange(cl)
                i2 = random.randrange(cl)
                # Ensure edges are non-adjacent
                if abs(i1 - i2) <= 1 or abs(i1 - i2) >= cl - 1:
                    continue
                u1, v1 = cycle[i1], cycle[(i1 + 1) % cl]
                u2, v2 = cycle[i2], cycle[(i2 + 1) % cl]
                if u2 in adj[u1] or v2 in adj[v1]:
                    continue
                if _check_switch_valid(adj, u1, v1, u2, v2, g, n):
                    found = True
                    break
            if not found:
                # Random switch
                u1 = random.randrange(n)
                if not adj[u1]:
                    continue
                v1 = random.choice(list(adj[u1]))
                u2 = random.randrange(n)
                if u2 in (u1, v1) or not adj[u2]:
                    continue
                v2 = random.choice(list(adj[u2]))
                if v2 in (u1, v1, u2) or v2 == v1:
                    continue
                if v2 in adj[u1] or u2 in adj[v1]:
                    continue
                if _check_switch_valid(adj, u1, v1, u2, v2, g, n):
                    found = True
                if not found:
                    continue
        
        # Verify
        girth = _nonbacktracking_girth(adj, n, g)
        if girth >= g:
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
    
    # Final check
    girth = _nonbacktracking_girth(adj, n, g)
    if girth >= g:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    return None


def construct(k, g):
    if k == 5 and g == 9:
        total_time = 140.0
        start_time = time.time()
        
        # Try switching with various n values
        # Moore bound for (5,9): 1 + 5 + 5*4 + 5*4^2 + 5*4^3 = 1+5+20+80+320 = 426
        # So n >= 426 (actually 428 per problem statement)
        # Known best: 1116
        target_ns = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1400, 1600, 2000, 2500]
        
        best_result = None
        for n in target_ns:
            remaining = total_time - (time.time() - start_time)
            if remaining < 15:
                break
            t_alloc = min(20.0, remaining * 0.4)
            res = _switching_construct(k, g, n, time_limit=t_alloc)
            if res is not None:
                best_result = res
                # If we found one, try to find smaller ones
                # But don't spend too much time
                if n <= 800:
                    break
                remaining2 = total_time - (time.time() - start_time)
                if remaining2 < 30:
                    break
        
        if best_result is not None:
            return best_result
        
        # Fallback
        return _luw_construct(k, g)
    else:
        return _luw_construct(k, g)
