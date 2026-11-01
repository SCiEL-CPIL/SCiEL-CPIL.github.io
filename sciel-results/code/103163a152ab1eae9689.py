import random
from collections import deque
import time

def construct(k, g):
    # For k=4, g=10, use random 4-regular graph + edge switching to increase girth
    if k != 4 or g != 10:
        return _fallback_construct(k, g)
    
    start_time = time.time()
    deadline = start_time + 115  # Leave 35s margin
    
    # Try multiple random 4-regular graphs on 320 vertices
    best_result = None
    best_girth_found = 0
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        n = 320
        # Generate random 4-regular graph using configuration model
        adj, edges = _random_4_regular(n)
        
        # Iteratively increase girth via 2-switches
        current_girth = _estimate_girth(adj, n)
        
        if current_girth >= g:
            return n, edges
        
        # Try to improve girth via local edge switches
        for switch_iter in range(500):
            if time.time() > deadline:
                break
            
            # Find a short cycle and try to break it
            short_cycle = _find_short_cycle(adj, n, g)
            if short_cycle is None:
                # No short cycle found, verify and return
                if _check_girth(adj, n, g):
                    return n, edges
                break
            
            # Try to break the cycle by swapping edges
            if not _try_break_cycle(adj, edges, short_cycle, n):
                # Random 2-switch
                if _random_2_switch(adj, edges, n):
                    continue
                else:
                    break
        
        # Check final girth
        if _check_girth(adj, n, g):
            return n, edges
    
    # Fallback: LUW graph
    return _fallback_construct(k, g)


def _random_4_regular(n):
    """Generate a random 4-regular graph using configuration model."""
    # Create stubs: each vertex has 4 stubs
    stubs = []
    for i in range(n):
        for _ in range(4):
            stubs.append(i)
    
    # Randomly pair up stubs
    random.shuffle(stubs)
    edges = []
    edge_set = set()
    adj = [[] for _ in range(n)]
    
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            continue  # Skip loops for now, will fix later
        key = (min(u, v), max(u, v))
        if key not in edge_set:
            edge_set.add(key)
            edges.append(key)
            adj[u].append(v)
            adj[v].append(u)
    
    # Fix any degree issues by swapping
    for vertex in range(n):
        while len(adj[vertex]) < 4:
            # Find a vertex with degree > 4 and an edge to swap
            fixed = False
            for other in range(n):
                if len(adj[other]) > 4 and other not in adj[vertex]:
                    # Find an edge (other, x) where x != vertex
                    for x in adj[other][:]:
                        if x != vertex and x not in adj[vertex]:
                            # Swap: remove (other, x), add (vertex, x) and (other, vertex)
                            adj[other].remove(x)
                            adj[x].remove(other)
                            adj[other].append(vertex)
                            adj[vertex].append(other)
                            adj[x].append(vertex)
                            adj[vertex].append(x)
                            # Update edges
                            edges = []
                            for u in range(n):
                                for v in adj[u]:
                                    if u < v:
                                        edges.append((u, v))
                            fixed = True
                            break
                    if fixed:
                        break
            if not fixed:
                # Give up on this vertex
                break
    
    # Ensure exactly 4-regular
    for vertex in range(n):
        while len(adj[vertex]) > 4:
            # Remove an edge
            to_remove = adj[vertex].pop()
            adj[to_remove].remove(vertex)
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
    
    return adj, edges


def _estimate_girth(adj, n):
    """Quick estimate of girth using BFS from a few vertices."""
    min_girth = float('inf')
    sample_vertices = random.sample(range(n), min(10, n))
    for start in sample_vertices:
        girth = _bfs_girth(adj, start, n, max_depth=12)
        if girth < min_girth:
            min_girth = girth
    return min_girth


def _bfs_girth(adj, start, n, max_depth):
    """Find shortest cycle through start vertex using BFS."""
    visited = {start: 0}
    parent = {start: -1}
    queue = deque([start])
    
    while queue:
        cur = queue.popleft()
        depth = visited[cur]
        if depth >= max_depth:
            continue
        for nb in adj[cur]:
            if nb == parent[cur]:
                continue
            if nb in visited:
                # Found a cycle
                cycle_len = depth + 1 + visited[nb]
                if cycle_len < max_depth:
                    return cycle_len
            else:
                visited[nb] = depth + 1
                parent[nb] = cur
                queue.append(nb)
    
    return max_depth


def _find_short_cycle(adj, n, target_girth):
    """Find a cycle shorter than target_girth, or None if none found."""
    for start in range(n):
        # Non-backtracking BFS
        visited = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((start, nb))
            visited[(start, nb)] = 1
        
        while queue:
            cur, prev = queue.popleft()
            depth = visited[(cur, prev)]
            if depth + 1 >= target_girth:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                new_state = (nb, cur)
                if new_state in visited:
                    d2 = visited[new_state]
                    if depth + 1 + d2 < target_girth:
                        # Found a short cycle, reconstruct it
                        return _reconstruct_cycle(adj, start, cur, prev, nb, visited, d2)
                else:
                    visited[new_state] = depth + 1
                    if depth + 1 < target_girth:
                        queue.append(new_state)
    return None


def _reconstruct_cycle(adj, start, cur, prev, nb, visited, d2):
    """Reconstruct the cycle vertices."""
    # This is a simplified reconstruction - just return the cycle as a list
    # For now, just return the edge that closes the cycle
    return [(prev, cur), (cur, nb), (nb, start)]  # Simplified


def _try_break_cycle(adj, edges, cycle, n):
    """Try to break a cycle by swapping edges."""
    if not cycle:
        return False
    
    # Get the vertices in the cycle
    cycle_vertices = set()
    for u, v in cycle:
        cycle_vertices.add(u)
        cycle_vertices.add(v)
    
    # Try to find an edge swap that breaks the cycle
    for u, v in cycle:
        for w in adj[u]:
            if w == v:
                continue
            if w in cycle_vertices:
                continue
            # Check if we can swap (u,v) with (u,w)
            # This would require finding another edge to complete the swap
            pass
    
    # For simplicity, just do a random 2-switch
    return _random_2_switch(adj, edges, n)


def _random_2_switch(adj, edges, n):
    """Perform a random 2-switch: pick two edges (a,b) and (c,d) and swap to (a,c) and (b,d)."""
    if len(edges) < 2:
        return False
    
    # Pick two random edges
    i, j = random.sample(range(len(edges)), 2)
    e1, e2 = edges[i], edges[j]
    a, b = e1
    c, d = e2
    
    # Check if swap is valid (no loops, no multi-edges)
    if a == c or a == d or b == c or b == d:
        return False
    
    # Check if new edges would create multi-edges
    if (min(a, c), max(a, c)) in set(edges) or (min(b, d), max(b, d)) in set(edges):
        return False
    
    # Perform the swap
    adj[a].remove(b)
    adj[b].remove(a)
    adj[c].remove(d)
    adj[d].remove(c)
    
    adj[a].append(c)
    adj[c].append(a)
    adj[b].append(d)
    adj[d].append(b)
    
    # Update edges
    edges[i] = (min(a, c), max(a, c))
    edges[j] = (min(b, d), max(b, d))
    
    return True


def _check_girth(adj, n, girth):
    """Check if graph has no cycle shorter than girth using non-backtracking BFS."""
    for start in range(n):
        # State: (current, prev) -> depth
        visited = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((start, nb))
            visited[(start, nb)] = 1
        
        while queue:
            cur, prev = queue.popleft()
            depth = visited[(cur, prev)]
            if depth + 1 >= girth:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                new_state = (nb, cur)
                if new_state in visited:
                    d2 = visited[new_state]
                    if depth + 1 + d2 < girth:
                        return False
                else:
                    visited[new_state] = depth + 1
                    if depth + 1 < girth:
                        queue.append(new_state)
    return True


def _fallback_construct(k, g):
    """Fallback to LUW-style construction."""
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
