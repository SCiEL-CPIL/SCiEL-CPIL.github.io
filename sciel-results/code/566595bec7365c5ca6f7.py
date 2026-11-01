import time
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g.
    For k=3, g=16, we use random 3-regular graph generation with local search.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    if k != 3:
        # Fallback for other k (should not be hit for this task)
        return _luw_construct(k, g)

    # Moore bound for k=3, g=16 is 512.
    # We'll try n from 512 upward in steps of 2 (must be even for 3-regular).
    
    random.seed(42)
    
    best_result = None
    best_n = float('inf')
    
    for n in range(512, 2000, 2):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 5 if n < 700 else 3
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random 3-regular graph
            edges = _generate_random_3reg(n)
            if edges is None:
                continue
            
            # Check if girth is already >= g
            if _verify_girth_fast(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_result = (n, edges)
                # Found valid graph at this n, try smaller n? No, we're going up.
                # Return immediately since we want minimum n.
                return n, edges
            
            # Try local search to improve girth
            improved_edges = _local_search_girth(n, edges, g, deadline)
            if improved_edges is not None:
                if _verify_girth_fast(n, improved_edges, g):
                    if n < best_n:
                        best_n = n
                        best_result = (n, improved_edges)
                    return n, improved_edges
        
        # If we found a valid graph at a smaller n, we would have returned already.
        # Continue to larger n.
    
    # If not found, return the best we have or a fallback
    if best_result is not None:
        return best_result
    
    # Fallback: LUW construction
    return _luw_construct(3, g)


def _generate_random_3reg(n):
    """Generate a random 3-regular graph on n vertices using configuration model."""
    if n < 4 or n % 2 != 0:
        return None
    
    # Create stubs: each vertex has 3 stubs
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    # Pair up stubs to form edges
    edges = []
    edge_set = set()
    
    for i in range(0, n * 3, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            # Self-loop, invalid for simple graph
            return None
        # Check for duplicate edge
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            # Duplicate edge, invalid
            return None
        edge_set.add((u, v))
        edges.append((u, v))
    
    return edges


def _local_search_girth(n, edges, g, deadline):
    """
    Local search to improve girth: find short cycles and swap edges to break them.
    Returns improved edge list or None if no improvement found within deadline.
    """
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find all cycles of length < g
    short_cycles = _find_short_cycles(n, adj, g)
    
    if not short_cycles:
        return edges  # Already good
    
    # Try to break cycles by edge swaps
    current_edges = list(edges)
    current_adj = [list(neighbors) for neighbors in adj]
    
    max_swaps = 100
    for _ in range(max_swaps):
        if time.time() > deadline - 5:  # Leave 5s safety margin
            break
        
        # Find a short cycle
        short_cycles = _find_short_cycles(n, current_adj, g)
        if not short_cycles:
            break
        
        cycle = random.choice(short_cycles)
        
        # Try to swap an edge in the cycle with a non-adjacent edge to break it
        swapped = False
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            
            # Find a non-adjacent edge to swap with
            # We want to remove (u,v) and add (u,w) and (x,v) where w and x are non-adjacent to u and v respectively
            # And (u,w) and (x,v) don't already exist
            
            # Simple approach: try random swaps
            for _ in range(20):
                # Pick random vertex w not adjacent to u
                w_candidates = [x for x in range(n) if x != u and x not in current_adj[u]]
                if not w_candidates:
                    continue
                w = random.choice(w_candidates)
                
                # Pick random vertex x not adjacent to v
                x_candidates = [x for x in range(n) if x != v and x not in current_adj[v] and x != w]
                if not x_candidates:
                    continue
                x = random.choice(x_candidates)
                
                # Check if (u,w) and (x,v) would create duplicates or self-loops
                if w in current_adj[u] or x in current_adj[v]:
                    continue
                if (min(u, w), max(u, w)) in {(min(a, b), max(a, b)) for a, b in current_edges}:
                    continue
                if (min(x, v), max(x, v)) in {(min(a, b), max(a, b)) for a, b in current_edges}:
                    continue
                
                # Perform swap
                # Remove (u,v)
                edge_set = set((min(a, b), max(a, b)) for a, b in current_edges)
                if (min(u, v), max(u, v)) in edge_set:
                    current_edges.remove((u, v)) if (u, v) in current_edges else current_edges.remove((v, u))
                    current_adj[u].remove(v)
                    current_adj[v].remove(u)
                    
                    # Add (u,w) and (x,v)
                    current_edges.append((u, w))
                    current_edges.append((x, v))
                    current_adj[u].append(w)
                    current_adj[w].append(u)
                    current_adj[x].append(v)
                    current_adj[v].append(x)
                    
                    swapped = True
                    break
            
            if swapped:
                break
        
        if not swapped:
            break
    
    return current_edges


def _find_short_cycles(n, adj, max_len):
    """Find all cycles of length < max_len using BFS."""
    cycles = []
    visited_global = [0] * n
    mark = 0
    
    for start in range(n):
        mark += 1
        visited = [-1] * n
        visited[start] = 0
        parent = [-1] * n
        queue = deque([(start, 0)])
        
        while queue:
            node, depth = queue.popleft()
            if depth >= max_len - 1:
                continue
            
            for neighbor in adj[node]:
                if neighbor == parent[node]:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    parent[neighbor] = node
                    queue.append((neighbor, depth + 1))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < max_len:
                        # Reconstruct cycle
                        cycle = [node]
                        cur = node
                        while cur != neighbor:
                            cur = parent[cur]
                            cycle.append(cur)
                        cycle.reverse()
                        cycles.append(cycle)
    
    return cycles


def _verify_girth_fast(n, edges, g):
    """Verify girth >= g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    dist = [0] * n
    parent = [-1] * n
    seen_mark = [0] * n
    mark = 0
    
    for s in range(n):
        mark += 1
        dist[s] = 0
        parent[s] = -1
        seen_mark[s] = mark
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= g // 2:
                continue
            for v in adj[u]:
                if seen_mark[v] != mark:
                    dist[v] = du + 1
                    parent[v] = u
                    seen_mark[v] = mark
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
                        
    return True


def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for k-regular graphs.
    We try smaller d values to get fewer vertices, checking girth empirically.
    """
    if k != 3:
        q = k
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

    # For k=3, we use q=3.
    q = 3

    def build_luw(d):
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

    # Try d=8 first. If girth < 16, try d=9, etc.
    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    # If all fail (unlikely for d=11), return the largest one (which should be valid by theory)
    return build_luw(11)


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_girth_fast(n, edges, min_girth):
    """Fast girth check using BFS from each vertex with parent tracking."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])

        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True
