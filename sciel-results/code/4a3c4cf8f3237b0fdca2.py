"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    For k=3, g=16: use random 3-regular graph + iterative cycle-breaking repair.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    if k != 3:
        # Fallback for other k: use the original L-U-W construction
        return _luw_construct(k, g)
    
    # Target: try to find a 3-regular graph with girth >= g
    # Start with a reasonable size and grow if needed
    # Moore bound for (3,16) is 512; best known is 936
    # Try sizes that are even (required for 3-regular)
    
    best_result = None
    best_n = float('inf')
    
    # Try different starting sizes
    candidate_sizes = []
    # Start near known good range
    for n in range(600, 1200, 2):
        candidate_sizes.append(n)
    
    random.shuffle(candidate_sizes)
    
    for n in candidate_sizes:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations for this size
        attempts = 3
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random 3-regular graph
            graph = _random_3_regular(n)
            if graph is None:
                continue
            
            # Repair: break short cycles
            repaired = _repair_girth(graph, n, g, deadline)
            
            if repaired is not None:
                edges = _graph_to_edges(repaired)
                if n < best_n:
                    best_n = n
                    best_result = (n, edges)
                    if n <= 936:  # Match or beat best known
                        break
        
        if best_result is not None and best_n <= 936:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: use L-U-W construction
    return _luw_construct(3, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction (original)."""
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


def _random_3_regular(n):
    """Generate a random 3-regular graph on n vertices (n must be even)."""
    if n < 4 or n % 2 != 0:
        return None
    
    # Use configuration model with rejection
    # Each vertex has 3 stubs
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    
    # Try multiple random pairings
    for _ in range(100):
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            edge = (min(u,v), max(u,v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        
        if valid:
            # Build adjacency
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            # Verify all have degree 3
            if all(len(adj[i]) == 3 for i in range(n)):
                return adj
    return None


def _find_short_cycle(adj, n, max_len):
    """Find a cycle of length <= max_len using BFS from each vertex."""
    for start in range(n):
        # BFS tracking paths
        # visited[v] = (parent, depth)
        visited = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((start, nb, 1))
            visited[(start, nb)] = (start, 1)
        
        while queue:
            prev, curr, depth = queue.popleft()
            if depth >= max_len:
                continue
            for nb in adj[curr]:
                if nb == prev:
                    continue
                if nb == start:
                    return depth + 1  # found cycle of length depth+1
                if (prev, nb) not in visited:
                    visited[(prev, nb)] = (curr, depth + 1)
                    queue.append((curr, nb, depth + 1))
    return None


def _repair_girth(adj, n, target_girth, deadline):
    """Iteratively break short cycles until girth >= target_girth."""
    adj = [list(x) for x in adj]  # copy
    max_cycle_len = target_girth - 1  # we want no cycles shorter than target_girth
    
    iterations = 0
    while time.time() < deadline - 5:
        iterations += 1
        # Find a short cycle
        cycle_len = _find_short_cycle(adj, n, max_cycle_len)
        
        if cycle_len is None:
            # Verify girth
            actual_girth = _compute_girth(adj, n)
            if actual_girth >= target_girth:
                return adj
            # Else, there's a shorter cycle we missed
            continue
        
        # Try to break the cycle by edge switching
        # Find the actual cycle
        cycle = _find_cycle_path(adj, n, max_cycle_len)
        if cycle is None:
            break
        
        # Try edge switches to break this cycle
        success = False
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            # Try to swap: remove (u,v), add (u, x), (v, y) where x,y are non-neighbors
            # and this doesn't create a shorter cycle
            for x in adj[u]:
                if x == v:
                    continue
                for y in adj[v]:
                    if y == u:
                        continue
                    if x == y:
                        continue
                    # Check if (u,x) and (v,y) would be valid new edges
                    if x in adj[v] or y in adj[u]:
                        continue
                    # Try the switch
                    new_adj = _try_edge_switch(adj, u, v, x, y)
                    if new_adj is not None:
                        # Check if this helps (doesn't create shorter cycle)
                        new_girth = _compute_girth_fast(new_adj, n, max_cycle_len)
                        if new_girth is not None and new_girth >= cycle_len:
                            adj = new_adj
                            success = True
                            break
            if success:
                break
        
        if not success:
            # Try random switches
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                # Pick a random edge
                u = random.randint(0, n-1)
                v = random.choice(adj[u])
                if v < u:
                    u, v = v, u
                # Find other neighbors
                other_u = [x for x in adj[u] if x != v]
                other_v = [x for x in adj[v] if x != u]
                if not other_u or not other_v:
                    continue
                x = random.choice(other_u)
                y = random.choice(other_v)
                if x == y or x in adj[v] or y in adj[u]:
                    continue
                new_adj = _try_edge_switch(adj, u, v, x, y)
                if new_adj is not None:
                    adj = new_adj
                    break
    
    # Final verification
    if time.time() < deadline - 2:
        final_girth = _compute_girth(adj, n)
        if final_girth >= target_girth:
            return adj
    
    return None


def _try_edge_switch(adj, u, v, x, y):
    """Try switching edge (u,v) to (u,x) and (v,y). Returns new adj or None."""
    # Check validity
    if x == y:
        return None
    if x in adj[v]:
        return None
    if y in adj[u]:
        return None
    
    new_adj = [list(a) for a in adj]
    
    # Remove u from v's adj and v from u's adj
    new_adj[v].remove(u)
    new_adj[u].remove(v)
    
    # Add x to u's adj and u to x's adj
    new_adj[u].append(x)
    new_adj[x].append(u)
    
    # Add y to v's adj and v to y's adj
    new_adj[v].append(y)
    new_adj[y].append(v)
    
    # Verify degrees
    for i in range(len(new_adj)):
        if len(new_adj[i]) != 3:
            return None
        if len(set(new_adj[i])) != 3:
            return None
    
    return new_adj


def _find_cycle_path(adj, n, max_len):
    """Find the actual path of a short cycle."""
    for start in range(n):
        visited = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((start, [start, nb], 1))
        
        while queue:
            prev, path, depth = queue.popleft()
            if depth >= max_len:
                continue
            for nb in adj[prev]:
                if nb == path[-2] if len(path) >= 2 else False:
                    continue
                if nb == start and depth + 1 <= max_len:
                    return path  # found cycle
                # Check if we've been here
                key = (prev, nb)
                if key not in visited:
                    visited[key] = True
                    queue.append((nb, path + [nb], depth + 1))
    return None


def _compute_girth(adj, n):
    """Compute the girth (shortest cycle) of the graph."""
    min_girth = float('inf')
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            curr = queue.popleft()
            for nb in adj[curr]:
                if dist[nb] == -1:
                    dist[nb] = dist[curr] + 1
                    parent[nb] = curr
                    queue.append(nb)
                elif nb != parent[curr]:
                    # Found a cycle
                    cycle_len = dist[curr] + dist[nb] + 1
                    min_girth = min(min_girth, cycle_len)
    
    return min_girth if min_girth != float('inf') else n


def _compute_girth_fast(adj, n, max_check):
    """Quick check if girth >= max_check+1 (i.e., no cycle <= max_check)."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            curr = queue.popleft()
            if dist[curr] >= max_check:
                continue
            for nb in adj[curr]:
                if dist[nb] == -1:
                    dist[nb] = dist[curr] + 1
                    parent[nb] = curr
                    queue.append(nb)
                elif nb != parent[curr]:
                    cycle_len = dist[curr] + dist[nb] + 1
                    if cycle_len <= max_check:
                        return cycle_len
    
    return max_check + 1  # No short cycle found


def _graph_to_edges(adj):
    """Convert adjacency list to edge list (no duplicates)."""
    edges = []
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges
# EVOLVE-BLOCK-END
