"""Construct small 5-regular graphs of high girth using random construction + local search."""
import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # For the specific case k=5, g=9, use a targeted approach
    # Moore bound for n(5,9) is 428; known best is ~1116
    # Try to find a good graph via random construction + girth improvement
    
    if k == 5 and g == 9:
        return _construct_5_9(deadline)
    
    # Fallback: use LUW construction for other parameters
    return _luw_construct(k, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
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


def _construct_5_9(deadline):
    """
    Construct a 5-regular graph of girth at least 9.
    
    Strategy: 
    1. Try several random 5-regular graphs of varying sizes
    2. For each, use local search to eliminate cycles of length < 9
    3. Return the smallest valid graph found
    """
    best_n = float('inf')
    best_edges = None
    rng = random.Random(42)
    
    # Candidate sizes to try (must be even for 5-regular)
    # Moore bound is 428, known constructions are ~1000+
    # Try a range of sizes
    candidate_sizes = [500, 520, 540, 560, 580, 600, 640, 680, 720, 760, 800]
    
    for n in candidate_sizes:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random graphs of this size
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            
            edges = _random_5_regular(n, rng)
            if edges is None:
                continue
            
            # Try to improve girth
            improved_edges = _improve_girth(edges, n, target_girth=9, deadline=deadline, rng=rng)
            
            if improved_edges is not None:
                if n < best_n:
                    best_n = n
                    best_edges = improved_edges
                    print(f"Found graph with n={n}, girth>=9", file=sys.stderr)
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback: use LUW
    return _luw_construct(5, 9)


def _random_5_regular(n, rng):
    """
    Generate a random 5-regular graph on n vertices.
    Uses the configuration model (pairing stubs).
    Returns list of edges or None if not simple.
    """
    if n < 6 or n % 2 != 0:
        return None
    
    # Create stubs: each vertex has 5 stubs
    stubs = []
    for v in range(n):
        for _ in range(5):
            stubs.append(v)
    
    # Randomly pair up stubs
    while True:
        rng.shuffle(stubs)
        edges_set = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edges_set:
                valid = False
                break
            edges_set.add(edge)
        
        if valid:
            return list(edges_set)
        
        # Retry (configuration model may take a few tries for small n)
        # For large n, it usually succeeds on first try
        if n > 100:
            break
    
    return None


def _get_adjacency(edges, n):
    """Build adjacency list from edge list."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_short_cycle(adj, n, max_len):
    """
    Find a cycle of length <= max_len using BFS from each vertex.
    Returns the cycle as a list of vertices, or None.
    """
    # For efficiency, only check cycles up to max_len
    # Use BFS with parent tracking
    for start in range(n):
        # BFS from start, tracking paths
        # We want to find a cycle back to start of length <= max_len
        # Use BFS layer by layer
        
        # dist[v] = shortest distance from start
        # parent[v] = previous vertex on shortest path
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        found = False
        while queue and not found:
            u = queue.popleft()
            if dist[u] >= max_len:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v == start and dist[u] + 1 <= max_len and dist[u] + 1 >= 3:
                    # Found a cycle back to start
                    # Reconstruct path
                    cycle = [start]
                    cur = u
                    while cur != start:
                        cycle.append(cur)
                        cur = parent[cur]
                    cycle.append(start)
                    return cycle
                elif v != start and dist[v] <= dist[u]:
                    # Potential cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if 3 <= cycle_len <= max_len:
                        # Reconstruct two paths
                        path1 = [u]
                        cur = u
                        while cur != start:
                            cur = parent[cur]
                            path1.append(cur)
                        path1.reverse()
                        
                        path2 = [v]
                        cur = v
                        while cur != start:
                            cur = parent[cur]
                            path2.append(cur)
                        path2.reverse()
                        
                        # Combine: path1 + reversed path2 (excluding start)
                        cycle = path1 + path2[1:]
                        return cycle
        
    return None


def _improve_girth(edges, n, target_girth, deadline, rng):
    """
    Iteratively find and break short cycles using edge switching.
    Returns improved edge list or None if failed.
    """
    adj = _get_adjacency(edges, n)
    current_girth = _estimate_girth(adj, n, target_girth)
    
    if current_girth >= target_girth:
        return edges
    
    max_iterations = 500
    for iteration in range(max_iterations):
        if time.time() > deadline - 3:
            break
        
        # Find a short cycle
        cycle = _find_short_cycle(adj, n, target_girth - 1)
        if cycle is None:
            # No more short cycles
            return list(set(tuple(sorted(e)) for e in edges))
        
        # Remove one edge from the cycle and add a new edge to maintain regularity
        # Strategy: pick an edge in the cycle, remove it, then find a replacement
        # that maintains 5-regularity
        
        # Get unique vertices in cycle
        cycle_vertices = list(set(cycle))
        
        # Try removing different edges from the cycle
        success = False
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i + 1]
            if (u, v) not in edges and (v, u) not in edges:
                continue
            
            # Check if this edge is in our edge list
            edge_found = False
            for idx, (a, b) in enumerate(edges):
                if (a == u and b == v) or (a == v and b == u):
                    edge_found = True
                    break
            
            if not edge_found:
                continue
            
            # Remove this edge
            new_edges = [e for e in edges if not ((e[0] == u and e[1] == v) or (e[0] == v and e[1] == u))]
            
            # Now u and v each have degree 4
            # Need to add an edge between some x (neighbor of u, not v) and y (neighbor of v, not u)
            # such that x != y and (x,y) is not already an edge
            
            # Get neighbors of u (excluding v) and neighbors of v (excluding u)
            neigh_u = [w for w in adj[u] if w != v]
            neigh_v = [w for w in adj[v] if w != u]
            
            # We need to add edge (x, y) where x is a neighbor of u that we "free up"
            # and y is a neighbor of v. Actually, we need to do a 2-switch:
            # Remove (u,v) and (x,y), add (u,x) and (v,y) or similar
            
            # Simpler: just add a random edge between a neighbor of u and a neighbor of v
            # that doesn't create a loop or duplicate
            
            candidates = []
            for x in neigh_u:
                for y in neigh_v:
                    if x == y:
                        continue
                    if (x, y) in adj[x]:
                        continue
                    if (x, y) in [(a, b) for a, b in new_edges] or (y, x) in [(a, b) for a, b in new_edges]:
                        continue
                    candidates.append((x, y))
            
            if candidates:
                x, y = rng.choice(candidates)
                new_edges.append((min(x, y), max(x, y)))
                
                # Verify it's still 5-regular
                deg = [0] * n
                for a, b in new_edges:
                    deg[a] += 1
                    deg[b] += 1
                if all(d == 5 for d in deg):
                    # Check no duplicate edges
                    edge_set = set(tuple(sorted(e)) for e in new_edges)
                    if len(edge_set) == len(new_edges):
                        # Update adjacency
                        adj = _get_adjacency(new_edges, n)
                        edges = new_edges
                        success = True
                        break
        
        if not success:
            # Try a different approach: 2-switch
            # Find two edges (a,b) and (c,d) where we can swap to (a,c) and (b,d)
            # This is complex; for now, just break
            break
    
    # Final verification
    adj = _get_adjacency(edges, n)
    deg = [len(adj[i]) for i in range(n)]
    if not all(d == 5 for d in deg):
        return None
    
    # Check girth
    girth = _estimate_girth(adj, n, target_girth)
    if girth >= target_girth:
        return edges
    
    return None


def _estimate_girth(adj, n, upper_bound):
    """
    Estimate the girth by checking for cycles of length 3 to upper_bound.
    Returns the minimum cycle length found, or upper_bound if none found.
    """
    for length in range(3, upper_bound + 1):
        if _has_cycle_of_length(adj, n, length):
            return length
    return upper_bound


def _has_cycle_of_length(adj, n, length):
    """Check if there's a cycle of exactly 'length' using BFS."""
    for start in range(n):
        # BFS up to depth length-1
        # dist[v] = distance from start
        # parent[v] = previous
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= length - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v == start and dist[u] + 1 == length:
                    return True
                elif v != start and dist[v] <= dist[u] and dist[u] + dist[v] + 1 == length:
                    return True
    return False
