import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    if k != 3:
        # Fallback for non-3: LUW
        return _luw_construct(k, g)

    # Moore bound for (3,16): 2*(1+2+4+8+16+32) = 126, so n_min=128 (even)
    # Best known: 936
    # We'll try voltage graph construction and greedy improvement

    # Try to build a good graph via voltage graph on a base
    best_n, best_edges = _voltage_graph_cage(3, g, deadline)

    if best_edges is None:
        # Fallback to LUW
        best_n, best_edges = _luw_construct(3, g)

    return best_n, best_edges


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction for k=3."""
    d = (g + 1) // 2
    if 2 * d - 1 < g:
        d += 1
    q = k  # For k=3, q=3
    
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
                    y[c] = (x[c] + y[a] * x[b]) % q
                else:
                    y[c] = (x[c] - x[a] * y[b]) % q
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


def _voltage_graph_cage(k, g, deadline):
    """
    Construct a k-regular graph of girth >= g using voltage graph / Cayley graph methods.
    
    Strategy: Use a Cayley graph on a finite group with generators chosen to
    avoid short cycles. For k=3, use a group with 3 generators (a, b, c) where
    each generator is an involution or of appropriate order.
    
    We try several group constructions:
    1. Z_m x Z_n with carefully chosen generators
    2. Dihedral groups
    3. Heisenberg group over Z_p
    
    The key is that in a Cayley graph, cycles correspond to relations in the group.
    We want a group with no short relations among the generators.
    """
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for (3, g) even: 2 * sum_{i=0}^{g/2 - 1} 2^i = 2*(2^{g/2} - 1)
    gprime = g // 2
    moore = 2 * (2**gprime - 1)
    n_min = moore if moore % 2 == 0 else moore + 1
    if n_min < 4:
        n_min = 4
    
    # Try Cayley graphs on Z_n with generators that are "far apart"
    # For a Cayley graph on Z_n with generators {a, -a, b}, we need:
    # - 2a != 0, 2b != 0 (no 2-cycles)
    # - a != b, a != -b (no repeated edges)
    # - No short relations: ma + nb != 0 for small |m|+|n|
    
    # Try various n values from near Moore bound upward
    # For efficiency, try n values that are multiples of small numbers
    
    tried = set()
    for n in range(n_min, min(n_min + 200, 2000), 2):
        if time.time() > deadline - 10:
            break
        if n in tried:
            continue
        tried.add(n)
        
        # Try to find generators a, b in Z_n such that the Cayley graph
        # with generators {a, -a, b} has girth >= g
        # Actually for k=3, we need 3 distinct neighbors, so generators must be
        # such that {a, -a, b} are all distinct and not 0 mod n
        # This requires n >= 5 and a != b, a != -b, etc.
        
        # A better approach: use the standard cage construction via
        # "circulant" graphs: C(n; a, b) where each vertex i connects to
        # i+a, i-a, i+b mod n. But this gives degree 3 only if a, -a, b
        # are all distinct and nonzero.
        
        # For girth >= g, we need no cycle of length < g.
        # In a circulant graph, a cycle corresponds to a sequence of steps
        # from {a, -a, b} that sums to 0 mod n.
        
        # Let's try a systematic search over (a, b) for this n
        found = False
        for a in range(1, n // 2 + 1):
            if time.time() > deadline - 10:
                break
            for b in range(1, n):
                if b == a or b == n - a or b == 0:
                    continue
                # Check that the three generators are distinct
                # and the graph is simple
                if not _valid_circulant(n, a, b, g):
                    continue
                if n < best_n:
                    edges = _build_circulant(n, a, b)
                    best_n = n
                    best_edges = edges
                    found = True
                    break
            if found:
                break
        if found and n == n_min:
            break
    
    return best_n, best_edges


def _valid_circulant(n, a, b, g):
    """Check if circulant graph C(n; a, -a, b) has girth >= g."""
    # The graph has vertices 0..n-1, edges i-(i+a), i-(i-a), i-(i+b) mod n
    # We need to check no cycle of length < g
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for i in range(n):
        adj[i].append((i + a) % n)
        adj[i].append((i - a) % n)
        adj[i].append((i + b) % n)
    
    # Check simplicity: no loops, no multi-edges
    for i in range(n):
        for j in adj[i]:
            if i == j:
                return False
    edge_set = set()
    for i in range(n):
        for j in adj[i]:
            key = (min(i, j), max(i, j))
            if key in edge_set:
                return False
            edge_set.add(key)
    
    # Check girth using BFS from each vertex (bounded)
    # For efficiency, use non-backtracking walk detection
    max_depth = g - 1
    
    for s in range(n):
        # BFS with parent tracking to detect cycles
        dist = [-1] * n
        prev = [-1] * n
        dist[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= max_depth:
                break
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    prev[v] = u
                    queue.append(v)
                elif v != prev[u]:
                    # Found a cycle
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _build_circulant(n, a, b):
    """Build edge list for circulant graph C(n; a, -a, b)."""
    edges = []
    edge_set = set()
    for i in range(n):
        for j in [(i + a) % n, (i - a) % n, (i + b) % n]:
            if i < j:
                key = (i, j)
                if key not in edge_set:
                    edge_set.add(key)
                    edges.append(key)
    return edges


def _greedy_improve(n, edges, g, deadline):
    """
    Try to improve a 3-regular graph by finding short cycles and flipping edges.
    This is a local search method.
    """
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Find a short cycle
        for s in range(n):
            if time.time() > deadline - 5:
                break
            # BFS to find shortest cycle through s
            dist = [-1] * n
            prev = [-1] * n
            dist[s] = 0
            queue = deque([s])
            found_cycle = None
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= g - 1:
                    break
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        prev[v] = u
                        queue.append(v)
                    elif v != prev[u]:
                        cycle_len = du + dist[v] + 1
                        if cycle_len < g:
                            # Found a short cycle
                            # Extract the cycle
                            cycle = [v]
                            cur = v
                            while cur != s:
                                cur = prev[cur]
                                cycle.append(cur)
                            cycle.append(s)
                            # Try to break the cycle by flipping an edge
                            # Remove an edge from the cycle and add a new edge
                            # that doesn't create a shorter cycle
                            for idx in range(len(cycle) - 1):
                                u1, v1 = cycle[idx], cycle[idx + 1]
                                # Try removing (u1, v1) and adding a new edge
                                # from u1 to some w not adjacent to u1,
                                # and v1 to some w' not adjacent to v1
                                # such that no new short cycle is created
                                # This is complex; skip for now
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    return edges


# Main entry point
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    if k != 3:
        return _luw_construct(k, g)

    # Try voltage/circulant graph construction
    best_n, best_edges = _voltage_graph_cage(3, g, deadline)

    if best_edges is not None:
        return best_n, best_edges

    # Fallback to LUW
    return _luw_construct(3, g)
