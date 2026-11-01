"""Voltage graph lift approach for cage problem n(4,10)."""
import numpy as np
from itertools import product
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth >= g using voltage graph lifting.
    
    Strategy: 
    1. Try to find a circulant graph (voltage lift of K1) with n=320, k=4, girth>=10.
    2. If not found, use a voltage lift of K4,4 over Z40 with a heuristic search.
    """
    if k != 4 or g != 10:
        # Fallback to LUW for other parameters
        return _luw_construct(k, g)
    
    n_target = 320
    
    # Attempt 1: Circulant Graph C_320(a, b)
    # A circulant graph is a voltage lift of a single vertex.
    # We search for generators a, b such that the girth is at least g.
    found_graph = _find_circulant_cage(n_target, k, g)
    if found_graph:
        return found_graph
        
    # Attempt 2: Voltage lift of K4,4 over Z40
    # This is more complex. We use a deterministic "good" assignment if known,
    # or a fast local search.
    # For now, we will try a specific known construction pattern.
    # The 320-vertex cage is often a lift of K4,4.
    # We will use a simple heuristic search for voltages.
    
    return _voltage_lift_k44(n_target, k, g)


def _find_circulant_cage(n, k, g):
    """Search for a circulant graph C_n(S) with degree k and girth >= g."""
    # For k=4, S = {a, -a, b, -b}
    # We need a, b in 1..n//2
    # Ensure a != b, a != -b, etc.
    
    best_girth = 0
    best_pair = None
    
    # Optimization: only check a <= b
    for a in range(1, n // 2 + 1):
        for b in range(a, n // 2 + 1):
            # Check distinctness
            if a == b or a == n - b:
                continue
            # Also need a != -a (n even, a=n/2 is a loop? No, a=n/2 means edge to n/2. 
            # In C_n, vertex i connects to i+a. If a=n/2, i connects to i+n/2.
            # This is a valid edge. But we need 4 distinct neighbors.
            # Neighbors of 0: a, n-a, b, n-b.
            # Distinct if a != b, a != n-b, n-a != b, n-a != n-b.
            # a != n-b is a+b != n.
            # n-a != b is a+b != n.
            # n-a != n-b is a != b.
            if a + b == n:
                continue
            if a == n - a: # a = n/2
                # Then neighbors are a, a, b, n-b. Only 3 distinct if b != a.
                # Actually if a=n/2, edge (0, n/2) is a single edge.
                # Degree is still 4? No, (0, n/2) is one edge.
                # The set of neighbors is {a, b, n-b}. Size 3.
                # So a cannot be n/2 for k=4 simple graph.
                continue
                
            # Compute girth
            girth = _compute_girth_circulant(n, [a, n-a, b, n-b], g)
            if girth >= g:
                # Found a valid graph
                edges = []
                for i in range(n):
                    for s in [a, b]:
                        # Add edge (i, i+s) and (i, i-s)
                        # To avoid duplicates, only add if i < j
                        j1 = (i + s) % n
                        j2 = (i - s) % n
                        if i < j1:
                            edges.append((i, j1))
                        if i < j2:
                            edges.append((i, j2))
                return n, edges
            if girth > best_girth:
                best_girth = girth
                best_pair = (a, b)
                
    return None


def _compute_girth_circulant(n, generators, limit):
    """Compute girth of circulant graph using BFS from 0."""
    # Generators are the steps from 0
    # We want the shortest non-empty path back to 0
    # But we must avoid backtracking (non-backtracking walks)
    # Girth is the length of the shortest cycle.
    
    # BFS state: (current_node, previous_node)
    # Start: (0, None)
    # But for girth, we can just do BFS on the graph and find the first
    # time we return to 0 via a different path?
    # Standard girth algorithm: BFS from each node.
    # For circulant, BFS from 0 is enough? No, cycles can be far from 0.
    # However, for highly symmetric graphs, BFS from 0 often finds the girth.
    # Let's do BFS from 0 and check for cycles.
    
    # Actually, a simpler check for circulant graphs:
    # The girth is the minimum L such that there exist x, y with 
    # x*a + y*b = 0 mod n and |x|+|y|=L, with the path being non-backtracking.
    # Non-backtracking means we don't immediately reverse direction.
    
    # Let's use a BFS on the state space (node, prev_node)
    # visited[node][prev_node] = distance
    # This is O(n^2) which is 320^2 = 102400. Fast.
    
    from collections import deque
    q = deque()
    # State: (node, prev)
    # We start from 0 with prev=None. 
    # To avoid trivial backtracking, we don't allow returning to prev.
    
    # Initialize
    for s in generators:
        q.append((s, 0)) # Move from 0 to s, prev is 0
        # Distance 1
    
    visited = {}
    for s in generators:
        visited[(s, 0)] = 1
        
    min_girth = float('inf')
    
    while q:
        node, prev = q.popleft()
        dist = visited[(node, prev)]
        
        if node == 0 and dist >= 3: # Cycle found (length dist)
             # We must ensure it's a simple cycle.
             # BFS on (node, prev) ensures non-backtracking.
             # If we reach 0, we have a cycle.
             # The length is dist.
             if dist < min_girth:
                 min_girth = dist
             # We can stop expanding this path if we only care about min
             # But we need to check if there's a shorter one?
             # BFS guarantees shortest path to state (node, prev).
             # So the first time we hit (0, prev) with dist>=3 is the girth?
             # Not necessarily, because prev matters.
             # But any cycle back to 0 is a valid cycle.
             # We can break if we want just the min.
             # However, we should continue to ensure we don't miss a shorter one?
             # BFS processes in increasing distance. So the first hit is the min.
             # But we have multiple states for 0.
             # We can return immediately?
             # Yes, BFS expands by distance. The first time we pop a state with node==0 (and dist>1)
             # is the shortest cycle length?
             # Wait, we started at distance 1.
             # If we pop (0, s) at distance L, it means there is a path 0->...->s->0 of length L.
             # This is a cycle of length L.
             # Since BFS is level-order, the first such L is the girth.
             
             # One caveat: The path must be simple.
             # Non-backtracking BFS on (node, prev) finds simple cycles?
             # Yes, because it prevents immediate reversal.
             # Does it prevent other repeats?
             # A non-backtracking walk that repeats a vertex other than the start is a closed walk
             # that contains a cycle. The shortest such is the girth.
             # So yes, the first return to 0 in non-backtracking BFS is the girth.
             
             return min_girth if min_girth < limit else min_girth
             
        for s in generators:
            next_node = (node + s) % n
            # Non-backtracking: next_node != prev
            if next_node == prev:
                continue
            state = (next_node, node)
            if state not in visited:
                visited[state] = dist + 1
                q.append(state)
                
    return min_girth if min_girth < float('inf') else n # Fallback


def _voltage_lift_k44(n, k, g):
    """Fallback: Voltage lift of K4,4."""
    # This is a placeholder for a more complex search.
    # If circulant fails, we might return LUW with d=5 as last resort.
    return _luw_construct(k, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) construction."""
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


# EVOLVE-BLOCK-END
