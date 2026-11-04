import time
import random
from collections import defaultdict

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses a voltage graph construction over K4 for k=3.
    Falls back to LUW if search fails or time runs out.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    if k != 3:
        # Fallback to LUW for other k if needed, though task specifies k=3
        return _construct_luw(k, g)

    # Base graph K4
    # Vertices 0, 1, 2, 3
    # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # Map edges to indices 0..5
    edge_map = {
        (0, 1): 0, (1, 0): 0,
        (0, 2): 1, (2, 0): 1,
        (0, 3): 2, (3, 0): 2,
        (1, 2): 3, (2, 1): 3,
        (1, 3): 4, (3, 1): 4,
        (2, 3): 5, (3, 2): 5
    }
    
    # Generate reduced closed walks of length 2 to g-1 in K4
    # A reduced walk does not immediately backtrack.
    # We represent a walk by (current_vertex, prev_edge_idx)
    # We want to find walks that start and end at the same vertex.
    
    target_girth = g
    max_walk_len = target_girth - 1
    
    # Precompute coefficient vectors for all reduced closed walks of length < g
    # c is a tuple of 6 integers
    coeff_vectors = set()
    
    # DFS from each start vertex
    for start_v in range(4):
        # State: (current_v, prev_edge, length, edge_counts)
        # We use a stack for DFS
        stack = [(start_v, -1, 0, [0]*6)]
        
        while stack:
            v, prev_e, l, counts = stack.pop()
            
            if l > 0 and v == start_v:
                # Closed walk found
                # Check if it is reduced (no immediate backtracking)
                # The DFS construction ensures no immediate backtracking if we skip the prev edge
                # But we need to ensure the *entire* walk is reduced?
                # Actually, "reduced" usually means no $e, e$ consecutive.
                # The DFS skipping `prev_e` ensures that.
                coeff_vectors.add(tuple(counts))
                # Do not extend further if we want only simple cycles?
                # No, we need all reduced closed walks.
                # A reduced closed walk can be extended?
                # If we extend a closed walk, it's no longer a *simple* cycle, but it is a closed walk.
                # However, if it contains a shorter closed sub-walk, it's not "primitive".
                # We only need to check primitive reduced closed walks?
                # If a walk is a concatenation of two closed walks, its voltage sum is the sum of the two.
                # If both are non-zero, the sum could be zero.
                # So we must check ALL reduced closed walks, not just primitive ones.
                # But the number of reduced closed walks grows exponentially.
                # For L=15, it's ~32k. This is fine.
                pass

            if l >= max_walk_len:
                continue

            # Extend the walk
            for next_v in range(4):
                if next_v == v:
                    continue
                e_idx = edge_map[(v, next_v)]
                if e_idx == prev_e:
                    continue # No backtracking
                
                new_counts = counts.copy()
                new_counts[e_idx] += 1
                
                stack.append((next_v, e_idx, l + 1, new_counts))

    # Remove the zero vector if present (length 0)
    if (0,0,0,0,0,0) in coeff_vectors:
        coeff_vectors.remove((0,0,0,0,0,0))
        
    # Convert to list for faster iteration
    coeff_list = list(coeff_vectors)
    
    # Search for a valid voltage assignment
    # Try primes m starting from lower bounds
    # Moore bound for (3, 16) is 510. 4*m >= 510 => m >= 128.
    # Primes: 131, 137, 139, 149, ...
    
    best_n = float('inf')
    best_edges = None
    best_m = None
    best_voltages = None
    
    # List of primes to try
    primes = [131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]
    
    for m in primes:
        if time.time() > deadline:
            break
            
        # Try random assignments for this m
        # Limit attempts per m to avoid spending too much time
        max_attempts = 50
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Random voltages in 1..m-1 (avoid 0 to help with 2*ve != 0)
            voltages = [random.randint(1, m-1) for _ in range(6)]
            
            valid = True
            for c in coeff_list:
                s = sum(c[i] * voltages[i] for i in range(6))
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                # Construct the graph
                n = 4 * m
                edges = []
                # Vertices are (v, i) where v in 0..m-1, i in 0..3
                # Index: v * 4 + i
                for u in range(4):
                    for w in range(4):
                        if u == w:
                            continue
                        e_idx = edge_map[(u, w)]
                        vol = voltages[e_idx]
                        for v in range(m):
                            u_node = v * 4 + u
                            w_node = ((v + vol) % m) * 4 + w
                            # Add edge (u_node, w_node)
                            # Ensure no duplicate edges
                            if u_node < w_node:
                                edges.append((u_node, w_node))
                            else:
                                edges.append((w_node, u_node))
                
                # Deduplicate edges just in case
                edges = list(set(edges))
                
                # Verify girth
                # Simple BFS girth check
                girth = _compute_girth(n, edges)
                
                if girth >= g:
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                        best_m = m
                        best_voltages = voltages
                        # If we found a very good one, we can stop
                        if n <= 936:
                            return n, edges
                else:
                    # Should not happen if logic is correct, but safety check
                    pass

    if best_edges is not None:
        return best_n, best_edges

    # Fallback to LUW
    return _construct_luw(k, g)

def _construct_luw(k, g):
    # Copy of the seed logic
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

def _compute_girth(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    girth = float('inf')
    for i in range(n):
        # BFS from i
        dist = [-1] * n
        parent = [-1] * n
        queue = [i]
        dist[i] = 0
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < girth:
                        girth = cycle_len
    return girth
