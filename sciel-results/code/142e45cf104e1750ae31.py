"""Construct a 3-regular graph with girth at least g on as few vertices as possible."""
import numpy as np
from collections import deque, defaultdict
import time

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses a deterministic cycle-avoiding greedy algorithm with non-backtracking walk tracking.
    """
    if k != 3:
        return _construct_general(k, g)
    
    # For k=3, g=16, use the non-backtracking walk greedy approach
    return _nb_walk_greedy(k, g)

def _nb_walk_greedy(k, g, max_time=110):
    """
    Deterministic greedy construction using non-backtracking walks.
    
    Strategy:
    1. Start with n vertices (try increasing n from Moore bound)
    2. Build edges greedily, maintaining non-backtracking walks of length g/2-1
    3. When adding edge (u,v), check if it creates a cycle shorter than g
       by checking if any non-backtracking walk from u of length g/2-1 reaches v
    4. Select edges to minimize collisions
    
    For g=16, we track non-backtracking walks of length 7 (g/2 - 1).
    A cycle of length < 16 exists iff there's a non-backtracking walk of length
    g/2-1 = 7 from some vertex back to itself (for even g) or similar.
    
    Actually, for girth >= g, we need no cycle of length < g.
    For even g=16: no cycle of length 4,6,8,10,12,14.
    A cycle of length 2l exists iff there's a non-backtracking walk of length l-1 
    that when extended creates a cycle.
    
    Simpler approach: When adding edge (u,v), check if there's a path of length 
    < g-1 between u and v. If so, adding (u,v) creates a cycle of length <= g-1.
    
    We optimize this by maintaining BFS distances up to depth g/2-1 from each vertex.
    """
    
    start_time = time.time()
    
    # Moore bound for (3,16)
    half_g = g // 2  # 8 for g=16
    # For even g: n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * sum_{i=0}^{7} 2^i = 2*(2^8-1) = 510
    moore_bound = 2 * sum((k-1)**i for i in range(half_g))
    if moore_bound % 2 != 0:
        moore_bound += 1
    
    # Try n starting from moore_bound
    for n in range(moore_bound, moore_bound + 500, 2):
        if time.time() - start_time > max_time - 5:
            break
        result = _try_construct_nb(n, k, g, time_limit=max_time - 5)
        if result is not None:
            return result
    
    # Fallback: use a larger known-good construction
    return _construct_fallback(k, g)

def _try_construct_nb(n, k, g, time_limit=100):
    """
    Try to construct a k-regular graph on n vertices with girth >= g
    using non-backtracking walk collision avoidance.
    
    Key insight: To avoid cycles of length < g, when adding edge (u,v),
    we need to ensure there's no path of length < g-1 between u and v.
    For g=16, this means no path of length < 15.
    
    We maintain for each vertex, the set of vertices reachable by non-backtracking
    walks of each length up to g/2-1.
    
    Actually, a simpler and correct approach:
    - Maintain adjacency list
    - When considering edge (u,v), do a limited BFS from u to see if v is reachable
      in < g-1 steps. If yes, skip this edge.
    - This is O(n * degree^(g-2)) in worst case, but with pruning it's manageable.
    
    For better efficiency, we precompute BFS trees and use them.
    """
    
    import time
    start = time.time()
    
    adj = [[] for _ in range(n)]
    edges = []
    
    # Precompute all candidate edge pairs, sorted by some heuristic
    # We'll use a deterministic ordering: by sum of indices, then by index
    # This gives a consistent, deterministic order
    
    # Actually, let's use a smarter approach:
    # Process vertices in order, and for each vertex, try to connect it to
    # other vertices that are "far away" in the current graph
    
    # Step 1: Create initial edges to form a 3-regular graph structure
    # We'll use a greedy matching approach
    
    # Generate all possible edges
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    
    # Deterministic shuffle using a simple hash-based ordering
    # We'll sort by (i+j, i) to get a deterministic but mixed order
    all_edges.sort(key=lambda e: (e[0] + e[1], e[0], e[1]))
    
    # For better results, let's use a different strategy:
    # Build the graph in phases
    # Phase 1: Create a spanning tree-like structure
    # Phase 2: Add remaining edges to complete 3-regularity
    
    # Actually, let's just do the greedy with collision checking
    # but optimize the collision check
    
    # Precompute: for each vertex, maintain a bitset or set of vertices
    # reachable within distance d for d = 0, 1, ..., g-2
    
    # For efficiency, we'll use a different approach:
    # Maintain a list of "forbidden" pairs based on current graph structure
    
    def check_no_short_cycle(u, v, adj, max_dist):
        """Check if there's a path from u to v of length <= max_dist in adj."""
        if u == v:
            return True  # distance 0
        if max_dist == 0:
            return False
        
        # BFS from u, limited to max_dist
        visited = [False] * len(adj)
        visited[u] = True
        current_level = [u]
        
        for dist in range(1, max_dist + 1):
            next_level = []
            for node in current_level:
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        if neighbor == v:
                            return True
                        visited[neighbor] = True
                        next_level.append(neighbor)
            if not next_level:
                break
            current_level = next_level
        
        return False
    
    # Greedy edge addition
    # We'll try edges in a deterministic order
    # To make it deterministic but effective, we use a specific ordering
    
    # Let's use a round-robin style: process vertices in order,
    # and for each vertex, try to connect to the farthest available vertices
    
    # Better: use a priority queue-like approach
    # For each vertex with degree < k, find the best candidate neighbor
    
    # For determinism, we'll process in a fixed order
    # Let's try multiple deterministic orderings and pick the best
    
    # Order 1: natural order
    # Order 2: reverse order  
    # Order 3: interleaved (even then odd)
    
    best_result = None
    
    # Try a few deterministic orderings
    orderings = [
        [(i, j) for i in range(n) for j in range(i+1, n)],
        [(i, j) for i in range(n-1, -1, -1) for j in range(i+1, n)],
        [(i, j) for i in range(0, n, 2) for j in range(i+1, n)] + 
        [(i, j) for i in range(1, n, 2) for j in range(i+1, n)],
    ]
    
    for edge_order in orderings:
        if time.time() - start > time_limit:
            break
        
        adj = [[] for _ in range(n)]
        edges = []
        edge_set = set()
        
        # Process edges in the given order
        for u, v in edge_order:
            if time.time() - start > time_limit:
                break
            
            if len(adj[u]) >= k or len(adj[v]) >= k:
                continue
            if (u, v) in edge_set or (v, u) in edge_set:
                continue
            if v in adj[u]:
                continue
            
            # Check if adding this edge creates a cycle shorter than g
            # We need to check if there's a path of length < g-1 between u and v
            # For g=16, check paths of length 1 to 14
            # But we can optimize: if u and v are not adjacent and there's no
            # path of length < g-1, it's safe
            
            # For efficiency, first check if they're already close
            # Use a quick check: if they share a neighbor, distance is 2
            if set(adj[u]) & set(adj[v]):
                continue  # distance 2, adding edge creates 3-cycle
            
            # For longer distances, do a limited BFS
            # But this is expensive. Let's use a heuristic:
            # Only do the full check if the graph is not too dense yet
            
            if len(edges) < n * k // 2:
                # Do the full check
                if check_no_short_cycle(u, v, adj, g - 2):
                    continue
            
            # Add the edge
            adj[u].append(v)
            adj[v].append(u)
            edges.append((u, v))
            edge_set.add((u, v))
            edge_set.add((v, u))
        
        # Check if we got a valid 3-regular graph
        valid = True
        for i in range(n):
            if len(adj[i]) != k:
                valid = False
                break
        
        if valid:
            # Verify girth >= g
            if _verify_girth(adj, g):
                best_result = (n, edges)
                break
    
    if best_result is not None:
        return best_result
    
    return None

def _verify_girth(adj, g):
    """Verify that the graph has girth >= g using BFS from each vertex."""
    n = len(adj)
    
    for start in range(n):
        # BFS from start, tracking distance and parent
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
                elif neighbor != parent[node]:
                    # Found a cycle
                    cycle_len = dist[node] + dist[neighbor] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _construct_fallback(k, g):
    """Fallback construction using Lazebnik-Ustimenko-Woldar graphs."""
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

def _construct_general(k, g):
    """General construction for other values of k and g."""
    return _construct_fallback(k, g)
